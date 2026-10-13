"""
backtest.py -- three books on the same AMZN tape, same starting cash, same calendar.

  BH    buy 100 shares on day 1 and hold.
  CC    covered call: buy 100 shares on day 1, sell the weekly call every week.
  PMCC  poor man's covered call: buy 1 Jan-2027 LEAP call on day 1, sell the SAME weekly call every week.

Rules (the website's Rules panel is generated from config.py with the same wording):
 R1 DECIDE    Each choice is made from ONE closing snapshot (stock close + option closing quotes) and filled
              at that same snapshot, as in HW2 (decide and fill on the same bar). Nothing after that close is used.
 R2 LEAP      Day 1 close: among Jan-2027 calls with a usable price, pick the strike whose Black-Scholes delta
              (IV backed out of its own mid) is closest to LEAP_TARGET_DELTA. Buy 1 at that closing mid.
              Held to the end of the window, marked daily at mid.
 R3 SHORT     Close of each week's first session: among calls expiring at the end of that week with a usable
              price, pick the strike whose delta is closest to SHORT_TARGET_DELTA and sell 1 at that closing
              mid. CC and PMCC sell the identical contract at the identical price.
 R4 FILL      Mid = (bid + ask) / 2 of that day's closing quote (MID_PRICE fallback, flagged). No usable
              price on the execution day -> no call that week (both books stay uncovered; logged).
 R5 EXPIRY    Expiry-day close <= strike: the call EXPIRES worthless, keep the premium.
              Expiry-day close  > strike: BUY BACK at intrinsic (close - strike) at that close. Reported as
              'assigned'. Same as having the shares called away at the strike and re-buying them at the
              close, so CC and PMCC are treated identically and the PMCC never ends up short stock.
 R6 CASH      All books start with the same cash = cost of 100 shares on day 1. Idle cash earns RISK_FREE
              (negative cash pays it), accrued daily on calendar days. No margin loan, no fees.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bs import call_delta, implied_vol, years
from tape import build_weeks, normalize_options

BOOKS = ("BH", "CC", "PMCC")


def load_tape(raw: dict, cfg):
    close = raw["stock"].copy()
    close.index = pd.Index([pd.Timestamp(d).date() for d in close.index])
    close = close.sort_index()
    opts = normalize_options(raw["options"], cfg.MAX_REL_SPREAD, cfg.USE_MID_PRICE_FALLBACK)
    contracts = raw["contracts"].copy()
    contracts["expiry"] = [pd.Timestamp(d).date() for d in contracts["expiry"]]
    spx = raw.get("spx")
    if spx is not None and len(spx):
        spx = spx.copy()
        spx.index = pd.Index([pd.Timestamp(d).date() for d in spx.index])
    return close, opts, contracts, spx


def quote(opts, ric, d):
    try:
        q = opts.loc[(ric, d)]
    except KeyError:
        return None
    return q if bool(q["usable"]) else None


def pick_by_delta(opts, contracts, close, d, expiry, target, cfg, week=None, role="weekly"):
    """Score every candidate on day d's close. Returns (chosen dict, table of all candidates)."""
    c = contracts[(contracts["role"] == role) & (contracts["expiry"] == expiry)]
    if week is not None:
        c = c[c["week"] == week]
    S, T = float(close.loc[d]), years(d, expiry)
    cand = []
    for r in c.itertuples():
        q = quote(opts, r.ric, d)
        if q is None:
            continue
        iv = implied_vol(float(q["mid"]), S, r.strike, T, cfg.RISK_FREE, cfg.DIV_YIELD)
        if iv is None:
            continue
        cand.append(dict(ric=r.ric, strike=float(r.strike), mid=float(q["mid"]), iv=iv,
                         delta=call_delta(S, r.strike, T, iv, cfg.RISK_FREE, cfg.DIV_YIELD)))
    if not cand:
        return None, pd.DataFrame()
    tab = pd.DataFrame(cand).sort_values("strike").reset_index(drop=True)
    tab["dist"] = (tab["delta"] - target).abs()
    best = tab.sort_values(["dist", "strike"], ascending=[True, False]).iloc[0].to_dict()
    best.update(spot=S, decision=d)
    return best, tab


def run_backtest(raw: dict, cfg) -> dict:
    close, opts, contracts, spx = load_tape(raw, cfg)
    weeks = build_weeks(close.index, cfg.WINDOW_START, cfg.WINDOW_END)
    if weeks.empty:
        raise RuntimeError("no complete week inside the window")
    M = cfg.MULTIPLIER

    # ------------------------------------------------------------------ day 1: LEAP + stock
    w0 = weeks.iloc[0]
    t0 = w0["first"]
    S0 = float(close.loc[t0])
    start_cash = math.ceil(cfg.SHARES * S0 / 100.0) * 100.0
    leap, leap_tab = pick_by_delta(opts, contracts, close, t0, cfg.LEAP_EXPIRY,
                                   cfg.LEAP_TARGET_DELTA, cfg, role="leap")
    if leap is None:
        raise RuntimeError(f"no LEAP candidate with a usable price on {t0}: cannot open the PMCC")
    lq = quote(opts, leap["ric"], t0)

    blot = []

    def add(book, d, kind, side, instrument, qty, price, cash, rule, note, week=None, strike=None, expiry=None,
            bid=None, ask=None, src=None, decision=None):
        blot.append(dict(book=book, date=d, week=week, kind=kind, side=side, instrument=instrument, qty=qty,
                         price=price, cash_delta=float(cash), rule=rule, note=note, strike=strike, expiry=expiry,
                         bid=bid, ask=ask, src=src, decision=decision))

    for b in ("BH", "CC"):
        add(b, t0, "STK", "BUY", cfg.TICKER, cfg.SHARES, S0, -cfg.SHARES * S0, "R6",
            f"buy {cfg.SHARES} sh at the {t0} close")
    lmid = float(lq["mid"])
    add("PMCC", t0, "OPT", "BUY", leap["ric"], cfg.CONTRACTS, lmid, -cfg.CONTRACTS * M * lmid, "R2",
        f"LEAP K={leap['strike']:g} exp {cfg.LEAP_EXPIRY}: delta {leap['delta']:.2f} at the {t0} close "
        f"(closest to {cfg.LEAP_TARGET_DELTA:.2f}); bought at that closing mid",
        strike=leap["strike"], expiry=cfg.LEAP_EXPIRY, bid=lq["bid"], ask=lq["ask"], src=lq["src"],
        decision=t0)

    # ------------------------------------------------------------------ weekly short calls
    log = []
    for w in weeks.itertuples():
        L = dict(week=w.week, first=w.first, expiry=w.expiry, status="SKIPPED", reason="", strike=None, delta=None, iv=None, ric=None,
                 spot_entry=float(close.loc[w.first]), premium=None, bid=None, ask=None, src=None,
                 expiry_close=float(close.loc[w.expiry]), outcome=None, buyback=None, call_pnl=0.0)
        if w.first >= w.expiry:
            L["reason"] = "one-session week: no time to sell a weekly"
            log.append(L)
            continue
        best, _ = pick_by_delta(opts, contracts, close, w.first, w.expiry, cfg.SHORT_TARGET_DELTA, cfg, week=w.week)
        if best is None:
            L["reason"] = f"no weekly call with a usable price on {w.first}"
            log.append(L)
            continue
        L.update(strike=best["strike"], delta=best["delta"], iv=best["iv"], ric=best["ric"])
        q = quote(opts, best["ric"], w.first)
        mid = float(q["mid"])
        prem = cfg.CONTRACTS * M * mid
        L.update(status="TRADED", premium=mid, bid=q["bid"], ask=q["ask"], src=q["src"])
        for b in ("CC", "PMCC"):
            add(b, w.first, "OPT", "SELL", best["ric"], cfg.CONTRACTS, mid, +prem, "R3/R4",
                f"K={best['strike']:g}: delta {best['delta']:.2f} at the {w.first} close (closest to "
                f"{cfg.SHORT_TARGET_DELTA:.2f}); sold at that closing mid", week=w.week,
                strike=best["strike"], expiry=w.expiry, bid=q["bid"], ask=q["ask"], src=q["src"], decision=w.first)
        Se = L["expiry_close"]
        if Se > best["strike"] + 1e-9:
            intr = Se - best["strike"]
            for b in ("CC", "PMCC"):
                add(b, w.expiry, "OPT", "BUYBACK", best["ric"], cfg.CONTRACTS, intr, -cfg.CONTRACTS * M * intr, "R5",
                    f"expiry close {Se:.2f} > K {best['strike']:g}: bought back at intrinsic {intr:.2f} ('assigned')",
                    week=w.week, strike=best["strike"], expiry=w.expiry)
            L.update(outcome="ASSIGNED", buyback=intr, call_pnl=M * (mid - intr))
        else:
            for b in ("CC", "PMCC"):
                add(b, w.expiry, "OPT", "EXPIRE", best["ric"], cfg.CONTRACTS, 0.0, 0.0, "R5",
                    f"expiry close {Se:.2f} <= K {best['strike']:g}: expired worthless", week=w.week,
                    strike=best["strike"], expiry=w.expiry)
            L.update(outcome="EXPIRED", buyback=0.0, call_pnl=M * mid)
        log.append(L)

    blotter = pd.DataFrame(blot)
    order = {"BUY": 0, "SELL": 0, "BUYBACK": 1, "EXPIRE": 1}
    blotter["_o"] = blotter["side"].map(order)
    blotter = blotter.sort_values(["date", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    weeks_log = pd.DataFrame(log)
    end = weeks.iloc[-1]["expiry"]
    ledger = build_ledger(close, opts, blotter, start_cash, t0, end, leap, cfg)
    return dict(blotter=blotter, weeks_log=weeks_log, ledger=ledger, start_cash=start_cash, t0=t0, end=end,
                leap=leap, leap_tab=leap_tab, close=close, opts=opts, contracts=contracts, spx=spx, weeks=weeks)


# ---------------------------------------------------------------------- ledger
def last_usable(opts, ric, d):
    """Last usable mid on or before d -> (mid, date) or (None, None)."""
    try:
        g = opts.loc[ric]
    except KeyError:
        return None, None
    g = g[g["usable"] & (g.index <= d)]
    if g.empty:
        return None, None
    return float(g["mid"].iloc[-1]), g.index[-1]


def build_ledger(close, opts, blotter, start_cash, t0, end, leap, cfg):
    """One row per session per book. Positions and cash move ONLY on blotter events; interest accrues
    on the previous close's cash for the calendar days in between. Everything else is a mark."""
    M = cfg.MULTIPLIER
    days = [d for d in close.index if t0 <= d <= end]
    out = []
    for b in BOOKS:
        ev = blotter[blotter["book"] == b]
        cash, shares, interest_cum, prev = float(start_cash), 0, 0.0, None
        leap_open, short = None, None          # short = dict(ric, strike, expiry, premium)
        leap_iv, short_iv = None, None
        for d in days:
            if prev is not None and cfg.CASH_EARNS_RF:
                it = cash * cfg.RISK_FREE * (d - prev).days / 365.0
                cash += it
                interest_cum += it
            for e in ev[ev["date"] == d].itertuples():
                cash += e.cash_delta
                if e.kind == "STK":
                    shares += e.qty if e.side == "BUY" else -e.qty
                elif e.side == "BUY":
                    leap_open = dict(ric=e.instrument, strike=e.strike, expiry=e.expiry, cost=e.price)
                elif e.side == "SELL":
                    if short is not None:
                        raise RuntimeError(f"{b}: second short call opened on {d}")
                    short = dict(ric=e.instrument, strike=e.strike, expiry=e.expiry, premium=e.price)
                else:
                    short = None
            S = float(close.loc[d])
            row = dict(book=b, date=d, spot=S, cash=cash, interest_cum=interest_cum, shares=shares,
                       leap_mark=None, leap_stale=False, leap_delta=None, short_ric=None, short_strike=None,
                       short_mark=None, short_stale=False, short_delta=None)
            mv = shares * S
            if leap_open:
                m, md = last_usable(opts, leap_open["ric"], d)
                T = years(d, leap_open["expiry"])
                iv = implied_vol(m, S, leap_open["strike"], T, cfg.RISK_FREE, cfg.DIV_YIELD)
                leap_iv = iv if iv is not None else leap_iv
                row.update(leap_mark=m, leap_stale=md != d,
                           leap_delta=call_delta(S, leap_open["strike"], T, leap_iv, cfg.RISK_FREE) if leap_iv else None)
                mv += M * m
            if short:
                m, md = last_usable(opts, short["ric"], d)
                T = years(d, short["expiry"])
                iv = implied_vol(m, S, short["strike"], T, cfg.RISK_FREE, cfg.DIV_YIELD)
                short_iv = iv if iv is not None else short_iv
                row.update(short_ric=short["ric"], short_strike=short["strike"], short_mark=m, short_stale=md != d,
                           short_delta=call_delta(S, short["strike"], T, short_iv, cfg.RISK_FREE) if short_iv else None)
                mv -= M * m
            else:
                short_iv = None
            row["nav"] = cash + mv
            sd = row["short_delta"] or 0.0
            if b == "BH":
                row["net_shares"] = shares
            elif b == "CC":
                row["net_shares"] = shares - M * sd
            else:
                row["net_shares"] = M * ((row["leap_delta"] or 0.0) - sd)
            out.append(row)
            prev = d
    return pd.DataFrame(out)
