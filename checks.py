"""
checks.py -- prove the books are internally consistent before anything is published.
run_pipeline.py refuses to build the site if any check fails. Where possible a check recomputes
the number through a different code path from the engine (HW1/HW2 habit: verify two ways).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bs import call_delta, implied_vol, years

TOL = 1e-6


def run_checks(res: dict, raw: dict, cfg) -> list:
    b, led, wl = res["blotter"], res["ledger"], res["weeks_log"]
    opts, close, contracts = res["opts"], res["close"], res["contracts"]
    M, start = cfg.MULTIPLIER, res["start_cash"]
    out = []

    def rec(name, ok, detail=""):
        out.append(dict(name=name, ok=bool(ok), detail=str(detail)))

    # raw quotes, read straight from the fetched tape (not the normalized frame)
    rq = raw["options"].copy()
    rq["date"] = pd.to_datetime(rq["date"]).dt.date
    rq = rq.drop_duplicates(["ric", "date"], keep="last").set_index(["ric", "date"])

    # 1 cash arithmetic on every blotter row
    def expected(r):
        if r["kind"] == "STK":
            return (-1 if r["side"] == "BUY" else 1) * r["qty"] * r["price"]
        sign = {"BUY": -1, "SELL": 1, "BUYBACK": -1, "EXPIRE": 0}[r["side"]]
        return sign * r["qty"] * M * r["price"]
    bad = [i for i, r in b.iterrows() if abs(r["cash_delta"] - expected(r)) > TOL]
    rec("every blotter row: cash change = +/- qty x price (x100 for options)", not bad, f"bad rows {bad[:5]}")

    # 2 every option fill equals the mid of that day's raw quote and sits inside bid/ask
    ok2, det2 = True, ""
    for r in b[b["side"].isin(["BUY", "SELL"]) & (b["kind"] == "OPT")].itertuples():
        try:
            q = rq.loc[(r.instrument, r.date)]
        except KeyError:
            ok2, det2 = False, f"{r.instrument} {r.date}: no raw row"
            break
        if r.src == "bidask":
            mid = (q["BID"] + q["ASK"]) / 2
            if not (abs(mid - r.price) < TOL and q["BID"] - TOL <= r.price <= q["ASK"] + TOL):
                ok2, det2 = False, f"{r.instrument} {r.date}: fill {r.price} vs bid/ask {q['BID']}/{q['ASK']}"
                break
        elif not abs(q["MID_PRICE"] - r.price) < TOL:
            ok2, det2 = False, f"{r.instrument} {r.date}: fill {r.price} vs MID_PRICE {q['MID_PRICE']}"
            break
    rec("option fills = mid of that day's raw closing quote, inside the bid/ask", ok2, det2)

    # 3 strike rule recomputed with an independent loop (raw quotes, own delta calc)
    def best_delta(d, expiry, role, target, week=None):
        c = contracts[(contracts["role"] == role) & (contracts["expiry"] == expiry)]
        if week is not None:
            c = c[c["week"] == week]
        S, T, best = float(close.loc[d]), years(d, expiry), None
        for r in c.itertuples():
            if (r.ric, d) not in opts.index or not bool(opts.loc[(r.ric, d), "usable"]):
                continue
            iv = implied_vol(float(opts.loc[(r.ric, d), "mid"]), S, r.strike, T, cfg.RISK_FREE)
            if iv is None:
                continue
            dist = abs(call_delta(S, r.strike, T, iv, cfg.RISK_FREE) - target)
            if best is None or dist < best[0] - 1e-12 or (abs(dist - best[0]) <= 1e-12 and r.strike > best[1]):
                best = (dist, float(r.strike))
        return None if best is None else best[1]
    ok3, det3 = True, ""
    leapbuy = b[(b["book"] == "PMCC") & (b["side"] == "BUY")].iloc[0]
    if best_delta(leapbuy["date"], cfg.LEAP_EXPIRY, "leap", cfg.LEAP_TARGET_DELTA) != leapbuy["strike"]:
        ok3, det3 = False, "LEAP strike"
    for r in b[(b["book"] == "PMCC") & (b["side"] == "SELL")].itertuples():
        k = best_delta(r.date, r.expiry, "weekly", cfg.SHORT_TARGET_DELTA, r.week)
        if k != r.strike:
            ok3, det3 = False, f"week {r.week}: recomputed {k}, booked {r.strike}"
            break
    rec(f"strikes = delta closest to {cfg.LEAP_TARGET_DELTA:.2f} (LEAP) / {cfg.SHORT_TARGET_DELTA:.2f} (weekly), "
        f"recomputed independently", ok3, det3)

    # 4 no-lookahead: each contract was chosen and filled from the same closing snapshot
    sells = b[b["side"].isin(["BUY", "SELL"]) & (b["kind"] == "OPT")]
    rec("every option was chosen from the same close it was filled at (decision date == fill date)",
        bool((sells["decision"] == sells["date"]).all()))

    # 5 expiry rule
    ok5, det5 = True, ""
    for bk in ("CC", "PMCC"):
        bb = b[b["book"] == bk]
        for ric, g in bb[(bb["kind"] == "OPT") & (bb["side"] != "BUY")].groupby("instrument"):
            sell, closer = g[g["side"] == "SELL"], g[g["side"].isin(["EXPIRE", "BUYBACK"])]
            if len(sell) != 1 or len(closer) != 1:
                ok5, det5 = False, f"{bk} {ric}: {len(sell)} sells, {len(closer)} closes"
                break
            c = closer.iloc[0]
            Se, K = float(close.loc[c["date"]]), c["strike"]
            if c["date"] != c["expiry"]:
                ok5, det5 = False, f"{ric}: closed {c['date']} not on expiry {c['expiry']}"
            elif c["side"] == "BUYBACK" and not (Se > K and abs(c["price"] - (Se - K)) < TOL):
                ok5, det5 = False, f"{ric}: bad buyback"
            elif c["side"] == "EXPIRE" and not Se <= K + 1e-9:
                ok5, det5 = False, f"{ric}: expired but close {Se} > K {K}"
            if not ok5:
                break
    rec("each short call: one sale, one close on its expiry day; bought back at intrinsic iff close > strike", ok5, det5)

    # 6 CC and PMCC sold the identical calls
    cols = ["date", "side", "instrument", "price"]
    a = b[(b["book"] == "CC") & (b["kind"] == "OPT")][cols].reset_index(drop=True)
    c = b[(b["book"] == "PMCC") & (b["kind"] == "OPT") & (b["side"] != "BUY")][cols].reset_index(drop=True)
    rec("CC and PMCC traded identical short calls at identical prices", a.equals(c))

    # 7 coverage
    ok7, det7 = True, ""
    leap_k = float(leapbuy["strike"])
    for r in b[(b["book"] == "PMCC") & (b["side"] == "SELL")].itertuples():
        if not (r.strike >= leap_k and r.expiry <= cfg.LEAP_EXPIRY):
            ok7, det7 = False, f"{r.instrument}: K {r.strike} vs LEAP K {leap_k}"
    cc = led[led["book"] == "CC"]
    if ((cc["short_ric"].notna()) & (cc["shares"] < cfg.SHARES)).any():
        ok7, det7 = False, "CC short call open without 100 shares"
    rec("short call always covered (PMCC: LEAP strike <= short strike, LEAP expires later; CC: 100 shares held)", ok7, det7)

    # 8 equal start
    first = led[led["date"] == res["t0"]].set_index("book")["nav"]
    rec("all three books start day 1 with the same NAV (= starting cash)",
        bool(((first - start).abs() < TOL).all()), first.round(2).to_dict())

    # 9 ledger cash = start + cumulative blotter cash + cumulative interest (pandas path)
    ok9 = True
    for r in led.itertuples():
        cum = b[(b["book"] == r.book) & (b["date"] <= r.date)]["cash_delta"].sum()
        if abs(r.cash - (start + cum + r.interest_cum)) > TOL:
            ok9 = False
            break
    rec("ledger cash = start + cumulative blotter cash + accrued interest (every day, every book)", ok9)

    # 10 independent NAV replay from blotter positions and raw marks
    dmax = 0.0
    for r in led.itertuples():
        sub = b[(b["book"] == r.book) & (b["date"] <= r.date)]
        stk = sub[sub["kind"] == "STK"]
        sh = stk.loc[stk["side"] == "BUY", "qty"].sum() - stk.loc[stk["side"] == "SELL", "qty"].sum()
        mv = sh * r.spot
        opt = sub[sub["kind"] == "OPT"]
        for ric, g in opt.groupby("instrument"):
            pos = g.loc[g["side"] == "BUY", "qty"].sum() - g.loc[g["side"] == "SELL", "qty"].sum() + \
                g.loc[g["side"].isin(["BUYBACK", "EXPIRE"]), "qty"].sum()
            if pos != 0:
                hist = opts.loc[ric]
                hist = hist[hist["usable"] & (hist.index <= r.date)]
                mv += pos * M * float(hist["mid"].iloc[-1])
        dmax = max(dmax, abs(r.cash + mv - r.nav))
    rec("independent NAV replay matches the ledger (max abs diff <= 1e-6)", dmax <= TOL, f"max diff {dmax:.2e}")

    # 11 week log reconciles with blotter
    n_sell = int(((b["book"] == "PMCC") & (b["side"] == "SELL")).sum())
    n_tr = int((wl["status"] == "TRADED").sum())
    n_bb = int(((b["book"] == "PMCC") & (b["side"] == "BUYBACK")).sum())
    n_as = int((wl["outcome"] == "ASSIGNED").sum())
    rec("week log reconciles with the blotter (traded = calls sold; assigned = buybacks)",
        n_sell == n_tr and n_bb == n_as, f"sold {n_sell}, traded {n_tr}, buybacks {n_bb}, assigned {n_as}")

    # info
    stale = led[led["book"] == "PMCC"]
    out.append(dict(name="INFO: PMCC days marked with a stale price (no usable quote that day)", ok=True,
                    detail=f"LEAP {int(stale['leap_stale'].sum())}, short call {int(stale['short_stale'].sum())} "
                           f"of {len(stale)} days"))
    fills = b[b["side"].isin(["BUY", "SELL"]) & (b["kind"] == "OPT") & (b["book"] == "PMCC")]
    out.append(dict(name="INFO: option fills priced from MID_PRICE because bid/ask was missing", ok=True,
                    detail=f"{int((fills['src'] == 'mid_price').sum())} of {len(fills)}"))
    return out


def print_report(checks: list) -> bool:
    ok = True
    print("\n=== CHECKS ===")
    for c in checks:
        ok &= c["ok"]
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
    print("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***")
    return ok
