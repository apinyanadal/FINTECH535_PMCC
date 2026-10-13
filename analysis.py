"""
analysis.py -- every number the page quotes is computed here, so the write-up can be checked
against code (the commentary must match the charts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = {"BH": "Buy & hold", "CC": "Covered call", "PMCC": "Poor man's covered call"}


def book_metrics(res: dict, cfg) -> dict:
    led, b, wl = res["ledger"], res["blotter"], res["weeks_log"]
    start, M = res["start_cash"], cfg.MULTIPLIER
    S0 = float(res["close"].loc[res["t0"]])
    out = {}
    for bk in ("BH", "CC", "PMCC"):
        L = led[led["book"] == bk].reset_index(drop=True)
        nav = L["nav"].to_numpy()
        r = nav[1:] / nav[:-1] - 1
        n = len(r)
        total = nav[-1] / nav[0] - 1
        gmrr = (nav[-1] / nav[0]) ** (252 / n) - 1 if n else np.nan
        vol = float(np.std(r, ddof=1) * np.sqrt(252)) if n > 1 else np.nan
        bb = b[b["book"] == bk]
        interest = float(L["interest_cum"].iloc[-1])
        pnl = float(nav[-1] - start)
        if bk == "PMCC":
            deployed = float(bb[bb["side"] == "BUY"]["price"].iloc[0] * M)
        else:
            deployed = cfg.SHARES * S0
        m = dict(name=NAMES[bk], start=start, nav_end=float(nav[-1]), pnl=pnl, ret=float(total),
                 gmrr=float(gmrr), vol=vol, sharpe=float((gmrr - cfg.RISK_FREE) / vol) if vol else np.nan,
                 max_dd=float((nav / np.maximum.accumulate(nav) - 1).min()), interest=interest,
                 deployed=deployed, ret_on_deployed=(pnl - interest) / deployed, n_days=n + 1)
        if bk != "BH":
            m["premium"] = float(bb[bb["side"] == "SELL"]["cash_delta"].sum())
            m["buybacks"] = float(-bb[bb["side"] == "BUYBACK"]["cash_delta"].sum())
            m["call_pnl"] = m["premium"] - m["buybacks"]
        out[bk] = m
    # PMCC-specific
    p = out["PMCC"]
    leap_row = b[(b["book"] == "PMCC") & (b["side"] == "BUY")].iloc[0]
    Lp = led[led["book"] == "PMCC"].iloc[-1]
    p["leap_cost"] = float(leap_row["price"] * M)
    p["leap_end"] = float(Lp["leap_mark"] * M)
    p["leap_pnl"] = p["leap_end"] - p["leap_cost"]
    p["leap_strike"] = float(leap_row["strike"])
    p["capital_ratio"] = p["deployed"] / out["CC"]["deployed"]
    out["CC"]["stock_pnl"] = cfg.SHARES * (float(Lp["spot"]) - S0)
    return out


def spread_cost(res: dict, cfg) -> dict:
    """What mid fills hide: half the quoted spread on every option fill, plus closing the LEAP at the bid
    at the end. Fills priced from MID_PRICE have no spread to measure and are counted separately."""
    b, opts, M = res["blotter"], res["opts"], cfg.MULTIPLIER
    out = {}
    for bk in ("CC", "PMCC"):
        f = b[(b["book"] == bk) & b["side"].isin(["BUY", "SELL"]) & (b["kind"] == "OPT")]
        known = f[f["src"] == "bidask"]
        half = float(((known["ask"] - known["bid"]) / 2 * M * known["qty"]).sum())
        out[bk] = dict(half_spread=half, n_known=int(len(known)), n_unknown=int(len(f) - len(known)))
    leap = b[(b["book"] == "PMCC") & (b["side"] == "BUY")].iloc[0]
    g = opts.loc[leap["instrument"]]
    g = g[(g["src"] == "bidask") & (g.index <= res["end"])]
    exit_half = float((g["ask"].iloc[-1] - g["bid"].iloc[-1]) / 2 * M) if len(g) else 0.0
    out["PMCC"]["half_spread"] += exit_half
    out["PMCC"]["leap_exit_half_spread"] = exit_half
    return out


def week_stats(res: dict) -> dict:
    wl = res["weeks_log"]
    t = wl[wl["status"] == "TRADED"]
    a = t[t["outcome"] == "ASSIGNED"]
    e = t[t["outcome"] == "EXPIRED"]
    return dict(n_weeks=int(len(wl)), n_traded=int(len(t)), n_skipped=int((wl["status"] == "SKIPPED").sum()),
                n_assigned=int(len(a)), n_expired=int(len(e)),
                avg_delta=float(t["delta"].mean()) if len(t) else None,
                avg_premium=float(t["premium"].mean() * 100) if len(t) else None,
                avg_prem_pct=float((t["premium"] / t["spot_entry"]).mean() * 100) if len(t) else None,
                pnl_expired=float(e["call_pnl"].sum()), pnl_assigned=float(a["call_pnl"].sum()),
                worst_week=(t.loc[t["call_pnl"].idxmin(), "week"], float(t["call_pnl"].min())) if len(t) else None)


def benchmarks(res: dict) -> dict:
    c = res["close"]
    t0, t1 = res["t0"], res["end"]
    out = dict(amzn=float(c.loc[t1] / c.loc[t0] - 1), amzn_t0=float(c.loc[t0]), amzn_t1=float(c.loc[t1]))
    spx = res.get("spx")
    if spx is not None and len(spx) and t0 in spx.index and t1 in spx.index:
        out["spx"] = float(spx.loc[t1] / spx.loc[t0] - 1)
    return out


def pnl_split(res: dict) -> pd.DataFrame:
    """PMCC P&L by source, cumulative per day: LEAP (mark - cost), short calls (premium - buyback - open mark),
    interest. Plus the CC stock leg for comparison. Each line is a mark-to-market, not only realised."""
    led, b = res["ledger"], res["blotter"]
    M = 100
    p = led[led["book"] == "PMCC"].reset_index(drop=True)
    leap = b[(b["book"] == "PMCC") & (b["side"] == "BUY")].iloc[0]
    rows = []
    calls = b[(b["book"] == "PMCC") & (b["side"].isin(["SELL", "BUYBACK"]))]
    S0 = float(res["close"].loc[res["t0"]])
    for r in p.itertuples():
        realised = float(calls[calls["date"] <= r.date]["cash_delta"].sum())
        open_mv = -M * r.short_mark if r.short_mark is not None and not pd.isna(r.short_mark) else 0.0
        rows.append(dict(date=r.date, leap=M * (r.leap_mark - leap["price"]), calls=realised + open_mv,
                         interest=r.interest_cum, total=r.nav - res["start_cash"], cc_stock=100 * (r.spot - S0)))
    return pd.DataFrame(rows)
