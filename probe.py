"""
probe.py -- run FIRST, with LSEG Workspace open:   python probe.py

Data plumbing only. It never computes a P&L, so reading its output cannot bias the strategy rules.
It answers, in order:
  1. Does the daily stock pull work, and over what dates?
  2. What strike spacing do AMZN weeklies use (near the money), early and late in the window?
  3. Which daily fields come back for expired weeklies (BID / ASK / MID_PRICE / TRDPRC_1)?
  4. What is the most recent weekly expiry the expired-options archive returns (=> WINDOW_END)?
  5. Which RIC format works for the still-live January 2027 LEAP, and what does it cover?
Paste the whole output back.
"""
from __future__ import annotations

import datetime as dt
import traceback

import numpy as np
import pandas as pd

import config as cfg
import lseg_io
from ric_codec import ric_for_expiry

TODAY = dt.date.today()


def hr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def coverage(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "no rows"
    parts = [f"{len(df)} rows {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}"]
    for f in ("BID", "ASK", "MID_PRICE", "TRDPRC_1"):
        if f in df.columns:
            parts.append(f"{f} {df[f].notna().mean() * 100:.0f}%")
    return " | ".join(parts)


def week_table(stock: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"date": stock.index.date})
    d["key"] = [x.isocalendar()[:2] for x in d["date"]]
    g = d.groupby("key")["date"]
    return pd.DataFrame({"first": g.min(), "last": g.max()}).reset_index(drop=True)


def main():
    lseg_io.open_session()
    try:
        # ---------------------------------------------------------------- 1. stock
        hr("1. daily stock history")
        start = (pd.Timestamp(cfg.WINDOW_START) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        stock = lseg_io.history(cfg.STOCK_RIC, ["TRDPRC_1"], start, TODAY.isoformat())
        if stock.empty:
            print(f"  {cfg.STOCK_RIC}: NO DATA. Stop here and paste this output.")
            return
        close = stock["TRDPRC_1"].dropna()
        print(f"  {cfg.STOCK_RIC}: {len(close)} sessions {close.index.min():%Y-%m-%d} .. {close.index.max():%Y-%m-%d}")
        print(f"  first close {close.iloc[0]:.2f}   last close {close.iloc[-1]:.2f}   min {close.min():.2f}   max {close.max():.2f}")
        spx = lseg_io.history(cfg.BENCH_RIC, ["TRDPRC_1"], start, TODAY.isoformat())
        print(f"  {cfg.BENCH_RIC}: {coverage(spx)}")
        weeks = week_table(close.to_frame())

        def spot_before(day):
            s = close[close.index.date < day]
            return float(s.iloc[-1])

        # ---------------------------------------------------------------- 2+3. strike grid and fields
        hr("2-3. weekly strike grid and daily field coverage (one RIC per call)")
        steps = {}
        for label, target in (("early", pd.Timestamp(cfg.WINDOW_START).date() + dt.timedelta(days=4)),
                              ("late", TODAY - dt.timedelta(days=21))):
            w = weeks[(weeks["last"] >= target)].iloc[0]
            expiry, first = w["last"], w["first"]
            spot = spot_before(first)
            base = int(np.floor(spot))
            print(f"\n  [{label}] expiry {expiry} (week starts {first}), spot before week {spot:.2f}")
            found = []
            for k in [base + i for i in range(-2, 9)] + [base + 0.5, base + 1.5, base + 2.5]:
                ric = ric_for_expiry(cfg.OPTION_ROOT, expiry, float(k))
                df = lseg_io.history(ric, cfg.OPTION_FIELDS, (pd.Timestamp(first) - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                                     (pd.Timestamp(expiry) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                ok = not df.empty
                if ok:
                    found.append(float(k))
                print(f"    {ric:<24} {'DATA ' if ok else 'missing'}  {coverage(df) if ok else ''}")
            found = sorted(found)
            if len(found) >= 2:
                st = min(np.diff(found))
                steps[label] = st
                print(f"    -> listed strikes found: {found}  smallest spacing {st:g}")
            else:
                print("    -> fewer than 2 strikes found; paste this output")

        # ---------------------------------------------------------------- 4. archive cutoff
        hr("4. most recent weekly expiry in the expired-options archive")
        latest = None
        done = weeks[(weeks["last"] < TODAY) & weeks["last"].map(lambda d: d.weekday() in (3, 4))]   # finished weeks only
        recent = done.tail(6).iloc[::-1]
        for w in recent.itertuples():
            spot = spot_before(w.first)
            base = int(np.floor(spot))
            hits = 0
            for k in (base - 1, base, base + 1, base + 2, base + 3, base + 5):
                ric = ric_for_expiry(cfg.OPTION_ROOT, w.last, float(k))
                df = lseg_io.history(ric, cfg.OPTION_FIELDS, w.first.isoformat(),
                                     (pd.Timestamp(w.last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                hits += not df.empty
            print(f"  expiry {w.last}: {hits} of 6 probe strikes returned data")
            if hits and latest is None:
                latest = w.last

        # ---------------------------------------------------------------- 5. LEAP
        hr("5. January 2027 LEAP (still live): which RIC format works?")
        d0 = weeks[weeks["first"] >= pd.Timestamp(cfg.WINDOW_START).date()].iloc[0]["first"]
        spot0 = spot_before(d0)
        print(f"  decision-day spot (close before {d0}): {spot0:.2f}")
        lstart = (pd.Timestamp(d0) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        leap_ok = {}
        for k in [5 * round(spot0 * f / 5) for f in (0.70, 0.75, 0.80)]:
            for expired in (False, True):
                ric = ric_for_expiry(cfg.OPTION_ROOT, cfg.LEAP_EXPIRY, float(k), expired=expired)
                df = lseg_io.history(ric, cfg.OPTION_FIELDS, lstart, TODAY.isoformat())
                ok = not df.empty
                leap_ok.setdefault(expired, 0)
                leap_ok[expired] += ok
                print(f"    {ric:<24} ({'expired-style' if expired else 'live-style'})  {'DATA ' if ok else 'missing'}  {coverage(df) if ok else ''}")

        # ---------------------------------------------------------------- summary
        hr("SUGGESTED config.py LINES (review, then freeze)")
        if steps:
            print(f"  WEEKLY_STRIKE_STEP = {min(steps.values()):g}      # smallest weekly spacing seen {steps}")
        if latest:
            print(f'  WINDOW_END = "{latest}"')
        if leap_ok:
            fmt = "live-style (no ^ suffix)" if leap_ok.get(False) else ("expired-style" if leap_ok.get(True) else "NEITHER WORKED")
            print(f"  LEAP RIC format that returned data: {fmt}")
        if lseg_io.ERRORS:
            print(f"\n  WARNING: {len(lseg_io.ERRORS)} unexpected errors (not 'not found'). First 5:")
            for r, m in lseg_io.ERRORS[:5]:
                print(f"    {r}: {m}")
    except Exception:
        traceback.print_exc()
    finally:
        lseg_io.close_session()


if __name__ == "__main__":
    main()
