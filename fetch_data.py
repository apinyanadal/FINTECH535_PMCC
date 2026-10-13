"""
fetch_data.py -- pull everything the backtest needs from LSEG into tape.pkl (run AFTER probe.py
and AFTER freezing config.py).    python fetch_data.py

  stock     AMZN.O and .SPX daily closes
  weeklies  for every week: calls expiring that week, strikes on WEEKLY_STRIKE_STEP within WEEKLY_BAND
            x the close of the session BEFORE the week starts; daily rows from that session to expiry
  LEAP      Jan 2027 calls, strikes on LEAP_STRIKE_STEP within LEAP_BAND x the close before day 1;
            daily rows from there to WINDOW_END

Candidates are centred on the PRIOR close, so the candidate set itself cannot look ahead. One RIC per call, cached in raw_cache/: if it stops, just run it again.
"""
from __future__ import annotations

import datetime as dt
import pickle
import sys

import numpy as np
import pandas as pd

import config
import lseg_io
from ric_codec import ric_for_expiry
from tape import build_weeks


def grid(spot, band, step):
    lo, hi = np.ceil(spot * band[0] / step) * step, np.floor(spot * band[1] / step) * step
    return [round(float(k), 2) for k in np.arange(lo, hi + step / 2, step)]


def main():
    cfg = config.load()
    s0 = (pd.Timestamp(cfg.WINDOW_START) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    s1 = (pd.Timestamp(cfg.WINDOW_END) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    lseg_io.open_session()
    try:
        stock = lseg_io.history(cfg.STOCK_RIC, cfg.STOCK_FIELDS, s0, s1, cache_dir=cfg.RAW_CACHE_DIR)
        spx = lseg_io.history(cfg.BENCH_RIC, cfg.STOCK_FIELDS, s0, s1, cache_dir=cfg.RAW_CACHE_DIR)
        if stock.empty:
            sys.exit("No stock data came back. Run the connection test and probe.py first.")
        close = stock["TRDPRC_1"].dropna()
        close.index = close.index.date
        weeks = build_weeks(close.index, cfg.WINDOW_START, cfg.WINDOW_END)
        print(f"{cfg.STOCK_RIC}: {len(close)} sessions; {len(weeks)} weeks in the window")

        rows, contracts = [], []

        def pull(ric, expiry, strike, role, start, end, week=None):
            df = lseg_io.history(ric, cfg.OPTION_FIELDS, start, end, cache_dir=cfg.RAW_CACHE_DIR)
            if df.empty:
                return 0
            d = df.reset_index()
            d["ric"] = ric
            rows.append(d)
            contracts.append(dict(ric=ric, expiry=expiry, strike=strike, role=role, week=week))
            return 1

        # ---- LEAP
        d0 = weeks.iloc[0]["decision"]
        spot0 = float(close.loc[d0])
        ks = grid(spot0, cfg.LEAP_BAND, cfg.LEAP_STRIKE_STEP)
        n = sum(pull(ric_for_expiry(cfg.OPTION_ROOT, cfg.LEAP_EXPIRY, k, expired=cfg.LEAP_RIC_EXPIRED_STYLE),
                     cfg.LEAP_EXPIRY, k, "leap", d0.isoformat(), s1) for k in ks)
        print(f"LEAP {cfg.LEAP_EXPIRY}: {n} of {len(ks)} candidate strikes returned data (spot {spot0:.2f})")
        if n == 0:
            sys.exit("No LEAP data. Check LEAP_RIC_EXPIRED_STYLE against probe.py section 5.")

        # ---- weeklies
        for w in weeks.itertuples():
            spot = float(close.loc[w.decision])
            ks = grid(spot, cfg.WEEKLY_BAND, cfg.WEEKLY_STRIKE_STEP)
            end = (pd.Timestamp(w.expiry) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            n = sum(pull(ric_for_expiry(cfg.OPTION_ROOT, w.expiry, k), w.expiry, k, "weekly",
                         w.decision.isoformat(), end, w.week) for k in ks)
            print(f"  {w.week}: expiry {w.expiry}  spot {spot:8.2f}  {n:2d} of {len(ks)} strikes returned data")

        options = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if "date" not in options.columns and len(options):
            options = options.rename(columns={options.columns[0]: "date"})
        meta = dict(synthetic=False, ticker=cfg.TICKER, fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    window=(cfg.WINDOW_START, cfg.WINDOW_END))
        spx_close = spx["TRDPRC_1"].dropna().rename("close")
        spx_close.index = [pd.Timestamp(d).date() for d in spx_close.index]  # works when empty too
        with open(cfg.TAPE_FILE, "wb") as f:
            pickle.dump(dict(stock=close.rename("close"), spx=spx_close,
                             options=options, contracts=pd.DataFrame(contracts), meta=meta), f)
        print(f"\nWrote {cfg.TAPE_FILE}: {len(close)} stock days, {len(options)} option rows, {len(contracts)} contracts")
        if lseg_io.ERRORS:
            print(f"WARNING: {len(lseg_io.ERRORS)} unexpected errors (timeouts etc.). Run fetch_data.py again; "
                  f"cached answers are reused and only the failed RICs are retried.")
            for r, m in lseg_io.ERRORS[:5]:
                print(f"  {r}: {m}")
    finally:
        lseg_io.close_session()


if __name__ == "__main__":
    main()
