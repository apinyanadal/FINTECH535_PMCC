"""
tape.py -- shared calendar and quote rules (used by fetch_data.py, backtest.py, checks.py).

Daily data only. A day's option row is that day's CLOSING snapshot.

QUOTE RULE (the only definition of a usable price anywhere in the project):
  * BID and ASK both present, bid > 0, ask >= bid, ask - bid <= MAX_REL_SPREAD x mid  ->  mid = (bid + ask) / 2
  * otherwise, if USE_MID_PRICE_FALLBACK and MID_PRICE > 0                            ->  mid = MID_PRICE (flagged)
  * otherwise no usable price that day (no fill possible; marks carry the last usable mid, flagged stale)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def build_weeks(sessions, window_start: str, window_end: str) -> pd.DataFrame:
    """One row per ISO week inside the window.
    first    = first session of the week (Monday, or Tuesday after a Monday holiday): the short call is
               chosen and sold at this session's close
    expiry   = last session of the week (Friday, or Thursday when Friday is a holiday) = weekly expiry
    decision = the session BEFORE `first`. Only fetch_data.py uses it, to centre the band of candidate
               strikes it requests, so even the candidate list is built from data known before the week."""
    sessions = sorted(pd.to_datetime(pd.Series(list(sessions))).dt.date.unique())
    ws, we = pd.Timestamp(window_start).date(), pd.Timestamp(window_end).date()
    groups: dict = {}
    for d in sessions:
        groups.setdefault(d.isocalendar()[:2], []).append(d)
    rows = []
    for key in sorted(groups):
        ds = groups[key]
        first, last = ds[0], ds[-1]
        if first < ws or last > we:
            continue
        prev = [d for d in sessions if d < first]
        if not prev:
            continue
        rows.append(dict(week=f"{key[0]}-W{key[1]:02d}", decision=prev[-1], first=first, expiry=last, n_sessions=len(ds)))
    return pd.DataFrame(rows)


def normalize_options(long_df: pd.DataFrame, max_rel_spread: float, use_mid_price: bool) -> pd.DataFrame:
    """long frame (date, ric, BID, ASK, TRDPRC_1, MID_PRICE) -> indexed (ric, date) with
    bid, ask, trd, mid_price, mid, src ('bidask' | 'mid_price' | None), usable (bool)."""
    cols = ["bid", "ask", "trd", "mid_price", "mid", "src", "usable"]
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_arrays([[], []], names=["ric", "date"]))
    df = long_df.rename(columns={"BID": "bid", "ASK": "ask", "TRDPRC_1": "trd", "MID_PRICE": "mid_price"}).copy()
    for c in ("bid", "ask", "trd", "mid_price"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    m = (df["bid"] + df["ask"]) / 2.0
    ok_ba = df["bid"].notna() & df["ask"].notna() & (df["bid"] > 0) & (df["ask"] >= df["bid"]) & \
        ((df["ask"] - df["bid"]) <= max_rel_spread * m)
    ok_mp = df["mid_price"].notna() & (df["mid_price"] > 0) & use_mid_price
    df["mid"] = np.where(ok_ba, m, np.where(ok_mp, df["mid_price"], np.nan))
    df["src"] = np.where(ok_ba, "bidask", np.where(ok_mp, "mid_price", None))
    df["usable"] = df["mid"].notna()
    df = df.drop_duplicates(["ric", "date"], keep="last").set_index(["ric", "date"]).sort_index()
    return df[cols]
