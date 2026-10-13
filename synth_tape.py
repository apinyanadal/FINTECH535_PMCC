"""
synth_tape.py -- a made-up tape in exactly the format fetch_data.py writes, used ONLY to dry-run
the pipeline and the website before real data exists. Anything built from it is stamped SYNTHETIC.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from bs import call_price
from ric_codec import ric_for_expiry
from tape import build_weeks

HOLIDAYS = {dt.date(2026, 4, 3), dt.date(2026, 5, 25), dt.date(2026, 6, 19), dt.date(2026, 7, 3),
            dt.date(2026, 9, 7), dt.date(2026, 11, 26)}


def sessions(a, b):
    d, out = a, []
    while d <= b:
        if d.weekday() < 5 and d not in HOLIDAYS:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def make_synthetic_raw(cfg, seed=11, s0=225.0, vol=0.32):
    rng = np.random.default_rng(seed)
    a = (pd.Timestamp(cfg.WINDOW_START) - pd.Timedelta(days=14)).date()
    b = (pd.Timestamp(cfg.WINDOW_END) + pd.Timedelta(days=3)).date()
    days = sessions(a, b)
    px, path = s0, []
    for _ in days:
        px *= float(np.exp(rng.normal(0.0004, vol / np.sqrt(252))))
        path.append(round(px, 2))
    close = pd.Series(path, index=days, name="close")
    spx = pd.Series(6500 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(days))), index=days, name="close")
    weeks = build_weeks(days, cfg.WINDOW_START, cfg.WINDOW_END)
    rows, contracts = [], []

    def quotes(ric, K, expiry, d_from, d_to, iv0, spread_frac):
        for d in [x for x in days if d_from <= x <= d_to]:
            S = float(close.loc[d])
            iv = iv0 * (1 + 0.15 * (K / S - 1)) * float(np.exp(rng.normal(0, 0.03)))
            mid = call_price(S, K, max((expiry - d).days, 0) / 365, max(iv, 0.05), cfg.RISK_FREE)
            sp = max(0.02, spread_frac * mid)
            bid, ask = max(round(mid - sp / 2, 2), 0.0), round(mid + sp / 2, 2)
            r = dict(date=pd.Timestamp(d), ric=ric, BID=bid, ASK=ask, TRDPRC_1=round(mid, 2), MID_PRICE=round(mid, 2))
            u = rng.random()
            if u < 0.04:
                r["BID"] = r["ASK"] = np.nan                    # only MID_PRICE that day
            elif u < 0.06:
                continue                                        # no row at all
            if bid <= 0:
                r["BID"] = np.nan
            rows.append(r)

    d0 = weeks.iloc[0]["decision"]
    spot0 = float(close.loc[d0])
    for k in np.arange(np.ceil(spot0 * cfg.LEAP_BAND[0] / 5) * 5, spot0 * cfg.LEAP_BAND[1], 5):
        ric = ric_for_expiry(cfg.OPTION_ROOT, cfg.LEAP_EXPIRY, float(k), expired=False)
        contracts.append(dict(ric=ric, expiry=cfg.LEAP_EXPIRY, strike=float(k), role="leap", week=None))
        quotes(ric, float(k), cfg.LEAP_EXPIRY, d0, b, 0.33, 0.02)
    for w in weeks.itertuples():
        spot = float(close.loc[w.decision])
        for k in np.arange(np.ceil(spot * cfg.WEEKLY_BAND[0] / 2.5) * 2.5, spot * cfg.WEEKLY_BAND[1], 2.5):
            ric = ric_for_expiry(cfg.OPTION_ROOT, w.expiry, float(k))
            contracts.append(dict(ric=ric, expiry=w.expiry, strike=float(k), role="weekly", week=w.week))
            quotes(ric, float(k), w.expiry, w.decision, w.expiry, 0.30, 0.06)
    meta = dict(synthetic=True, ticker=cfg.TICKER, fetched_at="synthetic", window=(cfg.WINDOW_START, cfg.WINDOW_END))
    return dict(stock=close, spx=spx, options=pd.DataFrame(rows), contracts=pd.DataFrame(contracts), meta=meta)
