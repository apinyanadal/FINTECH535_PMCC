"""
lseg_io.py -- the ONLY module that talks to LSEG (used by probe.py and fetch_data.py).

Needs LSEG Workspace open and logged in, and lseg-data.config.json in the folder you run from.
Same call pattern that worked in HW1/HW2:
    ld.open_session() -> ld.get_history(universe=[ric], fields=[...], interval="daily", start=..., end=...)

ONE RIC PER CALL (HW2 finding: a multi-RIC request fails as a whole if any RIC in it does not
exist, which made real contracts look missing). 'Not found' just means that strike is not listed.
Every answer, including 'not found', is cached in raw_cache/ so a rerun resumes instantly.
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import List

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
_ld = None
ERRORS: list = []          # (ric, message) for failures that are NOT a plain 'not found'


def open_session():
    global _ld
    import lseg.data as ld
    _ld = ld
    ld.open_session()
    return ld


def close_session():
    if _ld is not None:
        try:
            _ld.close_session()
        except Exception:
            pass


def _is_not_found(exc: Exception) -> bool:
    m = str(exc).lower()
    return any(t in m for t in ("not found", "no data", "no successful response", "70005", "universe"))


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df


def history(ric: str, fields: List[str], start: str, end: str, interval: str = "daily",
            cache_dir: str = None, pause: float = 0.1) -> pd.DataFrame:
    """Daily history for ONE RIC. Returns an empty frame if the RIC does not exist.
    If the full field list is rejected, retries with TRDPRC_1 + MID_PRICE, then TRDPRC_1 alone."""
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        safe = ric.replace("^", "_").replace(".", "_")
        path = os.path.join(cache_dir, f"{safe}_{start}_{end}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    out = pd.DataFrame()
    tried = [list(fields)]
    for alt in (["TRDPRC_1", "MID_PRICE"], ["TRDPRC_1"]):
        if alt != list(fields):
            tried.append(alt)
    answered = False
    for flds in tried:
        try:
            df = _ld.get_history(universe=[ric], fields=flds, interval=interval, start=start, end=end)
            if df is not None and len(df):
                out = _flatten(df).dropna(how="all")
            answered = True
            break
        except Exception as e:                       # noqa
            if _is_not_found(e):
                answered = True                       # contract does not exist: cache the empty answer
                break
            ERRORS.append((ric, str(e)[:200]))
            continue
    time.sleep(pause)
    for f in fields:
        if f not in out.columns:
            out[f] = float("nan")
    if path and answered:                             # never cache a timeout/other error as 'missing'
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out
