"""
config.py -- every parameter of the AMZN Poor Man's Covered Call backtest.

ANTI DATA-SNOOPING: the STRATEGY RULES block is frozen BEFORE any P&L is computed.
probe.py only reads data plumbing (formats, strike grid, field coverage, archive cutoff),
never a P&L, so running it cannot bias the rules. Set the DATA PLUMBING values from the
probe output, then run fetch_data.py and run_pipeline.py. Do not edit the rules afterwards.

The website's Rules panel is generated from these values, so the write-up always
describes exactly what the code did.
"""
import datetime as dt
from types import SimpleNamespace

# --------------------------------------------------------------------------
# UNDERLYING
# --------------------------------------------------------------------------
TICKER = "AMZN"
STOCK_RIC = "AMZN.O"
OPTION_ROOT = "AMZN"
BENCH_RIC = ".SPX"              # S&P 500, context line only

# --------------------------------------------------------------------------
# WINDOW (about 6 months of weekly cycles). WINDOW_END = last Friday for which the
# expired-options archive has data -> set from probe.py (HW1/HW2: archive lags ~1 week).
# --------------------------------------------------------------------------
WINDOW_START = "2026-04-06"     # Monday of the first week
WINDOW_END = "2026-10-02"       # PLACEHOLDER until probe.py confirms

# --------------------------------------------------------------------------
# DATA PLUMBING (set from probe.py)
# --------------------------------------------------------------------------
INTERVAL = "daily"
WEEKLY_STRIKE_STEP = 2.5        # candidate grid when ASKING LSEG for weeklies; missing RICs fail soft
LEAP_STRIKE_STEP = 5.0          # candidate grid for the LEAP
WEEKLY_BAND = (0.95, 1.12)      # weekly candidates: strikes in [0.95, 1.12] x the PRIOR close (wide enough for a Monday gap)
LEAP_BAND = (0.55, 0.95)        # LEAP candidates: strikes in [0.55, 0.95] x the PRIOR close
OPTION_FIELDS = ["BID", "ASK", "TRDPRC_1", "MID_PRICE"]
STOCK_FIELDS = ["TRDPRC_1"]     # daily TRDPRC_1 = official close
USE_MID_PRICE_FALLBACK = True   # if BID/ASK missing that day but MID_PRICE exists, use MID_PRICE as the mid
MAX_REL_SPREAD = 0.50           # a bid/ask is usable only if ask - bid <= 50% of mid

# --------------------------------------------------------------------------
# STRATEGY RULES (FROZEN BEFORE THE BACKTEST)
# --------------------------------------------------------------------------
LEAP_EXPIRY = dt.date(2027, 1, 15)   # standard January 2027 monthly (3rd Friday); still LIVE in Oct 2026
LEAP_RIC_EXPIRED_STYLE = False       # live contract -> no ^ suffix (confirm with probe.py section 5)
LEAP_TARGET_DELTA = 0.80
SHORT_TARGET_DELTA = 0.30
RISK_FREE = 0.0375              # Black-Scholes rate, Sharpe ratio and interest on idle cash
DIV_YIELD = 0.0                 # AMZN pays no dividend
CASH_EARNS_RF = True            # uninvested cash accrues RISK_FREE (same rule for all three books)
SHARES = 100
MULTIPLIER = 100
CONTRACTS = 1

# --------------------------------------------------------------------------
# FILES
# --------------------------------------------------------------------------
TAPE_FILE = "tape.pkl"
RAW_CACHE_DIR = "raw_cache"
SITE_DIR = "docs"
COMMENTARY_FILE = "commentary.json"
GITHUB_URL = ""                 # paste your repo URL here before the final build


def load(**overrides):
    import config as _c
    d = {k: getattr(_c, k) for k in dir(_c) if k.isupper()}
    d.update(overrides)
    return SimpleNamespace(**d)
