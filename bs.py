"""
bs.py -- Black-Scholes call price, delta and implied volatility (no dividend for AMZN).

Delta is not a field we trust from a vendor: we back the implied volatility out of each option's
own mid, then compute delta from it. Same inputs every time: spot = stock close of that day,
T = calendar days to expiry / 365, r = RISK_FREE.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_price(S, K, T, sigma, r, q=0.0):
    if T <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d1 - v)


def call_delta(S, K, T, sigma, r, q=0.0):
    if T <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return math.exp(-q * T) * _ncdf(d1)


def implied_vol(price, S, K, T, r, q=0.0, lo=0.01, hi=5.0):
    """None when the price is outside the no-arbitrage range (e.g. a stale mid below intrinsic)."""
    if T <= 0 or price is None or price <= 0:
        return None
    f = lambda s: call_price(S, K, T, s, r, q) - price
    try:
        if f(lo) > 0 or f(hi) < 0:
            return None
        return brentq(f, lo, hi, xtol=1e-7)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def years(d0, d1):
    return max((d1 - d0).days, 0) / 365.0
