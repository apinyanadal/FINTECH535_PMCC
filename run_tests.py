"""run_tests.py -- no LSEG needed.   python run_tests.py"""
import copy
import datetime as dt
import sys

import numpy as np
import pandas as pd

import config
from backtest import run_backtest
from bs import call_delta, call_price, implied_vol
from checks import run_checks
from ric_codec import parse_option_ric, ric_for_expiry
from synth_tape import make_synthetic_raw

CFG = config.load()


def test_ric_formats():
    assert ric_for_expiry("AMZN", dt.date(2026, 8, 7), 230.0) == "AMZNH072623000.U^H26"        # padded day
    assert ric_for_expiry("AMZN", dt.date(2026, 9, 18), 232.5) == "AMZNI182623250.U^I26"
    assert ric_for_expiry("AMZN", dt.date(2027, 1, 15), 200.0, expired=False) == "AMZNA152720000.U"
    assert ric_for_expiry("UUUU", dt.date(2026, 6, 26), 10.0, "P") == "UUUUR262601000.U^F26"     # HW1 put rule
    p = parse_option_ric("AMZNH072623000.U^H26")
    assert p["strike"] == 230.0 and p["expiry"] == dt.date(2026, 8, 7) and p["expired"]


def test_black_scholes_roundtrip():
    px = call_price(230, 235, 4 / 365, 0.32, 0.0375)
    assert abs(implied_vol(px, 230, 235, 4 / 365, 0.0375) - 0.32) < 1e-5
    assert 0.45 < call_delta(230, 230, 30 / 365, 0.3, 0.0375) < 0.6
    assert implied_vol(1.0, 230, 200, 0.1, 0.0375) is None          # below intrinsic -> rejected, not forced


def test_all_checks_pass_on_synthetic():
    raw = make_synthetic_raw(CFG)
    res = run_backtest(raw, CFG)
    failed = [c for c in run_checks(res, raw, CFG) if not c["ok"]]
    assert not failed, failed


def test_no_lookahead_future_data_cannot_change_past_trades():
    raw = make_synthetic_raw(CFG)
    cut = dt.date(2026, 7, 1)
    raw2 = copy.deepcopy(raw)
    rng = np.random.default_rng(3)
    s = raw2["stock"]
    after = [d > cut for d in s.index]
    s[after] = s[after] * np.exp(rng.normal(0, 0.05, sum(after)))
    o = raw2["options"]
    m = pd.to_datetime(o["date"]).dt.date > cut
    for c in ("BID", "ASK", "MID_PRICE"):
        o.loc[m, c] = o.loc[m, c] * 1.37
    a, b = run_backtest(raw, CFG)["blotter"], run_backtest(raw2, CFG)["blotter"]
    cols = ["book", "date", "side", "instrument", "price"]
    pa = a[a["date"] <= cut][cols].reset_index(drop=True)
    pb = b[b["date"] <= cut][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)


def test_checks_catch_tampering():
    raw = make_synthetic_raw(CFG)
    res = run_backtest(raw, CFG)
    bad = copy.deepcopy(res)
    i = bad["blotter"].index[(bad["blotter"]["side"] == "SELL")][0]
    bad["blotter"].loc[i, "price"] *= 1.2
    names = {c["name"] for c in run_checks(bad, raw, CFG) if not c["ok"]}
    assert any("cash change" in n for n in names) and any("option fills" in n for n in names), names
    bad2 = copy.deepcopy(res)
    bad2["ledger"].loc[10, "nav"] += 1.0
    names2 = {c["name"] for c in run_checks(bad2, raw, CFG) if not c["ok"]}
    assert any("NAV replay" in n for n in names2), names2


def test_expiry_rule_numbers():
    raw = make_synthetic_raw(CFG)
    res = run_backtest(raw, CFG)
    for w in res["weeks_log"][res["weeks_log"]["status"] == "TRADED"].itertuples():
        if w.expiry_close > w.strike:
            assert w.outcome == "ASSIGNED" and abs(w.call_pnl - 100 * (w.premium - (w.expiry_close - w.strike))) < 1e-9
        else:
            assert w.outcome == "EXPIRED" and abs(w.call_pnl - 100 * w.premium) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
