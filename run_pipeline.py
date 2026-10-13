"""
run_pipeline.py -- tape.pkl -> three books -> checks -> site/

    python run_pipeline.py              real data (needs tape.pkl from fetch_data.py)
    python run_pipeline.py --synthetic  dry run on a made-up tape, written to site_synthetic/ (never publish)
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import config
from analysis import book_metrics, week_stats
from backtest import run_backtest
from build_site import make_book, write_site
from checks import print_report, run_checks


def main():
    cfg = config.load()
    synthetic = "--synthetic" in sys.argv
    if synthetic:
        from synth_tape import make_synthetic_raw
        raw, out_dir = make_synthetic_raw(cfg), "site_synthetic"
    else:
        if not os.path.exists(cfg.TAPE_FILE):
            sys.exit("tape.pkl not found: run python fetch_data.py first")
        with open(cfg.TAPE_FILE, "rb") as f:
            raw = pickle.load(f)
        out_dir = cfg.SITE_DIR
    res = run_backtest(raw, cfg)
    checks = run_checks(res, raw, cfg)
    ok = print_report(checks)
    if not ok and "--force" not in sys.argv:
        sys.exit("Checks failed: site not built. Paste this output.")

    m, ws = book_metrics(res, cfg), week_stats(res)
    print(f"\nWindow {res['t0']} -> {res['end']}: {ws['n_traded']} calls sold, {ws['n_expired']} expired, "
          f"{ws['n_assigned']} bought back, {ws['n_skipped']} skipped")
    for bk in ("BH", "CC", "PMCC"):
        x = m[bk]
        print(f"  {x['name']:<25} deployed {x['deployed']:>10,.0f}  P&L {x['pnl']:>9,.0f}  "
              f"ret/deployed {x['ret_on_deployed'] * 100:6.1f}%  maxDD {x['max_dd'] * 100:6.1f}%")

    comm = {}
    if os.path.exists(cfg.COMMENTARY_FILE):
        with open(cfg.COMMENTARY_FILE, encoding="utf-8") as f:
            comm = json.load(f)
    book, tables = make_book(res, raw, cfg, checks, comm)
    write_site(book, tables, cfg, out_dir)
    print(f"\nSite written to {out_dir}/  (open it with: cd {out_dir}  then  python -m http.server 8000)")


if __name__ == "__main__":
    main()
