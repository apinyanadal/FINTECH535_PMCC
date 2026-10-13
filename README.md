# AMZN Poor Man's Covered Call: backtest

Jan-2027 LEAP call (delta ~0.80) + a 30-delta weekly short call, compared with a covered call
(100 shares + the same weekly call) and buy-and-hold, on the same starting cash. Daily LSEG data.

## Run (Windows, PowerShell, inside the .venv)
1. Put `lseg-data.config.json` in this folder. Open LSEG Workspace and log in.
2. `pip install -r requirements.txt`
3. `python run_tests.py`  (no LSEG; expect 6/6)
4. `python run_pipeline.py --synthetic`  (dry run, writes `site_synthetic/`; never publish it)
5. `python probe.py`  (data plumbing only, no P&L). Set `WINDOW_END`, `WEEKLY_STRIKE_STEP`, `LEAP_RIC_EXPIRED_STYLE` in `config.py`.
6. `python fetch_data.py`  (writes `tape.pkl`; rerun if interrupted, answers are cached in `raw_cache/`)
7. `python run_pipeline.py`  (runs the checks; builds `site/` only if every check passes)
8. Write `commentary.json`, run step 7 again, then publish `site/`.

## Files
| file | what it does |
|---|---|
| `config.py` | every parameter; strategy rules frozen before results |
| `ric_codec.py` | LSEG/OPRA option RICs (zero-padded day; `^MYY` suffix only for expired contracts) |
| `lseg_io.py` | the only LSEG code: one RIC per call, cached |
| `probe.py`, `fetch_data.py` | check the data, then pull it into `tape.pkl` |
| `tape.py`, `bs.py` | calendar, quote rule, Black-Scholes delta / implied vol |
| `backtest.py` | the three books: blotter, weekly log, daily ledger |
| `checks.py` | invariant checks, including an independent NAV replay |
| `analysis.py`, `build_site.py` | metrics and the static page |
