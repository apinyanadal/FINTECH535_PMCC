"""
build_site.py -- bake the books into a static site (no live LSEG on the page). Same look as HW1/HW2:
navy gradient, amber Georgia titles, #12172a panels, Plotly 2.35.2, all numbers in book.js.

    site/index.html   the page
    site/book.js      window.BOOK = {...}
    site/data.html    'Data connection required' stub (data is pulled locally, never from GitHub Pages)
    site/*.csv        blotter, weekly log, daily ledger (so the boss can check any number)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

from analysis import NAMES, benchmarks, book_metrics, pnl_split, spread_cost, week_stats


def clean(o):
    if o is None or o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, pd.Series, pd.Index, np.ndarray)):
        return [clean(v) for v in list(o)]
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, dt.date):
        return o.isoformat()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def usd(x, d=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{abs(x):,.{d}f}"
    return f"-${s}" if x < 0 else f"${s}"


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def rules_block(cfg):
    return [
        dict(id="R1", title="One snapshot per decision",
             text="Every choice uses one closing snapshot (stock close plus that day's option closing quotes) and is filled "
                  "at that same snapshot, the same convention as HW2. Nothing after that close is used. Strikes are picked by "
                  "delta: implied volatility is backed out of each option's own mid with Black-Scholes "
                  f"(r = {cfg.RISK_FREE * 100:.2f}%, no dividend, T = calendar days / 365), then delta is computed from it."),
        dict(id="R2", title="Long leg (PMCC)",
             text=f"Day 1 close: buy 1 {cfg.LEAP_EXPIRY:%b %d, %Y} call, the strike whose delta is closest to "
                  f"{cfg.LEAP_TARGET_DELTA:.2f} (deep in the money, so it moves almost like stock). Held for the whole window, "
                  f"never rolled, marked daily at mid. The covered call and buy-and-hold books buy {cfg.SHARES} shares instead."),
        dict(id="R3", title="Short leg (CC and PMCC)",
             text=f"Close of each week's first session (Monday, or Tuesday after a holiday): sell 1 call expiring at the end of "
                  f"that week, the strike whose delta is closest to {cfg.SHORT_TARGET_DELTA:.2f}. Both books sell the identical "
                  f"contract at the identical price, so any difference between them comes only from the long leg."),
        dict(id="R4", title="Fill",
             text=f"At mid = (bid + ask) / 2 of the closing quote. A quote counts only if bid > 0, ask >= bid and the spread is at "
                  f"most {cfg.MAX_REL_SPREAD * 100:.0f}% of mid; if bid/ask is missing that day, LSEG's MID_PRICE is used and the fill "
                  f"is flagged. No usable price means no call that week (logged, never invented)."),
        dict(id="R5", title="Expiry",
             text="Expiry-day close at or below the strike: the call expires worthless and the premium is kept. Close above the "
                  "strike: the call is bought back at intrinsic value (close minus strike) at that close, reported as 'assigned'. "
                  "For the covered call this is identical to having the shares called away at the strike and buying them back "
                  "at the close. For the PMCC it is the only sensible action: if the call were exercised, the account would start "
                  "Monday short 100 shares while holding only a call."),
        dict(id="R6", title="Capital",
             text=f"All three books start with the same cash, the cost of {cfg.SHARES} shares on day 1, so the dollar P&L lines "
                  f"are directly comparable. Uninvested cash earns {cfg.RISK_FREE * 100:.2f}% a year (negative cash pays it), "
                  "accrued daily. No margin, commissions or taxes. The PMCC leaves most of its cash idle, which is the point "
                  "of the strategy, so 'return on capital deployed' is reported separately."),
    ]


def facts_block(cfg, m, sc, ws, bm):
    p, c, h = m["PMCC"], m["CC"], m["BH"]
    f = [
        f"Window {m['window'][0]} to {m['window'][1]}: {ws['n_weeks']} weeks, {ws['n_traded']} short calls sold, "
        f"{ws['n_skipped']} weeks skipped. AMZN {usd(bm['amzn_t0'], 2)} to {usd(bm['amzn_t1'], 2)} ({pct(bm['amzn'])})"
        + (f"; S&P 500 {pct(bm['spx'])}." if "spx" in bm else "."),
        f"Short calls: {ws['n_expired']} expired worthless, {ws['n_assigned']} finished in the money and were bought back. "
        f"Average delta at sale {ws['avg_delta']:.2f}; average premium {usd(ws['avg_premium'])} per contract "
        f"({ws['avg_prem_pct']:.2f}% of spot). Expired weeks made {usd(ws['pnl_expired'])}; in-the-money weeks lost "
        f"{usd(-ws['pnl_assigned'])}. Net from the short calls: {usd(p['call_pnl'])} (identical in both books).",
        f"Capital deployed: {usd(c['deployed'])} for 100 shares vs {usd(p['deployed'])} for the LEAP "
        f"(K {p['leap_strike']:g}), i.e. {p['capital_ratio'] * 100:.0f}% of the stock.",
        f"Dollar P&L on the same {usd(p['start'])} starting cash: buy & hold {usd(h['pnl'])}, covered call {usd(c['pnl'])}, "
        f"PMCC {usd(p['pnl'])}.",
        f"PMCC P&L by source: LEAP {usd(p['leap_pnl'])}, short calls {usd(p['call_pnl'])}, interest on idle cash "
        f"{usd(p['interest'])}. Covered call: stock {usd(c['stock_pnl'])}, short calls {usd(c['call_pnl'])}, interest "
        f"{usd(c['interest'])}.",
        f"Return on capital deployed (P&L excluding interest / capital deployed): buy & hold {pct(h['ret_on_deployed'])}, "
        f"covered call {pct(c['ret_on_deployed'])}, PMCC {pct(p['ret_on_deployed'])}.",
        f"Max drawdown of NAV (on the equal starting cash): buy & hold {pct(h['max_dd'])}, covered call {pct(c['max_dd'])}, "
        f"PMCC {pct(p['max_dd'])}.",
        f"Mid-fill cost: half the quoted spread on every option fill (plus selling the LEAP at the bid at the end) is "
        f"{usd(sc['PMCC']['half_spread'])} for the PMCC and {usd(sc['CC']['half_spread'])} for the covered call. "
        f"{sc['PMCC']['n_unknown']} PMCC fills used MID_PRICE (spread unknown).",
    ]
    return f


def make_book(res, raw, cfg, checks, commentary_user):
    m = book_metrics(res, cfg)
    sc = spread_cost(res, cfg)
    ws = week_stats(res)
    bm = benchmarks(res)
    m["window"] = (res["t0"].isoformat(), res["end"].isoformat())
    split = pnl_split(res)
    led, b, wl = res["ledger"], res["blotter"], res["weeks_log"]

    nav = {}
    for bk in ("BH", "CC", "PMCC"):
        L = led[led["book"] == bk]
        nav[bk] = dict(pnl=(L["nav"] - res["start_cash"]).to_numpy())
    sale_days = set(wl[wl["status"] == "TRADED"]["first"])
    net = {"dates": [d.isoformat() for d in sorted(sale_days)]}
    for bk in ("BH", "CC", "PMCC"):
        L = led[(led["book"] == bk) & led["date"].isin(sale_days)].sort_values("date")
        net[bk] = L["net_shares"].to_numpy()
    dates = [d.isoformat() for d in led[led["book"] == "PMCC"]["date"]]
    days = sorted(res["close"].index)
    hol = [d.isoformat() for d in pd.date_range(days[0], days[-1]).date if d.weekday() < 5 and d not in set(days)]

    trades = wl[wl["status"] == "TRADED"]
    segs = [dict(week=w.week, strike=w.strike, t0=w.first.isoformat(), t1=w.expiry.isoformat(), outcome=w.outcome,
                 close=w.expiry_close, delta=w.delta) for w in trades.itertuples()]

    table = []
    for bk in ("BH", "CC", "PMCC"):
        x = m[bk]
        table.append(dict(book=x["name"], deployed=usd(x["deployed"]), pnl=usd(x["pnl"]), ret=pct(x["ret"]),
                          rod=pct(x["ret_on_deployed"]), gmrr=pct(x["gmrr"]), vol=pct(x["vol"]),
                          sharpe=f"{x['sharpe']:.2f}" if not math.isnan(x["sharpe"]) else "n/a",
                          mdd=pct(x["max_dd"]), calls=usd(x.get("call_pnl")) if bk != "BH" else "-"))

    p, c = m["PMCC"], m["CC"]
    cards = [
        ("PMCC P&L", usd(p["pnl"]), f"covered call {usd(c['pnl'])} · hold {usd(m['BH']['pnl'])}"),
        ("Capital deployed", f"{p['capital_ratio'] * 100:.0f}%", f"LEAP {usd(p['deployed'])} vs shares {usd(c['deployed'])}"),
        ("Return on capital", pct(p["ret_on_deployed"]), f"covered call {pct(c['ret_on_deployed'])}"),
        ("Expired · Assigned", f"{ws['n_expired']} · {ws['n_assigned']}", f"{ws['n_traded']} calls over {ws['n_weeks']} weeks"),
        ("Short-call P&L", usd(p["call_pnl"]), f"premium {usd(p['premium'])} - buybacks {usd(p['buybacks'])}"),
        ("Max drawdown", pct(p["max_dd"]), f"covered call {pct(c['max_dd'])}"),
    ]

    blot = b.copy()
    blot["book"] = blot["book"].map({"BH": "Buy & hold", "CC": "Covered call", "PMCC": "PMCC"})
    blot_rows = blot[["date", "book", "kind", "side", "instrument", "qty", "price", "cash_delta", "src", "note"]].to_dict("records")
    wk_rows = wl[["week", "first", "expiry", "spot_entry", "strike", "delta", "iv", "premium", "bid", "ask", "src",
                  "expiry_close", "outcome", "buyback", "call_pnl", "status", "reason"]].to_dict("records")
    pm = led[led["book"] == "PMCC"].reset_index(drop=True)
    wide = pd.DataFrame(dict(date=pm["date"], spot=pm["spot"],
                             bh=led[led["book"] == "BH"]["nav"].to_numpy(), cc=led[led["book"] == "CC"]["nav"].to_numpy(),
                             pmcc=pm["nav"], cash=pm["cash"], leap=pm["leap_mark"], leap_delta=pm["leap_delta"],
                             short=pm["short_strike"], short_mark=pm["short_mark"], net=pm["net_shares"],
                             stale=pm["leap_stale"] | pm["short_stale"]))

    sections = [("purpose", "What the strategy is and why it uses less capital"),
                ("expiry_rule", "Why in-the-money calls are bought back"),
                ("what_happened", "What happened"),
                ("benefits", "Where the PMCC helped"),
                ("drawbacks", "Where it hurt"),
                ("verdict", "Should we deploy it?")]
    user = commentary_user or {}
    comm = [dict(key=k, title=t, text=(user.get(k) or "").strip()) for k, t in sections]

    book = dict(
        meta=dict(ticker=cfg.TICKER, synthetic=bool(raw["meta"].get("synthetic")), fetched_at=raw["meta"].get("fetched_at"),
                  generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), first=m["window"][0], last=m["window"][1],
                  github=cfg.GITHUB_URL, leap_expiry=cfg.LEAP_EXPIRY.isoformat(), short_delta=cfg.SHORT_TARGET_DELTA,
                  leap_delta=cfg.LEAP_TARGET_DELTA, rf=cfg.RISK_FREE, start=res["start_cash"]),
        cards=[dict(label=a, value=v, sub=s) for a, v, s in cards], rules=rules_block(cfg), table=table,
        facts=facts_block(cfg, m, sc, ws, bm), dates=dates, nav=nav, net=net, holidays=hol,
        price=dict(dates=[d.isoformat() for d in days if res["t0"] <= d <= res["end"]],
                   close=[float(res["close"].loc[d]) for d in days if res["t0"] <= d <= res["end"]]),
        leap_strike=p["leap_strike"], segments=segs,
        weekly=dict(week=trades["week"], pnl=trades["call_pnl"], outcome=trades["outcome"], strike=trades["strike"],
                    close=trades["expiry_close"], premium=trades["premium"]),
        split=dict(leap=split["leap"], calls=split["calls"], interest=split["interest"], total=split["total"],
                   cc_stock=split["cc_stock"]),
        blotter=blot_rows, weeks=wk_rows, ledger=wide.to_dict("records"), checks=checks, commentary=comm,
        rics=dict(weekly_example=wl[wl["ric"].notna()]["ric"].iloc[0] if wl["ric"].notna().any() else "",
                  leap=b[(b["book"] == "PMCC") & (b["side"] == "BUY")]["instrument"].iloc[0]),
    )
    return clean(book), dict(blotter=b, weeks=wl, ledger=wide)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="book.js"></script>
<style>
  :root { color-scheme: dark; --amber:#f5a623; --coral:#ff5d73; --violet:#7c5cff; --green:#3ddc97; --blue:#6ea8fe;
          --panel:#12172a; --line:#232a45; --mute:#9791b3; --soft:#cfc9e0; --txt:#e8e6f0; }
  * { box-sizing: border-box; }
  body { margin:0; padding:2.5rem 3rem; font-family:'Segoe UI', system-ui, sans-serif;
         background: radial-gradient(circle at top left, #151a30, #0b0e1a 60%); background-attachment: fixed; color:var(--txt); }
  h1 { font-family:'Georgia', serif; font-weight:700; letter-spacing:.5px; color:var(--amber); margin:0 0 .25rem 0; font-size:1.9rem; }
  h2 { font-family:'Georgia', serif; color:var(--amber); font-size:1.25rem; margin:2.2rem 0 .4rem 0; }
  .eyebrow { color:var(--mute); font-size:.78rem; text-transform:uppercase; letter-spacing:.6px; margin-top:2.2rem; }
  .sub { color:var(--mute); margin:0 0 1rem 0; font-size:.95rem; }
  nav a { color:var(--soft); margin-right:1.2rem; font-size:.9rem; text-decoration:none; border-bottom:1px solid var(--line); }
  nav a:hover { color:var(--amber); }
  .banner { background:#3b1219; border:1px solid var(--coral); color:#ffd7dc; border-radius:10px; padding:.8rem 1.1rem; margin:1rem 0; font-weight:600; }
  .stats-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1.2rem 0 1.5rem 0; }
  .stat-card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1rem 1.4rem; min-width:11rem; flex:1; }
  .stat-card .lbl { color:var(--mute); font-size:.78rem; text-transform:uppercase; letter-spacing:.5px; }
  .stat-card .val { color:var(--amber); font-size:1.6rem; font-weight:700; margin-top:.2rem; }
  .stat-card .sub2 { color:var(--mute); font-size:.78rem; margin-top:.15rem; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1rem; margin-bottom:1.5rem; }
  .note { color:var(--mute); font-size:.85rem; margin:.3rem 0 .9rem 0; line-height:1.5; }
  .note b { color:var(--soft); }
  .rules { display:grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap:1rem; }
  .rule { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.9rem 1.1rem; }
  .rule .rid { color:var(--amber); font-weight:700; font-family:Consolas, monospace; }
  .rule .rt { font-weight:600; margin-left:.4rem; }
  .rule p { color:var(--soft); font-size:.88rem; line-height:1.5; margin:.4rem 0 0 0; }
  .tablewrap { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:auto; max-height:560px; }
  table { border-collapse:collapse; width:100%; font-size:.82rem; }
  th { position:sticky; top:0; background:#0f1426; color:var(--mute); text-transform:uppercase; font-size:.7rem; letter-spacing:.5px;
       text-align:left; padding:.6rem .7rem; border-bottom:1px solid var(--line); white-space:nowrap; z-index:1; }
  td { padding:.45rem .7rem; border-bottom:1px solid #1a2038; vertical-align:top; color:var(--soft); }
  td.num, th.num { text-align:right; font-family:Consolas, 'Courier New', monospace; white-space:nowrap; }
  tr:hover td { background:#161c33; }
  .ric { font-family:Consolas, monospace; color:var(--txt); }
  .badge { display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.4px; }
  .b-BUY { background:rgba(61,220,151,.15); color:var(--green); }
  .b-SELL { background:rgba(245,166,35,.16); color:var(--amber); }
  .b-EXPIRE, .b-EXPIRED { background:rgba(110,168,254,.16); color:var(--blue); }
  .b-BUYBACK, .b-ASSIGNED { background:rgba(255,93,115,.16); color:var(--coral); }
  .b-SKIPPED { background:rgba(151,145,179,.16); color:var(--mute); }
  .pos { color:var(--green); } .neg { color:var(--coral); } .mute { color:var(--mute); }
  .notecell { max-width:520px; line-height:1.4; }
  .commentary { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1.25rem 1.5rem; line-height:1.6; color:var(--soft); font-size:.95rem; }
  .commentary strong { color:var(--amber); }
  .commentary p { margin:.2rem 0 1rem 0; }
  .todo { border:1px dashed var(--amber); color:var(--amber); border-radius:8px; padding:.5rem .8rem; font-style:italic; margin-bottom:1rem; }
  ul.facts { color:var(--soft); font-size:.9rem; line-height:1.6; padding-left:1.2rem; }
  ul.facts li { margin-bottom:.35rem; }
  details { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.8rem 1.1rem; }
  summary { cursor:pointer; color:var(--amber); font-weight:600; }
  .chk { font-size:.85rem; margin:.35rem 0; color:var(--soft); }
  .footer { color:var(--mute); font-size:.78rem; margin-top:2.5rem; }
  .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:1.5rem; }
  .dl a { color:var(--amber); margin-right:1.2rem; font-size:.88rem; }
  code { font-family:Consolas, monospace; color:var(--txt); background:#0f1426; padding:.05rem .35rem; border-radius:4px; }
  ol.steps { color:var(--soft); font-size:.9rem; line-height:1.7; }
  @media (max-width: 1000px) { body { padding:1.2rem; } .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1 id="title"></h1>
<p class="sub" id="subtitle"></p>
<nav><a href="index.html">Book</a><a href="data.html">Data</a></nav>
<div id="banner"></div>
<div class="stats-row" id="cards"></div>


<div class="eyebrow">Results</div>
<h2>Three books, same starting cash</h2>
<p class="note" id="table_note"></p>
<div class="tablewrap" style="max-height:none"><table id="perf"><thead><tr>
  <th>Book</th><th class="num">Capital deployed</th><th class="num">P&amp;L</th><th class="num">Return on start cash</th>
  <th class="num">Annualized (GMRR, on start cash)</th><th class="num">Return on capital deployed</th><th class="num">Annual vol</th>
  <th class="num">Sharpe</th><th class="num">Max drawdown</th><th class="num">Short-call P&amp;L</th>
</tr></thead><tbody></tbody></table></div>

<h2>Cumulative P&amp;L</h2>
<p class="note">Daily mark-to-market P&amp;L of each book, all starting from the same cash. The short calls are identical in the
  covered call and the PMCC, so the gap between those two lines is entirely the LEAP vs the shares (plus interest on the PMCC's idle cash).</p>
<div class="panel"><div id="fig_pnl" style="height:420px"></div></div>

<div class="eyebrow">Short calls</div>
<h2>Stock and the strikes sold: expired vs assigned</h2>
<p class="note">Daily close of the stock. Each bar is one weekly call from sale to expiry at its strike:
  <b style="color:#6ea8fe">blue = expired worthless</b>, <b style="color:#ff5d73">coral = finished in the money (bought back)</b>.
  The dotted line is the LEAP strike.</p>
<div class="panel"><div id="fig_price" style="height:400px"></div></div>
<h2>Weekly short-call P&amp;L</h2>
<p class="note">Premium collected minus the buyback cost, per contract-week. Small, steady gains in quiet weeks; large losses
  in the weeks the stock rallied through the strike.</p>
<div class="panel"><div id="fig_weekly" style="height:340px"></div></div>

<div class="eyebrow">Mechanism</div>
<div class="grid2">
  <div>
    <h2>Where the PMCC's P&amp;L came from</h2>
    <p class="note">Cumulative mark-to-market by source. Dotted: the covered call's stock leg, for comparison with the LEAP line.</p>
    <div class="panel"><div id="fig_split" style="height:380px"></div></div>
  </div>
  <div>
    <h2>Exposure at each sale: share-equivalents</h2>
    <p class="note">Measured at the close each weekly call was sold. PMCC = 100 x (LEAP delta - short call delta); covered call =
      100 x (1 - short call delta). It shows how much of a $1 stock move each position keeps, up or down.</p>
    <div class="panel"><div id="fig_delta" style="height:380px"></div></div>
  </div>
</div>

<div class="eyebrow">Write-up</div>
<h2>Numbers behind the write-up</h2>
<ul class="facts" id="facts"></ul>
<h2>Commentary</h2>
<div class="commentary" id="commentary"></div>

<div class="eyebrow">Records</div>
<h2>Weekly log</h2>
<p class="note">Every week in the window, including any week where no call was sold.</p>
<div class="tablewrap"><table id="weeks"><thead><tr>
  <th>Week</th><th>Sold</th><th>Expiry</th><th class="num">Spot</th><th class="num">Strike</th><th class="num">Delta</th><th class="num">IV</th>
  <th class="num">Bid</th><th class="num">Ask</th><th class="num">Mid (fill)</th><th class="num">Expiry close</th><th>Result</th><th class="num">Buyback</th><th class="num">Call P&amp;L</th><th>Note</th>
</tr></thead><tbody></tbody></table></div>

<h2>Blotter</h2>
<p class="note" id="blotter_note"></p>
<div class="tablewrap"><table id="blotter"><thead><tr>
  <th>Date</th><th>Book</th><th>Side</th><th>Instrument</th><th class="num">Qty</th><th class="num">Price</th><th class="num">Cash &Delta;</th><th>Notes</th>
</tr></thead><tbody></tbody></table></div>

<h2>Daily ledger</h2>
<p class="note">NAV of each book at every close, plus the PMCC's positions and marks. * = a mark carried from an earlier day
  (no usable quote that day). Short-call columns are blank on expiry days, after settlement.</p>
<div class="tablewrap"><table id="ledger"><thead><tr>
  <th>Date</th><th class="num">AMZN</th><th class="num">Hold NAV</th><th class="num">CC NAV</th><th class="num">PMCC NAV</th>
  <th class="num">PMCC cash</th><th class="num">LEAP mark</th><th class="num">LEAP delta</th><th class="num">Short K</th><th class="num">Short mark</th><th class="num">PMCC share-eq.</th>
</tr></thead><tbody></tbody></table></div>
<p class="dl" style="margin-top:.8rem"><a href="blotter.csv">blotter.csv</a><a href="weeks.csv">weeks.csv</a><a href="ledger.csv">ledger.csv</a></p>


<div class="footer" id="footer"></div>

<script>
const B = window.BOOK;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const usd = (x, d=0) => (x == null) ? '' : (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const num = (x, d=2) => (x == null) ? '' : Number(x).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const cls = (x) => x > 0 ? 'pos' : (x < 0 ? 'neg' : 'mute');
const AMBER='#f5a623', CORAL='#ff5d73', VIOLET='#7c5cff', GREEN='#3ddc97', BLUE='#6ea8fe', SOFT='#cfc9e0', GREY='#8a8fa8', GRID='#232a45', PANEL='#12172a';
function baseLayout(extra) {
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:PANEL, font:{color:SOFT, family:'Segoe UI, system-ui, sans-serif'},
    margin:{l:70, r:25, t:40, b:55}, hovermode:'x unified', hoverlabel:{bgcolor:'#0b0e1a', bordercolor:GRID, font:{color:'#e8e6f0'}},
    legend:{orientation:'h', y:-0.18}, xaxis:{gridcolor:GRID, zerolinecolor:GRID}, yaxis:{gridcolor:GRID, zerolinecolor:GRID}
  }, extra || {});
}
const CFG = {responsive:true, displaylogo:false};
const RB = [{bounds:['sat','mon']}].concat(B.holidays.length ? [{values:B.holidays}] : []);
const dateAxis = (extra) => Object.assign({gridcolor:GRID, type:'date', rangebreaks:RB}, extra || {});

// header
$('title').textContent = B.meta.ticker + ' · POOR MAN’S COVERED CALL';
$('subtitle').textContent = 'Jan-2027 LEAP (Δ≈' + B.meta.leap_delta.toFixed(2) + ') + weekly short call (Δ≈' + B.meta.short_delta.toFixed(2) +
  ') vs covered call vs buy & hold · ' + B.meta.first + ' to ' + B.meta.last + ' · daily LSEG data · static snapshot';
if (B.meta.synthetic) $('banner').innerHTML = '<div class="banner">SYNTHETIC DATA: pipeline dry run on a made-up tape. Do not publish this page.</div>';
$('cards').innerHTML = B.cards.map(c => `<div class="stat-card"><div class="lbl">${esc(c.label)}</div><div class="val">${esc(c.value)}</div><div class="sub2">${esc(c.sub)}</div></div>`).join('');

// performance table
$('table_note').textContent = `Every book starts with ${usd(B.meta.start)}. Annualized figures use ${B.nav.PMCC.pnl.length} trading days and are noisy over a window this short. ` +
  `Sharpe = (GMRR - ${(B.meta.rf*100).toFixed(2)}%) / annual vol. Return on capital deployed = P&L excluding interest / (cost of 100 shares, or cost of the LEAP).`;
document.querySelector('#perf tbody').innerHTML = B.table.map(r => `<tr><td><b>${esc(r.book)}</b></td><td class="num">${r.deployed}</td><td class="num">${r.pnl}</td>` +
  `<td class="num">${r.ret}</td><td class="num">${r.gmrr}</td><td class="num">${r.rod}</td><td class="num">${r.vol}</td><td class="num">${r.sharpe}</td><td class="num">${r.mdd}</td><td class="num">${r.calls}</td></tr>`).join('');

// cumulative P&L
Plotly.newPlot('fig_pnl', [
  {x:B.dates, y:B.nav.BH.pnl, name:'Buy & hold (100 shares)', mode:'lines', line:{color:GREY, width:3, dash:'dot'}, hovertemplate:'hold %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.nav.CC.pnl, name:'Covered call', mode:'lines', line:{color:VIOLET, width:4}, hovertemplate:'covered call %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.nav.PMCC.pnl, name:'PMCC', mode:'lines', line:{color:AMBER, width:4}, hovertemplate:'PMCC %{y:$,.0f}<extra></extra>'}
], baseLayout({title:{text:'Cumulative P&L, same starting cash', font:{size:14}}, xaxis:dateAxis(), yaxis:{gridcolor:GRID, tickprefix:'$', tickformat:',.0f', zerolinecolor:SOFT}}), CFG);

// price + strikes
(function() {
  const tr = [{x:B.price.dates, y:B.price.close, name:B.meta.ticker + ' close', mode:'lines', line:{color:SOFT, width:2.5}, hovertemplate:'$%{y:.2f}<extra>close</extra>'},
              {x:[B.price.dates[0], B.price.dates[B.price.dates.length-1]], y:[B.leap_strike, B.leap_strike], name:'LEAP strike', mode:'lines', line:{color:AMBER, width:2, dash:'dot'}, hoverinfo:'skip'}];
  [['EXPIRED', BLUE, 'Expired worthless'], ['ASSIGNED', CORAL, 'In the money: bought back']].forEach(([o, color, name]) => {
    const xs = [], ys = [], tx = [];
    B.segments.filter(s => s.outcome === o).forEach(s => {
      const l = `${s.week}: K ${s.strike} (Δ ${s.delta.toFixed(2)}) · expiry close ${s.close.toFixed(2)} · ${o}`;
      xs.push(s.t0, s.t1, null); ys.push(s.strike, s.strike, null); tx.push(l, l, '');
    });
    tr.push({x:xs, y:ys, text:tx, name:name, mode:'lines', connectgaps:false, line:{color:color, width:5}, hovertemplate:'%{text}<extra></extra>'});
  });
  Plotly.newPlot('fig_price', tr, baseLayout({hovermode:'closest', title:{text:'Stock vs short-call strikes', font:{size:14}}, xaxis:dateAxis(), yaxis:{gridcolor:GRID, tickprefix:'$'}}), CFG);
})();

// weekly bars
Plotly.newPlot('fig_weekly', [{x:B.weekly.week, y:B.weekly.pnl, type:'bar', marker:{color:B.weekly.outcome.map(o => o === 'ASSIGNED' ? CORAL : BLUE)},
  customdata:B.weekly.week.map((w, i) => [B.weekly.strike[i], B.weekly.close[i], B.weekly.premium[i] * 100, B.weekly.outcome[i]]),
  hovertemplate:'%{x}: %{y:$,.0f}<br>K %{customdata[0]} · expiry close %{customdata[1]:.2f}<br>premium %{customdata[2]:$,.0f} · %{customdata[3]}<extra></extra>'}],
  baseLayout({hovermode:'closest', showlegend:false, title:{text:'Short-call P&L per week (per contract)', font:{size:14}}, xaxis:{gridcolor:GRID, tickangle:-45}, yaxis:{gridcolor:GRID, tickprefix:'$', tickformat:',.0f'}}), CFG);

// split
Plotly.newPlot('fig_split', [
  {x:B.dates, y:B.split.leap, name:'LEAP', mode:'lines', line:{color:GREEN, width:3}, hovertemplate:'LEAP %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.split.calls, name:'Short calls', mode:'lines', line:{color:CORAL, width:3}, hovertemplate:'calls %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.split.interest, name:'Interest on idle cash', mode:'lines', line:{color:BLUE, width:3}, hovertemplate:'interest %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.split.total, name:'PMCC total', mode:'lines', line:{color:AMBER, width:4}, hovertemplate:'total %{y:$,.0f}<extra></extra>'},
  {x:B.dates, y:B.split.cc_stock, name:'CC stock leg', mode:'lines', line:{color:GREY, width:2.5, dash:'dot'}, hovertemplate:'CC stock %{y:$,.0f}<extra></extra>'}
], baseLayout({xaxis:dateAxis(), yaxis:{gridcolor:GRID, tickprefix:'$', tickformat:',.0f', zerolinecolor:SOFT}}), CFG);

// deltas
Plotly.newPlot('fig_delta', [
  {x:B.net.dates, y:B.net.BH, name:'Buy & hold', mode:'lines', line:{color:GREY, width:2.5, dash:'dot'}, hovertemplate:'hold %{y:.0f}<extra></extra>'},
  {x:B.net.dates, y:B.net.CC, name:'Covered call', mode:'lines+markers', line:{color:VIOLET, width:3}, marker:{size:7}, hovertemplate:'CC %{y:.0f} shares<extra></extra>'},
  {x:B.net.dates, y:B.net.PMCC, name:'PMCC', mode:'lines+markers', line:{color:AMBER, width:3}, marker:{size:7}, hovertemplate:'PMCC %{y:.0f} shares<extra></extra>'}
], baseLayout({xaxis:dateAxis(), yaxis:{gridcolor:GRID, title:'shares', range:[0, 105]}}), CFG);

// facts + commentary
$('facts').innerHTML = B.facts.map(f => `<li>${esc(f)}</li>`).join('');
$('commentary').innerHTML = B.commentary.map(c => `<p><strong>${esc(c.title)}${/[?.]$/.test(c.title) ? '' : '.'}</strong></p>` +
  (c.text ? c.text.split(/\n\s*\n/).map(p => `<p>${esc(p)}</p>`).join('') : '<div class="todo">Not written yet: add this section to commentary.json and rebuild.</div>')).join('');

// weekly log
document.querySelector('#weeks tbody').innerHTML = B.weeks.map(w => {
  const res = w.status === 'TRADED' ? `<span class="badge b-${w.outcome}">${w.outcome}</span>` : '<span class="badge b-SKIPPED">SKIPPED</span>';
  return `<tr><td style="white-space:nowrap">${esc(w.week)}</td><td style="white-space:nowrap">${esc(w.first)}</td><td style="white-space:nowrap">${esc(w.expiry)}</td><td class="num">${num(w.spot_entry)}</td><td class="num">${w.strike == null ? '' : num(w.strike)}</td>` +
    `<td class="num">${w.delta == null ? '' : num(w.delta)}</td><td class="num">${w.iv == null ? '' : (w.iv*100).toFixed(0) + '%'}</td><td class="num">${w.bid == null ? '' : num(w.bid)}</td><td class="num">${w.ask == null ? '' : num(w.ask)}</td>` +
    `<td class="num">${w.premium == null ? '' : num(w.premium, 3)}${w.src === 'mid_price' ? '†' : ''}</td><td class="num">${num(w.expiry_close)}</td><td>${res}</td>` +
    `<td class="num">${w.buyback ? num(w.buyback) : ''}</td><td class="num ${cls(w.call_pnl)}">${w.status === 'TRADED' ? usd(w.call_pnl) : ''}</td><td class="notecell">${esc(w.reason)}${w.src === 'mid_price' ? '† filled at MID_PRICE (no bid/ask that day)' : ''}</td></tr>`;
}).join('');

// blotter
(function() {
  const r = B.blotter, n = (f) => r.filter(f).length;
  $('blotter_note').textContent = `${r.length} rows across the three books: ${n(x => x.book === 'PMCC' && x.side === 'SELL')} calls sold per options book, ` +
    `${n(x => x.book === 'PMCC' && x.side === 'EXPIRE')} expired, ${n(x => x.book === 'PMCC' && x.side === 'BUYBACK')} bought back at intrinsic. Option prices are per share (x100 per contract).`;
  document.querySelector('#blotter tbody').innerHTML = r.map(x => `<tr><td style="white-space:nowrap">${esc(x.date)}</td><td>${esc(x.book)}</td>` +
    `<td><span class="badge b-${x.side}">${x.side}</span></td><td><span class="ric">${esc(x.instrument)}</span></td><td class="num">${x.qty}</td>` +
    `<td class="num">${num(x.price, x.kind === 'OPT' ? 3 : 2)}</td><td class="num ${cls(x.cash_delta)}">${x.cash_delta === 0 ? '0.00' : usd(x.cash_delta, 2)}</td><td class="notecell">${esc(x.note)}</td></tr>`).join('');
})();

// ledger
document.querySelector('#ledger tbody').innerHTML = B.ledger.map(r => `<tr><td style="white-space:nowrap">${esc(r.date)}</td><td class="num">${num(r.spot)}</td>` +
  `<td class="num">${usd(r.bh)}</td><td class="num">${usd(r.cc)}</td><td class="num"><b>${usd(r.pmcc)}</b></td><td class="num">${usd(r.cash)}</td>` +
  `<td class="num">${num(r.leap)}${r.stale ? '*' : ''}</td><td class="num">${r.leap_delta == null ? '' : num(r.leap_delta)}</td><td class="num">${r.short == null ? '' : num(r.short)}</td>` +
  `<td class="num">${r.short_mark == null ? '' : num(r.short_mark, 3)}</td><td class="num">${r.net == null ? '' : num(r.net, 0)}</td></tr>`).join('');

// footer
$('footer').textContent = 'Built ' + B.meta.generated_at + ' · data pulled from LSEG Workspace ' + B.meta.fetched_at + ' · all numbers baked into book.js; this page makes no live data calls.';
</script>
</body>
</html>
"""

DATA_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Data connection required</title>
<style>
  body { margin:0; padding:2.5rem 3rem; font-family:'Segoe UI', system-ui, sans-serif; background: radial-gradient(circle at top left, #151a30, #0b0e1a 60%); background-attachment: fixed; color:#e8e6f0; }
  h1 { font-family:Georgia, serif; color:#f5a623; }
  nav a { color:#cfc9e0; margin-right:1.2rem; font-size:.9rem; text-decoration:none; border-bottom:1px solid #232a45; }
  .panel { background:#12172a; border:1px solid #232a45; border-radius:12px; padding:1.5rem; max-width:640px; margin-top:1.5rem; line-height:1.6; color:#cfc9e0; }
  .big { color:#ff5d73; font-size:1.3rem; font-weight:700; }
</style></head><body>
<h1>Data</h1>
<nav><a href="index.html">Book</a><a href="data.html">Data</a></nav>
<div class="panel"><div class="big">Data connection required</div>
<p>Live LSEG data is only available from a local Workspace session. This published page is a static snapshot: every number on the
<a href="index.html" style="color:#f5a623">Book</a> page is baked into book.js, and the raw tables are downloadable as CSV there.</p></div>
</body></html>
"""


def write_site(book: dict, tables: dict, cfg, out_dir=None):
    out_dir = out_dir or cfg.SITE_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX_HTML.replace("__TITLE__", f"{book['meta']['ticker']} Poor Man's Covered Call"))
    with open(os.path.join(out_dir, "data.html"), "w", encoding="utf-8") as f:
        f.write(DATA_HTML)
    with open(os.path.join(out_dir, "book.js"), "w", encoding="utf-8") as f:
        f.write("window.BOOK = " + json.dumps(book, ensure_ascii=False, allow_nan=False) + ";\n")
    tables["blotter"].to_csv(os.path.join(out_dir, "blotter.csv"), index=False, encoding="utf-8")
    tables["weeks"].to_csv(os.path.join(out_dir, "weeks.csv"), index=False, encoding="utf-8")
    tables["ledger"].to_csv(os.path.join(out_dir, "ledger.csv"), index=False, encoding="utf-8")
    return out_dir
