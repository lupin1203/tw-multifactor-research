"""Step-1 batch: every table FACTOR_SPEC.md asks for, from one command.

    python run_step1.py --stage coverage    load, build the signals, count stocks. No returns.
    python run_step1.py                     everything, on data up to 2023-12
    python run_step1.py --end 2026-10 --open-holdout      the single final run

How it is put together
  load_raw()       reads the Taiwan data layer (data_tw.py)
  prepare()        cuts EVERYTHING at cfg.end before anything else is computed (the holdout
                   rule), then derives month-end trading figures and the market return
  derive()         the pieces that depend on a setting (universes, flat costs, delisting value)
  build_signals()  all factor values, labelled by formation month, listed stocks only
  t_*()            one function per table; run() calls them and writes csv files

Every fixed convention is a field of Cfg, so a sensitivity run is `replace(cfg, x=...)`.
Nothing here tunes a parameter: variants are reported side by side, never selected.

Choices made here because the spec leaves them open (all fixed before any result was seen):
  market return    cap-weighted return of all listed stocks from our own data (dividends
                   included). Stand-in for the official total-return index until that
                   series is downloaded and checked; 0050 is reported next to it.
  main reversal    the raw version (the spec's basic design); the other two are reported.
  composite        a stock needs at least 3 of the 6 slow factors.
  negative IC      an estimated IC below zero is used as zero: the portfolio stops trading
                   on a signal with no record, it never bets against the written direction.
  third universe   'tradable' = listed and average daily value >= the liquidity floor. It
                   is where the final portfolio buys, so the IC and the half-life are
                   measured there. Pass lines still refer to 'big' (cap >= NT$10bn).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import data_tw as dt
import evaluate as ev
import portfolio as pf
import resmom as rm
import signals as sg
import walkforward as wf

HERE = Path(__file__).parent
OUT = HERE / "output" / "step1"

SLOW = ["resmom", "revenue", "earnings", "lowvol", "value", "quality"]
FAST = ["reversal", "flows"]
PERIODS = {"full": (None, None), "2016+": ("2016-01", None)}
PRIMARY = "big"                 # universe the pass lines and the cut rule refer to
TRADE = "tradable"              # universe the final portfolio buys in
CAPITALS = (1e6, 1e7, 1e9)
DATASETS = {"revenue": "TaiwanStockMonthRevenue", "income": "TaiwanStockFinancialStatements",
            "cashflow": "TaiwanStockCashFlowsStatement", "balance": "TaiwanStockBalanceSheet",
            "yield": "TaiwanStockPER_full", "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
            "margin": "TaiwanStockMarginPurchaseShortSale", "sbl": "TaiwanDailyShortSaleBalances"}


@dataclass(frozen=True)
class Cfg:
    end: str = "2023-12"             # last month used; later data stays closed (holdout)
    open_holdout: bool = False
    big_cap: float = 1e10            # NT$, universe 'big'
    pct: float = 0.10                # winner / loser fraction in single-factor tests
    lags: int = 3                    # Newey-West lags for monthly series
    # costs
    fee: float = 0.001425
    fee_discount: float = 0.2
    tax: float = 0.003
    spread_ticks: float = 0.5
    borrow: float = 0.03
    impact_k: float = 1.0
    # delisting month
    delist_long: float = -0.30       # outcome unknown (financial trouble, or reason not verified)
    delist_short: float = 0.0
    merger_exit: float = 0.0         # merged / bought out / turned into a holding company: paid about the last price
    # signals
    reversal_skip: int = 1
    reversal_main: str = "raw"       # raw | residual | industry
    missing_rank: float | None = 0.5  # a factor a stock does not have counts as the middle rank; None = old rule:
    min_factors: int = 3             # ... average of the factors it has, at least this many (comparison only)
    # alpha model (walk-forward)
    ic_horizons: int = 6
    ic_min_months: int = 36
    ic_window: int | None = None     # None = expanding; 60 = five-year rolling
    # portfolio
    n: int = 30
    gamma: float = 1.0
    adv_floor: float = 3e7
    max_adv_share: float = 0.20
    max_industry: float = 0.30


SIM_ONLY = {"n", "gamma", "max_adv_share", "max_industry", "impact_k"}   # do not change the alpha


def kind(name: str) -> str:
    return "slow" if name in SLOW else "fast" if name in FAST else "compare"


# ===================================================================== small helpers
def vw_daily(dret: pd.DataFrame, me: pd.DataFrame) -> pd.Series:
    """Daily return of holding every listed stock in proportion to its market cap at the
    previous month-end (weights drift inside the month, reset each month-end). A stock
    without a return on a day counts as 0% that day; stocks listed during the month join
    at the next month-end."""
    per = dret.index.to_period("M")
    w_prev = me.shift(1)
    R = dret.to_numpy(dtype=float)
    out = np.full(len(dret), np.nan)
    for m in per.unique():
        if m not in w_prev.index:
            continue
        rows = np.flatnonzero(per == m)
        w = w_prev.loc[m].reindex(dret.columns).to_numpy(dtype=float)
        r = R[rows]
        ok = np.isfinite(w) & (w > 0) & np.isfinite(r).any(axis=0)
        if not ok.any():
            continue
        value = np.cumprod(1 + np.nan_to_num(r[:, ok]), axis=0) @ w[ok]
        level = np.concatenate([[w[ok].sum()], value])
        out[rows] = level[1:] / level[:-1] - 1
    return pd.Series(out, index=dret.index)


def to_monthly(d: pd.Series) -> pd.Series:
    """Compounded monthly return of a daily series (months without any value are missing)."""
    per = d.index.to_period("M")
    return np.expm1(np.log1p(d).groupby(per).sum(min_count=1))


def entry_dates(idx: pd.DatetimeIndex, day: int = 10) -> pd.Series:
    """Month -> the trading day after the monthly-revenue deadline. The deadline is the
    10th, or the next trading day when the market is closed on the 10th."""
    pos = pd.Series(np.arange(len(idx)), index=idx)
    out = {}
    for m, d in pd.Series(idx, index=idx.to_period("M")).groupby(level=0):
        late = d[d.dt.day >= day]
        if late.empty:
            continue
        i = int(pos[late.iloc[0]]) + 1
        if i < len(idx):
            out[m] = idx[i]
    return pd.Series(out)


def returns_between(dret: pd.DataFrame, dates: pd.Series, min_days: int = 10) -> pd.DataFrame:
    """dates: month -> a trading day. Row m = compounded return from the close of
    dates[m-1] to the close of dates[m] (needs `min_days` daily returns in between), so a
    signal labelled m-1 and traded at dates[m-1] earns row m: the same pairing as monthly
    returns, where dates are the month-ends."""
    dates = dates.sort_index()
    if len(dates) < 2:
        return pd.DataFrame(columns=dret.columns)
    assert ((dates.index[1:] - dates.index[:-1]) == pd.offsets.MonthEnd(1)).all(), "months must be consecutive"
    L = np.log1p(dret.fillna(0).clip(lower=-0.999)).cumsum()
    N = dret.notna().cumsum()
    a, b = dates.iloc[:-1].to_numpy(), dates.iloc[1:].to_numpy()
    r = np.expm1(L.loc[b].to_numpy() - L.loc[a].to_numpy())
    n = N.loc[b].to_numpy() - N.loc[a].to_numpy()
    out = pd.DataFrame(np.where(n >= min_days, r, np.nan), index=dates.index[1:], columns=dret.columns)
    return out.clip(*rm.RET_CLIP)


def flat_costs(close_m: pd.DataFrame, universe: pd.DataFrame, cfg: Cfg) -> tuple[float, float, float]:
    """(buy, sell, spread) as fractions for the single-factor tests, where every stock
    gets the same cost: discounted fee each way, tax on sells, and the universe's average
    spread_ticks x tick / price each way (all stock-months, equal weight)."""
    px = close_m.where(universe & (close_m > 0))
    half = pd.DataFrame(cfg.spread_ticks * pf.tick_size(px) / px, index=px.index, columns=px.columns)
    spread = float(np.nanmean(half.to_numpy())) if half.notna().any().any() else 0.0
    buy = cfg.fee * cfg.fee_discount + spread
    return buy, buy + cfg.tax, spread


def basket(members: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Equal-weight return, in the following month, of the stocks flagged True at each
    formation month; indexed by holding month."""
    f = ret.shift(-1)
    x = f.where(members.reindex(index=f.index, columns=f.columns).fillna(False).astype(bool)).mean(axis=1)
    x.index = x.index + 1
    return x.dropna()


def extreme(signal: pd.DataFrame, universe: pd.DataFrame, pct: float, top: bool = True) -> pd.DataFrame:
    """True for the top (or bottom) `pct` of the signal inside the universe."""
    r = sg.pct_rank(sg.within(signal, universe))
    return (r > 1 - pct) if top else (r < pct)


def market_adjusted(x: pd.Series, mkt_x: pd.Series, min_months: int = 36) -> pd.Series:
    """x minus beta x market excess return, the beta estimated from the months BEFORE each
    month only (expanding window); missing until `min_months` are available."""
    d = pd.concat([x.rename("x"), mkt_x.rename("m")], axis=1).dropna()
    out = pd.Series(np.nan, index=d.index)
    xv, mv = d["x"].to_numpy(), d["m"].to_numpy()
    for i in range(min_months, len(d)):
        c = np.cov(xv[:i], mv[:i])
        out.iloc[i] = xv[i] - (c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0) * mv[i]
    return out.dropna()


def stats(x: pd.Series, mkt: pd.Series, rf: pd.Series, lags: int = 6) -> dict:
    """Long-only series: performance, and alpha / beta against the market (both in excess of rf)."""
    x = x.dropna()
    a, at, b = ev.alpha_stats(x, mkt, rf, lags)
    return {**rm.perf(x, rf), "beta": b, "alpha_%/mo": a * 100, "alpha_t": at}


def stats_ls(x: pd.Series, mkt_x: pd.Series, lags: int = 6) -> dict:
    """Self-financing (long-short) series against the market's excess return."""
    x = x.dropna()
    a, at, b = ev.alpha_stats(x, mkt_x, None, lags)
    return {**rm.perf(x), "beta": b, "alpha_%/mo": a * 100, "alpha_t": at}


def goal(x: pd.Series, mkt: pd.Series) -> dict:
    """The main target: annual return >= 1.3 x the market and maximum drawdown <= 1.1 x
    the market's, in both halves. Month-end data, so drawdowns inside a month are not seen."""
    d = pd.concat([x.rename("p"), mkt.rename("m")], axis=1).dropna()
    if len(d) < 24:
        return {}
    ann = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
    k, out, ok = len(d) // 2, {}, True
    for tag, s in (("full", d), ("h1", d.iloc[:k]), ("h2", d.iloc[k:])):
        am, dm = ann(s["m"]), rm.max_drawdown(s["m"])[0]
        out[f"ret_x_mkt_{tag}"] = ann(s["p"]) / am if am > 0 else np.nan
        out[f"dd_x_mkt_{tag}"] = rm.max_drawdown(s["p"])[0] / dm if dm < 0 else np.nan
        if tag != "full":
            ok &= bool(out[f"ret_x_mkt_{tag}"] >= 1.3 and out[f"dd_x_mkt_{tag}"] <= 1.1)
    return {**out, "goal_met": ok}


def past_return(ret: pd.DataFrame, months: int) -> pd.DataFrame:
    """Compounded return of the `months` months ending at the formation month."""
    return np.expm1(np.log1p(ret.clip(lower=-0.999)).rolling(months, min_periods=months).sum())


# ============================================================================ data
def download_status() -> pd.DataFrame:
    n = len(dt.universe())
    rows = []
    for k, v in DATASETS.items():
        files = len(list((dt.RAW / v).glob("*.parquet"))) if (dt.RAW / v).exists() else 0
        rows.append({"data": k, "dataset": v, "files": files, "expected": n, "complete": files >= n})
    return pd.DataFrame(rows).set_index("data")


def load_raw() -> dict:
    """Everything from disk, uncut. Only prepare() may look at it."""
    st = download_status()
    have = lambda *ks: all(st.loc[k, "files"] > 0 for k in ks)
    p = dt.panel()
    raw = {k: p[k] for k in ("dret", "dret_raw", "close", "volume", "ret_raw", "me", "pbr", "shares", "close_m")}
    raw.update(fac=dt.factors(), info=dt.industries(), market=dt.market(), bench={"0050": dt.bench_0050()},
               mkt_level=dt.market_total_return(),
               rev=dt.revenue() if have("revenue") else None,
               stm=dt.statements() if have("income") else {},
               div_yield=dt.yields() if have("yield") else None,
               flows=dt.flows() if have("institutional", "margin") else None,
               mergers=dt.merger_exits())
    return raw


def prepare(raw: dict, cfg: Cfg) -> dict:
    """Cut every input at cfg.end, then derive what only depends on the data. After this
    function nothing later than cfg.end exists anywhere in D."""
    end = wf.guard_holdout(cfg.end, cfg.open_holdout)
    me = raw["me"]
    end = min(end, me.index.max())
    months = pd.period_range(me.index.min(), end, freq="M")
    cols = me.columns[me.loc[me.index <= end].notna().any()]
    cols = pd.Index(np.asarray(cols, dtype=object))    # plain object labels: pandas' arrow strings make label look-ups slow
    last_day = end.end_time
    M = lambda x: x.reindex(index=months, columns=cols)
    B = lambda x: M(x).fillna(False).astype(bool)
    Dy = lambda x: x.loc[x.index <= last_day]

    D = {"end": end, "months": months, "cols": cols}
    for k in ("dret", "close", "volume"):
        D[k] = Dy(raw[k]).reindex(columns=cols)
    for k in ("ret_raw", "me", "pbr", "shares", "close_m"):
        D[k] = M(raw[k])
    D["listed"] = D["me"].notna()
    # months in which a stock has stopped trading, and what the position was worth where that
    # can be computed from data up to cfg.end (see data_tw.exit_returns)
    D["gone"], D["gap_ret"], D["exits"] = dt.exit_returns(
        D["ret_raw"], D["close"], Dy(raw["dret_raw"]).reindex(columns=cols), D["shares"],
        incomplete_last=bool(end >= me.index.max()))
    # exits where the holder was paid out (merger, buy-out, holding-company conversion), from the
    # list of verified cases; only delistings dated up to cfg.end count
    D["exits"]["merger"] = dt.merger_flags(D["exits"], raw.get("mergers"), last_day)
    D["merged"] = pd.DataFrame(False, index=months, columns=cols)
    for r in D["exits"][D["exits"]["merger"]].itertuples():
        D["merged"].at[r.month, r.stock] = True

    # month-end trading figures (all known at that day's close)
    value = D["close"] * D["volume"]
    D["adv"] = M(sg.at_month_end(value.rolling(20, min_periods=10).mean()))
    D["dvol"] = M(sg.at_month_end(D["dret"].rolling(60, min_periods=30).std()))
    up, dn = pf.limit_flags(D["close"])
    up, dn = B(sg.at_month_end(up)), B(sg.at_month_end(dn))
    shut = B(sg.at_month_end(D["close"].isna()))                 # no trade on the month's last day: nothing can be opened
    halted = shut & D["listed"]                                  # ... and, while still listed, nothing can be closed.
    # Once the stock is no longer listed the position IS closed at the next month-end (after the
    # delisting-month return has been applied); otherwise it would sit in the book for ever.
    D["no_buy"], D["no_sell"] = up | shut, dn | halted
    D["no_short"], D["no_cover"] = dn | shut, up | halted        # a short opens by selling, closes by buying

    # market, risk-free rate, benchmarks
    fac = raw["fac"]
    D["fac"] = fac.loc[fac.index <= end]
    D["rf"] = D["fac"]["RF"].reindex(months).bfill().ffill()     # the rate before the first factor month: the first known one
    # the market: the exchange's own total-return index where given (TWSE-listed stocks, dividends
    # reinvested); the self-built value-weighted return of every stock in the data stays as a check
    D["mkt_own_d"] = vw_daily(D["dret"], D["me"])
    D["mkt_own"] = to_monthly(D["mkt_own_d"]).reindex(months)
    level = raw.get("mkt_level")
    if level is None:
        D["mkt_d"], D["mkt"] = D["mkt_own_d"], D["mkt_own"]
    else:
        level = Dy(level.dropna())
        D["mkt_d"] = level.reindex(level.index.union(D["dret"].index)).ffill().pct_change(fill_method=None).reindex(D["dret"].index)
        D["mkt"] = level.groupby(level.index.to_period("M")).last().pct_change(fill_method=None).reindex(months)
    D["mkt_x"] = D["mkt"] - D["rf"]
    D["bench"] = {k: v.loc[v.index <= end] for k, v in raw.get("bench", {}).items()}
    D["market"] = {k: Dy(v) for k, v in raw.get("market", {}).items()}

    # stock attributes (today's classification; see data_tw.industries)
    info = raw["info"].reindex(cols)
    D["groups"] = info["industry"]                              # for ranking inside an industry
    D["groups_official"] = info["industry_official"] if "industry_official" in info else info["industry"]   # for the 30% cap
    D["financials"] = list(info.index[info["financial"].fillna(False).astype(bool)])
    D["late"] = list(info.index[info["late"].fillna(False).astype(bool)])

    # fundamentals: only what was already published by cfg.end
    rev = raw.get("rev")
    D["rev"] = None if rev is None else rev.loc[rev.index + 1 <= end].reindex(columns=cols)
    D["stm"] = {k: v.loc[[sg.usable_month(q) <= end for q in v.index]].reindex(columns=cols)
                for k, v in (raw.get("stm") or {}).items()}
    dy = raw.get("div_yield")
    D["div_yield"] = None if dy is None else M(dy)
    fl = raw.get("flows")
    D["flows"] = None if fl is None else {k: Dy(v).reindex(columns=cols) for k, v in fl.items()}
    return derive(D, cfg)


def derive(D: dict, cfg: Cfg) -> dict:
    """The pieces of D that depend on a setting. Returns a new dict; D itself is not changed."""
    D = dict(D)
    D["cfg"] = cfg
    # stopped trading: the actual outcome where known; a merger exit pays about the last price
    # (both sides); otherwise the value that hurts each side
    out = D["gap_ret"].mask(D["merged"] & D["gap_ret"].isna(), cfg.merger_exit)
    D["ret"] = D["ret_raw"].mask(D["gone"], out.fillna(cfg.delist_long))
    D["ret_short"] = D["ret_raw"].mask(D["gone"], out.fillna(cfg.delist_short))
    L = D["listed"]
    D["U"] = {"all": L, "big": L & (D["me"] >= cfg.big_cap), "tradable": L & (D["adv"] >= cfg.adv_floor)}
    D["cost"] = {u: flat_costs(D["close_m"], U, cfg) for u, U in D["U"].items()}
    return D


# ========================================================================= signals
def build_signals(D: dict) -> tuple[dict, dict, pd.DataFrame]:
    """(S, X, status). S: name -> formation month x stock, high = expected to do better,
    listed stocks only. X: other matrices the tables need (conditions, volatility).
    status: what was built and what was skipped for lack of data."""
    cfg, months, cols, L = D["cfg"], D["months"], D["cols"], D["listed"]
    groups, fin, late, fac, rr, st = D["groups"], D["financials"], D["late"], D["fac"], D["ret_raw"], D["stm"]
    S, X, status = {}, {}, []
    nan = lambda: pd.DataFrame(np.nan, index=months, columns=cols)
    note = lambda name, ok, why="": status.append({"item": name, "built": ok, "note": why})

    # price based
    res, tot = rm.scores(rr, fac)
    S["resmom"], S["mom"] = sg.to_formation(res), sg.to_formation(tot)
    S["lowvol"] = -sg.daily_vol(D["dret"])
    X["resvol"] = sg.residual_vol(rr, fac)
    S["lowvol_resid"] = -X["resvol"]
    S["high52"] = sg.high_52w(D["dret"])
    listed_d = L.reindex(D["dret"].index.to_period("M")).to_numpy()
    abn, long_run = sg.turnover_stats(D["volume"].where(listed_d), D["shares"])
    S["turnover_abn"] = abn                                           # high = unusually busy month
    S["turnover_low"] = sg.pct_rank(sg.within(-long_run, L), groups)   # high = quiet stock, within industry
    rv = sg.score_reversal(D["dret"], rr, fac, groups, universe=L, skip=cfg.reversal_skip)
    S["reversal"] = rv[cfg.reversal_main]                              # the version used as the fast signal
    for k, v in rv.items():
        if k != cfg.reversal_main:
            S[f"reversal_{k}"] = v
    X["reversal_raw"] = rv["raw"]                                      # defines "last month's losers" in the condition tables
    for k in ("resmom", "mom", "lowvol", "lowvol_resid", "high52", "turnover_abn", "turnover_low", "reversal"):
        note(k, True)
    X["past3m"], X["past20d"] = past_return(rr, 3), sg.at_month_end(
        np.expm1(np.log1p(D["dret"].clip(lower=-0.999)).rolling(20, min_periods=15).sum()))
    X["past12_1"] = past_return(rr.shift(1), 11)                       # months m-11 .. m-1

    # monthly revenue
    if D["rev"] is not None:
        r = sg.score_revenue(D["rev"], fin, dt.BASIS_BREAK_M, universe=L)
        S["revenue"], S["rev_yoy3"], S["rev_accel"] = r["score"], r["yoy3"], r["accel"]
        X["rev_mom"], X["rev_dyoy"] = r["mom"], r["d_yoy"]
        note("revenue", True)
    else:
        note("revenue", False, "monthly revenue not downloaded")

    # quarterly statements
    if all(k in st for k in ("ni", "rev", "gross", "oper")):
        e = sg.score_earnings(st["ni"], st["rev"], st["gross"], st["oper"], months, fin, dt.BASIS_BREAK_Q, late, universe=L)
        S["earnings"], S["sue"], S["earn_accel"], S["margin_chg"] = e["score"], e["sue"], e["accel"], e["margin"]
        note("earnings", True)
        dy = D["div_yield"]
        v = sg.score_value(D["pbr"], st["ni"], D["me"], nan() if dy is None else dy, groups, late, universe=L)
        S["value"], S["value_mkt"], S["bp"], S["ep"], S["dp"] = v["score"], v["score_market_wide"], v["bp"], v["ep"], v["dp"]
        note("value", True, "" if dy is not None else "dividend yield missing: B/P and E/P only")
        miss = [k for k in ("cfo", "assets", "liab") if k not in st]
        q = sg.score_quality(st["ni"], st.get("cfo", st["ni"] * np.nan), st.get("assets", st["ni"] * np.nan),
                             st.get("liab", st["ni"] * np.nan), D["me"], D["pbr"], groups, fin, late, universe=L)
        S["quality"], S["roe"] = q["score"], q["roe"]
        S["accruals_low"], S["roe_stable"], S["leverage_low"] = -q["accruals"], -q["roe_sd"], -q["leverage"]
        note("quality", True, "" if not miss else "missing: " + ", ".join(miss))
    else:
        for k in ("earnings", "value", "quality"):
            note(k, False, "income statements not downloaded")

    # stock-level flows
    fl = D["flows"]
    if fl is not None:
        f = sg.score_flows(fl["foreign_net"], fl["trust_net"], fl["margin_bal"], fl.get("sbl_bal"), D["shares"], universe=L)
        S["flows"], S["foreign"], S["foreign_raw"], S["trust"], S["margin_low"] = (
            f["score"], f["foreign"], f["foreign_raw"], f["trust"], -f["margin"])
        sh = D["shares"]
        # month-on-month change of the balances, as published one trading day before month-end
        mom_chg = lambda bal: sg.within(M_(sg.at_month_end(bal.shift(1)), months, cols).diff() / sh, L)
        X["margin_mom"], X["short_mom"] = mom_chg(fl["margin_bal"]), mom_chg(fl["short_bal"])
        S["short_chg"] = sg.within(M_(sg.change_at_month_end(fl["short_bal"].shift(1), 20), months, cols) / sh, L)
        note("flows", True, "" if "sbl_bal" in fl else "borrowed-share short balance missing: foreign buying not adjusted")
    else:
        note("flows", False, "institutional / margin data not downloaded")
    note("fcf_ev", False, "not built: the capital-expenditure field differs by company and period")

    S = {k: sg.within(M_(v, months, cols), L).replace([np.inf, -np.inf], np.nan) for k, v in S.items()}
    X = {k: M_(v, months, cols) for k, v in X.items()}
    return S, X, pd.DataFrame(status).set_index("item")


def M_(x: pd.DataFrame, months, cols) -> pd.DataFrame:
    return x.reindex(index=months, columns=cols)


# ========================================================================== tables
def t_coverage(D: dict, S: dict) -> pd.DataFrame:
    """Number of stocks with a value each December, per signal and universe. No returns."""
    dec = D["months"][D["months"].month == 12]
    rows = {}
    for k, s in S.items():
        for u, U in D["U"].items():
            rows[k, u] = (s.notna() & U).loc[dec].sum(axis=1).set_axis(dec.year)
    out = pd.DataFrame(rows).T
    out.index.names = ["signal", "universe"]
    return out


def ports_and_ics(D: dict, S: dict, names=None, universes=None) -> tuple[dict, dict]:
    """Long-short returns (by holding month) and rank ICs (index moved to the holding
    month) of each signal in each universe, computed once and reused by the tables."""
    cfg, fwd = D["cfg"], D["ret"].shift(-1)
    P, IC = {}, {}
    for u in universes or D["U"]:
        for k in names or S:
            if k not in S:
                continue
            P[k, u] = ev.ls_returns(S[k], D["ret"], D["U"][u], pct=cfg.pct, ret_short=D["ret_short"])
            ic = ev.rank_ic(S[k], fwd, D["U"][u])
            ic.index = ic.index + 1
            IC[k, u] = ic
    return P, IC


def t_single(D: dict, S: dict, P: dict, IC: dict) -> pd.DataFrame:
    cfg, rows = D["cfg"], []
    for (k, u), port in P.items():
        if port.empty:
            continue
        buy, sell, _ = D["cost"][u]
        for pname, (a, b) in PERIODS.items():
            r = ev.factor_report(S[k], D["ret"], buy, sell, D["U"][u], D["mkt_x"], cfg.borrow, a, b, cfg.lags,
                                 port=port, ic=IC[k, u])
            if not r:
                continue
            pc = rm.cut(port, a, b)
            w = (pc["W"] - pc["TO_W"].fillna(pc["TO_W"].mean()) * (buy + sell)).dropna()   # the long leg alone, after costs
            ws = stats(w, D["mkt"], D["rf"])
            ms = rm.perf(D["mkt"].reindex(w.index), D["rf"])
            r.update({"W_net_ann_%": ws.get("ann_return_%"), "W_sharpe": ws.get("sharpe"), "W_max_dd_%": ws.get("max_dd_%"),
                      "W_alpha_%/mo": ws["alpha_%/mo"], "W_alpha_t": ws["alpha_t"],
                      "mkt_ann_%": ms.get("ann_return_%"), "mkt_sharpe": ms.get("sharpe")})
            if k.startswith("lowvol"):       # its own pass line: market-adjusted alpha and a better Sharpe than the market
                r["pass_lowvol"] = bool(r.get("pass_alpha")) and bool(ws.get("sharpe", np.nan) > ms.get("sharpe", np.nan))
            rows.append({"signal": k, "kind": kind(k), "universe": u, "period": pname,
                         "first_month": str(pc.index[0]), **r})
    return pd.DataFrame(rows).set_index(["signal", "universe", "period"]) if rows else pd.DataFrame()


def t_corr(D: dict, P: dict, universe: str) -> pd.DataFrame:
    """Correlation of the gross long-short returns, with the market and 0050."""
    d = {k: p["WML"] for (k, u), p in P.items() if u == universe and not p.empty}
    d["market_excess"] = D["mkt_x"]
    for k, v in D["bench"].items():
        d[k] = v
    return pd.DataFrame(d).corr(min_periods=24)


def t_year(D: dict, P: dict, year: int = 2008) -> pd.DataFrame:
    """Compounded gross return of each long-short portfolio and of its legs in one year."""
    rows = {}
    mk = D["mkt"][D["mkt"].index.year == year]
    for (k, u), p in P.items():
        x = p[p.index.year == year] if not p.empty else p
        if len(x) < 6:
            continue
        c = lambda s: ((1 + s.dropna()).prod() - 1) * 100
        rows[k, u] = {"months": len(x), "WML_%": c(x["WML"]), "W_%": c(x["W"]), "L_%": c(x["L"]),
                      "market_%": c(mk.reindex(x.index)), "worst_month_WML_%": x["WML"].min() * 100}
    return pd.DataFrame(rows).T


def t_ic_horizon(D: dict, S: dict, names: list[str], universe: str = TRADE, horizons: int = 6) -> pd.DataFrame:
    """Mean rank IC between each signal and the return 1 .. 6 months later (full sample)."""
    rows = {}
    for k in names:
        if k not in S:
            continue
        row = {}
        for h in range(1, horizons + 1):
            ic = ev.rank_ic(S[k], D["ret"].shift(-h), D["U"][universe])
            row[f"ic_m{h}"], row[f"t_m{h}"] = ic.mean(), rm.nw_t(ic, D["cfg"].lags)
        rows[k] = row
    return pd.DataFrame(rows).T


def t_decay(D: dict, S: dict, names: list[str]) -> pd.DataFrame:
    """Cumulative long-short return and rank IC 1 .. 60 trading days after the month-end,
    and the half-life. half_life_1st_half repeats it on the first half of the months.
    For information only: no rule uses the half-life (a signal that fades too quickly for
    a monthly rebalance has a small one-month IC, so its expected return does not cover
    the trading cost)."""
    cfg, rows = D["cfg"], {}
    for u, U in D["U"].items():
        for k in names:
            if k not in S:
                continue
            sp, ic = ev.decay_daily(S[k], D["dret"], universe=U, pct=cfg.pct)
            sp, ic = sp.dropna(how="all"), ic.dropna(how="all")
            if sp.empty:
                continue
            hl = ev.half_life(sp.mean())
            rows[k, u] = {"months": len(sp), **{f"ls_%_d{h}": sp[h].mean() * 100 for h in sp.columns},
                          **{f"ic_d{h}": ic[h].mean() for h in ic.columns},
                          "half_life_days": hl, "half_life_1st_half": ev.half_life(sp.iloc[: len(sp) // 2].mean())}
    out = pd.DataFrame(rows).T
    out.index.names = ["signal", "universe"]
    return out


# ----------------------------------------------------------------- reversal conditions
REV_COND = {  # name: (matrix in X or S, what is expected among last month's losers)
    "B_mom": ("rev_mom", "none on its own (see the four cells)"),
    "B_dyoy": ("rev_dyoy", "higher (growth not slowing) -> stronger rebound"),
    "C_volume": ("turnover_abn", "higher (selling climax) -> stronger rebound"),
    "D_margin": ("margin_mom", "lower (margin positions cut) -> stronger rebound"),
    "E_short": ("short_mom", "higher (shorts to cover) -> stronger rebound; the opposite is also plausible"),
}


def t_reversal(D: dict, S: dict, X: dict) -> dict[str, pd.DataFrame]:
    """Conditions A-E among last month's biggest losers (the long side) and biggest
    winners (the short side, opposite expectations). The rank IC is the main evidence."""
    cfg, base = D["cfg"], X["reversal_raw"]
    Z = {**S, **X}
    sides = {"long_losers": (True, D["ret"].shift(-1)), "short_winners": (False, D["ret_short"].shift(-1))}
    A, B, C = {}, {}, {}
    for u, U in D["U"].items():
        n_groups, min_n = (3, 15) if u != PRIMARY else (2, 10)
        every = basket(U & base.notna(), D["ret"])
        for side, (top, fwd) in sides.items():
            sub = extreme(base, U, cfg.pct, top)
            x = basket(sub, D["ret"] if top else D["ret_short"])
            d = (x - every).dropna()
            A[u, side] = {"months": len(d), "names": sub.sum(axis=1).reindex(x.index - 1).mean(),
                          "next_month_%": x.mean() * 100, "universe_%": every.reindex(x.index).mean() * 100,
                          "minus_universe_%": d.mean() * 100, "t": rm.nw_t(d, cfg.lags)}
            for name, (key, expect) in REV_COND.items():
                if key not in Z:
                    continue
                rep = ev.condition_report(Z[key], fwd, sub, n_groups, min_n, cfg.lags)
                B[u, side, name] = {"expected_for_losers": expect, **rep}
            if "rev_mom" in Z and "rev_dyoy" in Z:
                cells = ev.cells_by_sign(Z["rev_mom"], Z["rev_dyoy"], fwd, sub, min_n=min_n, lags=cfg.lags)
                if cells:
                    C[u, side] = cells
    name = lambda d, n: pd.DataFrame(d).T.rename_axis(n) if d else pd.DataFrame()
    return {"reversal_A_baseline": name(A, ["universe", "side"]),
            "reversal_BE_conditions": name(B, ["universe", "side", "condition"]),
            "reversal_B_cells": name(C, ["universe", "side"])}


# ------------------------------------------------------------------ factor-specific
def t_revenue_timing(D: dict, S: dict) -> pd.DataFrame:
    """Monthly-revenue momentum traded at the month-end (about 20 days after the figures
    are out) versus at the close of the trading day after the legal deadline. Both arms
    use returns compounded from daily data with the same rules, so only the timing differs."""
    if "revenue" not in S:
        return pd.DataFrame()
    cfg, idx = D["cfg"], D["dret"].index
    arms = {"month_end": (sg.month_ends(idx), 0), "after_deadline": (entry_dates(idx), 1)}
    rows = {}
    for u, U in D["U"].items():
        buy, sell, _ = D["cost"][u]
        for arm, (dates, lag) in arms.items():
            r = returns_between(D["dret"], dates)
            # mid-month entry: the stock must already be listed at the previous month-end
            p = ev.ls_returns(S["revenue"], r, U.shift(lag, fill_value=False), pct=cfg.pct)
            if p.empty:
                continue
            net = ev.net_ls(p, buy, sell, cfg.borrow)
            rows[u, arm] = {"months": len(p), "gross_%/mo": p["WML"].mean() * 100, "gross_t": rm.nw_t(p["WML"], cfg.lags),
                            "net_%/mo": net.mean() * 100, "net_t": rm.nw_t(net, cfg.lags),
                            "turnover_%/mo_per_leg": (p["TO_W"] + p["TO_L"]).mean() * 50,
                            "entry_day_of_month": float(np.mean([d.day for d in dates]))}
    return pd.DataFrame(rows).T.rename_axis(["universe", "entry"])


def t_price_first(D: dict, S: dict, X: dict) -> pd.DataFrame:
    """Has the price already reacted? a = the news (high / low), b = the past return
    (high / low), independent median splits. See evaluate.two_by_two for the columns."""
    cfg, fwd = D["cfg"], D["ret"].shift(-1)
    pairs = [("rev_yoy3", "past3m"), ("sue", "past3m"), ("foreign", "past20d"), ("trust", "past20d")]
    rows = {}
    for u, U in D["U"].items():
        for a, b in pairs:
            if a in S:
                r = ev.two_by_two(S[a], X[b], fwd, U, lags=cfg.lags)
                if r:
                    rows[u, a, b] = r
    return pd.DataFrame(rows).T.rename_axis(["universe", "a", "b"])


def t_value_trap(D: dict, S: dict, X: dict) -> dict[str, pd.DataFrame]:
    """Cheap x quality and cheap x past return (two-by-two), and the fixed screen:
    cheapest 20% among stocks in the better half of quality (within industry) whose
    return over months m-11 .. m-1 is positive, against the cheapest 20% of everything."""
    if "value" not in S or "quality" not in S:
        return {}
    cfg, fwd = D["cfg"], D["ret"].shift(-1)
    cells, screen = {}, {}
    for u, U in D["U"].items():
        for name, b in (("quality", S["quality"]), ("past12_1", X["past12_1"])):
            r = ev.two_by_two(S["value"], b, fwd, U, lags=cfg.lags)
            if r:
                cells[u, "value", name] = r
        val = sg.within(S["value"], U)
        good = (sg.pct_rank(sg.within(S["quality"], U), D["groups"]) > 0.5) & (X["past12_1"] > 0)
        legs = {"cheapest_20%": sg.pct_rank(val) > 0.8, "screened_cheapest_20%": sg.pct_rank(val.where(good)) > 0.8,
                "universe": val.notna()}
        x = {k: basket(v, D["ret"]) for k, v in legs.items()}
        for k, v in x.items():
            screen[u, k] = {"names": legs[k].sum(axis=1).mean(), **stats(v, D["mkt"], D["rf"])}
        d = (x["screened_cheapest_20%"] - x["cheapest_20%"]).dropna()
        screen[u, "screened_minus_plain"] = {"months": len(d), "diff_%/mo": d.mean() * 100, "diff_t": rm.nw_t(d, cfg.lags)}
    return {"value_trap_cells": pd.DataFrame(cells).T.rename_axis(["universe", "a", "b"]),
            "value_trap_screen": pd.DataFrame(screen).T.rename_axis(["universe", "portfolio"])}


# ----------------------------------------------------- composite, alpha and portfolios
def cut_inputs(D: dict, P: dict) -> pd.DataFrame:
    """Net long-short return of each slow factor in the primary universe (low volatility:
    after taking out the market, with a beta known at the time). Input of the cut rule."""
    cfg = D["cfg"]
    buy, sell, _ = D["cost"][PRIMARY]
    net = {}
    for k in SLOW:
        p = P.get((k, PRIMARY))
        if p is None or p.empty:
            continue
        x = ev.net_ls(p, buy, sell, cfg.borrow)
        net[k] = market_adjusted(x, D["mkt_x"]) if k == "lowvol" else x
    return pd.DataFrame(net)


def composites(D: dict, S: dict, P: dict) -> dict:
    """slow_all: the slow factors, equal weight. slow_wf: the same after the walk-forward
    cut rule. slow_fast: slow_all plus the fast signals, as two more ranks.
    A factor a stock does not have counts as the middle rank (cfg.missing_rank)."""
    cfg = D["cfg"]
    slow = {k: S[k] for k in SLOW if k in S}
    kw = dict(missing=cfg.missing_rank, min_factors=1 if cfg.missing_rank is not None else min(cfg.min_factors, len(slow)))
    net = cut_inputs(D, P)
    keep = wf.cut_decisions(net, lags=cfg.lags, window=cfg.ic_window) if net.notna().any().any() else None
    fast = [k for k in FAST if k in S]
    C = {"slow_all": sg.composite(slow, **kw), "slow_wf": wf.wf_composite(slow, keep, **kw)}
    if fast:
        C["slow_fast"] = sg.composite({**slow, **{k: S[k] for k in fast}}, **kw)
    return {"C": C, "keep": keep, "fast": fast, "cut_net": net}


def build_alpha(D: dict, S: dict, X: dict, comp: pd.DataFrame, fast: list[str],
                residual: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected return of each stock over the life of the signals (Grinold: IC x residual
    volatility x score), every IC estimated from earlier months only, floored at zero.
    The composite uses the IC summed over 6 months; reversal one month; flows 6 months.
    residual=True: a fast signal only adds the part the composite does not explain that
    month (sg.orthogonal), and its IC is measured on that part, so what the two share is
    not counted twice. residual=False adds the fast signal as it is (comparison)."""
    cfg, U, vol = D["cfg"], D["U"][TRADE], X["resvol"]
    est = lambda s, h: wf.cum_ic(s, D["ret"], h, U, cfg.ic_min_months, cfg.ic_window)
    ic = est(comp, cfg.ic_horizons)
    alpha = wf.expected_return(comp, ic["cum"].clip(lower=0), vol)
    used = {"composite": ic["cum"]}
    for k in fast:
        sig = sg.orthogonal(S[k], comp, U) if residual else S[k]
        ick = est(sig, 1 if k == "reversal" else cfg.ic_horizons)["cum"]
        extra = wf.expected_return(sig, ick.clip(lower=0), vol).reindex(index=alpha.index, columns=alpha.columns)
        alpha = alpha + extra.fillna(0)             # a stock without the fast signal keeps its slow alpha
        used[k] = ick
    return alpha, pd.DataFrame(used)


def run_book(D: dict, alpha: pd.DataFrame, capital: float, short: bool = False, **kw) -> dict | None:
    """One simulated book. Long: buys in the tradable universe. Short: the lowest alphas
    among big caps; it cannot open at limit-down nor cover at limit-up."""
    cfg = D["cfg"]
    ready = alpha.notna().sum(axis=1) >= cfg.n
    if not ready.any():
        return None
    costs = pf.Costs(fee=cfg.fee, fee_discount=cfg.fee_discount, tax=cfg.tax, spread_ticks=cfg.spread_ticks,
                     impact_k=cfg.impact_k, borrow=cfg.borrow)
    extra = kw
    if short:
        kw = dict(ret=D["ret_short"], eligible=D["U"][PRIMARY] & D["U"][TRADE], no_buy=D["no_short"], no_sell=D["no_cover"])
    else:
        kw = dict(ret=D["ret"], eligible=D["U"][TRADE], no_buy=D["no_buy"], no_sell=D["no_sell"])
    return pf.simulate(alpha, price=D["close_m"], adv=D["adv"], dvol=D["dvol"], industry=D["groups_official"],
                       capital=capital, n=cfg.n, gamma=cfg.gamma, costs=costs, max_adv_share=cfg.max_adv_share,
                       max_industry=cfg.max_industry, short=short, start=str(ready.idxmax() + 1), **kw, **extra)


def book_row(D: dict, res: dict | None, short: bool = False) -> dict:
    if res is None or res["monthly"].empty:
        return {}
    m = res["monthly"]
    x = m["net"].astype(float)
    s = stats_ls(x, D["mkt_x"]) if short else {**stats(x, D["mkt"], D["rf"]), **goal(x, D["mkt"])}
    mk = rm.perf(D["mkt"].reindex(x.index), D["rf"])
    out = {"first_month": str(m.index[0]), **s, "mkt_ann_%": mk.get("ann_return_%"), "mkt_max_dd_%": mk.get("max_dd_%"),
           "names": m["names"].mean(), "turnover_%/mo": m["turnover"].mean() * 100, "cost_%/mo": m["cost"].mean() * 100,
           "cash_%": m["cash_share"].mean() * 100}
    for k, v in D["bench"].items():                  # tradable benchmarks over the months they share with the book
        b = rm.perf(v.reindex(x.index), D["rf"])
        out.update({f"{k}_months": b.get("months"), f"{k}_ann_%": b.get("ann_return_%"), f"{k}_max_dd_%": b.get("max_dd_%")})
    return out


def t_portfolio(D: dict, alphas: dict, capitals=CAPITALS) -> tuple[pd.DataFrame, dict]:
    """Each alpha version x capital: the long-only book, the short book, and both together
    (100% long + 100% short, the two books brought back to equal size every month)."""
    rows, books = {}, {}
    for a, alpha in alphas.items():
        for cap in capitals:
            lo, sh = run_book(D, alpha, cap), run_book(D, alpha, cap, short=True)
            books[a, cap] = (lo, sh)
            rows[a, cap, "long_only"] = book_row(D, lo)
            rows[a, cap, "short_book"] = book_row(D, sh, short=True)
            if lo is not None and sh is not None:
                both = (lo["monthly"]["net"] + sh["monthly"]["net"]).dropna().astype(float)
                rows[a, cap, "long_short"] = {"first_month": str(both.index[0]) if len(both) else "", **stats_ls(both, D["mkt_x"])}
    out = pd.DataFrame({k: v for k, v in rows.items() if v}).T
    out.index.names = ["alpha", "capital", "book"]
    return out, books


def t_penalty(D: dict, alpha: pd.DataFrame, capitals=CAPITALS) -> pd.DataFrame:
    """Soft penalty against no penalty: return given up, cost saved, and what the blocked
    stocks did afterwards."""
    rows = {}
    D0 = derive(D, replace(D["cfg"], gamma=0.0))
    for cap in capitals:
        a, b = run_book(D, alpha, cap), run_book(D0, alpha, cap)
        if a is not None and b is not None:
            rows[cap] = pf.penalty_report(a, b, D["ret"])
    return pd.DataFrame(rows).T.rename_axis("capital")


SENSITIVITY = [  # (setting, neighbouring values, capital, short book?)
    ("gamma", (0.5, 1.5, 2.0), 1e6, False), ("n", (20, 50), 1e6, False), ("adv_floor", (1e7, 5e7), 1e6, False),
    ("max_industry", (0.2, 0.4), 1e6, False), ("fee_discount", (0.6, 1.0), 1e6, False),
    ("spread_ticks", (0.0, 1.0), 1e6, False), ("delist_long", (0.0, -1.0), 1e6, False),
    ("merger_exit", (-0.30,), 1e6, False),
    ("ic_window", (60,), 1e6, False), ("missing_rank", (None,), 1e6, False), ("pct", (0.2, 0.3), 1e6, False),
    ("ic_horizons", (3, 12), 1e6, False), ("ic_min_months", (24, 60), 1e6, False), ("big_cap", (5e9, 2e10), 1e6, False),
    ("max_adv_share", (0.1, 0.3), 1e9, False), ("impact_k", (0.5, 2.0), 1e9, False),
    ("borrow", (0.0, 0.06), 1e7, True),
]


def pipeline(D: dict, S: dict, X: dict, P: dict | None = None) -> dict:
    """Cut rule -> composites -> alphas, for D's settings."""
    if P is None:
        P, _ = ports_and_ics(D, S, SLOW, [PRIMARY])
    cp = composites(D, S, P)
    A, used = {}, {}
    A["slow_wf"], used["slow_wf"] = build_alpha(D, S, X, cp["C"]["slow_wf"], [])
    A["slow_all"], used["slow_all"] = build_alpha(D, S, X, cp["C"]["slow_all"], [])
    if cp["fast"]:
        A["slow_wf+fast"], used["slow_wf+fast"] = build_alpha(D, S, X, cp["C"]["slow_wf"], cp["fast"])
        A["slow_wf+fast_raw"], used["slow_wf+fast_raw"] = build_alpha(D, S, X, cp["C"]["slow_wf"], cp["fast"], residual=False)
    return {**cp, "alpha": A, "ic_used": used}


def t_turnover_month(books: dict) -> pd.DataFrame:
    """Average turnover (%) of each long-only book by calendar month of the holding month.
    The cut decisions change in January: this shows whether trading bunches up there."""
    rows = {}
    for (a, cap), (lo, _) in books.items():
        if lo is not None and not lo["monthly"].empty:
            t = lo["monthly"]["turnover"].astype(float)
            rows[a, cap] = (t.groupby(t.index.month).mean() * 100).rename(lambda m: f"m{m:02d}")
    return pd.DataFrame(rows).T.rename_axis(["alpha", "capital"])


def t_benchmark(D: dict) -> pd.DataFrame:
    """The market series in use against the self-built one and the tradable benchmarks,
    over the months they share with the market series."""
    rows = {}
    for k, v in {"market_in_use": D["mkt"], "self_built_all_stocks": D["mkt_own"], **D["bench"]}.items():
        d = pd.concat([v.rename("x"), D["mkt"].rename("m")], axis=1).dropna()
        if len(d) < 12:
            continue
        diff = d["x"] - d["m"]
        rows[k] = {**rm.perf(d["x"], D["rf"]), "first_month": str(d.index[0]), "market_ann_%": rm.perf(d["m"], D["rf"]).get("ann_return_%"),
                   "corr_monthly": d["x"].corr(d["m"]), "tracking_error_%/yr": diff.std() * np.sqrt(12) * 100}
    return pd.DataFrame(rows).T


def t_sensitivity(D: dict, S: dict, X: dict, base: dict) -> pd.DataFrame:
    """One setting at a time moved to a neighbouring value, everything else unchanged.
    Base alpha: slow factors after the walk-forward cut. Read it for stability, not to pick."""
    cfg, rows = D["cfg"], {}
    done = {}
    for field, values, cap, short in SENSITIVITY:
        key = (cap, short)
        if key not in done:
            done[key] = book_row(D, run_book(D, base["alpha"]["slow_wf"], cap, short), short)
        rows[field, "base=" + str(getattr(cfg, field)), cap, short] = done[key]
        for v in values:
            D2 = derive(D, replace(cfg, **{field: v}))
            alpha = base["alpha"]["slow_wf"] if field in SIM_ONLY else pipeline(D2, S, X)["alpha"]["slow_wf"]
            rows[field, str(v), cap, short] = book_row(D2, run_book(D2, alpha, cap, short), short)
    out = pd.DataFrame({k: v for k, v in rows.items() if v}).T
    out.index.names = ["setting", "value", "capital", "short_book"]
    keep = ["first_month", "ann_return_%", "ann_vol_%", "sharpe", "max_dd_%", "beta", "alpha_%/mo", "alpha_t",
            "names", "turnover_%/mo", "cost_%/mo"]
    return out[[c for c in keep if c in out.columns]]


def t_single_sensitivity(D: dict, S: dict, X: dict) -> pd.DataFrame:
    """Single-factor net return and t in the primary universe when the winner / loser
    fraction is 10 / 20 / 30%, and reversal without skipping the last day."""
    cfg, rows = D["cfg"], {}
    buy, sell, _ = D["cost"][PRIMARY]
    names = [k for k in SLOW + FAST if k in S]

    def add(tag, name, sig, pct):
        p = ev.ls_returns(sig, D["ret"], D["U"][PRIMARY], pct=pct, ret_short=D["ret_short"])
        if p.empty:
            return
        net = ev.net_ls(p, buy, sell, cfg.borrow)
        h1, h2 = ev.halves(net)
        rows[name, tag] = {"gross_%/mo": p["WML"].mean() * 100, "net_%/mo": net.mean() * 100,
                           "net_t": rm.nw_t(net, cfg.lags), "net_1st_half_%/mo": h1 * 100, "net_2nd_half_%/mo": h2 * 100}

    for pct in (0.1, 0.2, 0.3):
        for k in names:
            add(f"pct={pct}", k, S[k], pct)
    no_skip = sg.within(-sg.month_return_skip_last(D["dret"], 0).clip(*rm.RET_CLIP).reindex(
        index=D["months"], columns=D["cols"]), D["listed"])
    add("no_skip_day", "reversal_raw", no_skip, cfg.pct)
    add(f"skip_{cfg.reversal_skip}_day", "reversal_raw", X["reversal_raw"], cfg.pct)
    return pd.DataFrame(rows).T.rename_axis(["signal", "variant"])


# --------------------------------------------------------------------- market state
def state_indicators(D: dict, days: int = 5) -> dict[str, tuple[pd.Series, bool]]:
    """name -> (daily indicator on the market's trading days, low is bad?)."""
    mk, cal = D["market"], D["mkt_d"].dropna().index
    out = {}
    if "inst_net" in mk and "turnover" in mk:
        d = pd.concat([mk["inst_net"], mk["turnover"]], axis=1, keys=["net", "turn"]).reindex(cal).dropna()
        out["B_institutional_selling"] = (d["net"].rolling(days).sum() / d["turn"].rolling(days).sum(), True)
    if "usdtwd" in mk:
        fx = mk["usdtwd"].reindex(cal.union(mk["usdtwd"].index)).ffill(limit=3).reindex(cal)
        out["C_twd_weakening"] = (fx.pct_change(days, fill_method=None), False)      # up = TWD weaker
    if "fut_foreign" in mk:
        out["D_foreign_futures"] = (mk["fut_foreign"].reindex(cal).dropna().diff(days), True)
    return out


def trend_flag(mkt: pd.Series, n: int = 10) -> pd.Series:
    """Indicator A: the total-return level at the month-end is below its own average of
    the last `n` month-ends (this one included)."""
    level = (1 + mkt.dropna()).cumprod()
    avg = level.rolling(n, min_periods=n).mean()
    return (level < avg).astype(float).where(avg.notna())            # 1 / 0 / missing


def month_flag_report(flag: pd.Series, y: pd.Series, lags: int = 3) -> dict:
    """y in the month AFTER a flagged month-end against the other months."""
    d = pd.concat([flag.shift(1).rename("f"), y.rename("y")], axis=1).dropna()
    if d.empty:
        return {}
    f = d["f"].astype(bool)
    out = {"months": len(d), "bad_months": int(f.sum()), "episodes": int((f & ~f.shift(1, fill_value=False)).sum())}
    if f.any() and not f.all():
        z = d["y"] * (f / f.mean() - (~f) / (~f).mean())
        out.update({"next_%_bad": d.loc[f, "y"].mean() * 100, "next_%_other": d.loc[~f, "y"].mean() * 100,
                    "diff_t": rm.nw_t(z, lags)})
    return out


def t_state(D: dict, P: dict) -> dict[str, pd.DataFrame]:
    """Each market-state indicator on its own (worst 10% of its own last 3 years), the
    neighbouring settings, B+C+D together, and low volatility after flagged month-ends."""
    cfg, md = D["cfg"], D["mkt_d"].dropna()
    base = dict(days=5, tail=0.10, window=750)
    variants = [("base", base)] + [(f"{k}={v}", {**base, k: v}) for k, vs in
                                   (("days", (3, 10)), ("tail", (0.05, 0.20)), ("window", (500, 1250))) for v in vs]
    daily, flags = {}, {}
    for tag, v in variants:
        for name, (x, low_bad) in state_indicators(D, v["days"]).items():
            rep = ev.state_report(x, md, 20, v["tail"], low_bad, v["window"])
            if rep:
                daily[name, tag] = rep
            if tag == "base":
                flags[name] = ev.state_flag(x.reindex(md.index), v["tail"], low_bad, v["window"])
    if len(flags) == 3:
        f = pd.concat(flags, axis=1)
        both = f.all(axis=1).where(f.notna().all(axis=1))
        rep = ev.flag_report(both, md, 20)
        if rep:
            daily["B+C+D_together", "base"] = rep
            flags["B+C+D_together"] = both

    monthly = {}
    for n in (10, 8, 12):
        rep = month_flag_report(trend_flag(D["mkt"], n), D["mkt"], cfg.lags)
        if rep:
            monthly["A_trend", f"months={n}", "market"] = rep
    lv = P.get(("lowvol", PRIMARY))
    if lv is not None and not lv.empty:
        buy, sell, _ = D["cost"][PRIMARY]
        adj = market_adjusted(ev.net_ls(lv, buy, sell, cfg.borrow), D["mkt_x"])
        mflags = {"A_trend": trend_flag(D["mkt"], 10)}
        for name, f in flags.items():           # published after the close: read one trading day before month-end
            mflags[name] = sg.at_month_end(f.shift(1).to_frame("f"))["f"]
        for name, f in mflags.items():
            rep = month_flag_report(f, adj, cfg.lags)
            if rep:
                monthly[name, "base", "lowvol_market_adjusted"] = rep
    return {"state_daily": pd.DataFrame(daily).T.rename_axis(["indicator", "setting"]),
            "state_monthly": pd.DataFrame(monthly).T.rename_axis(["indicator", "setting", "series"])}


# ============================================================================= run
def run(raw: dict, cfg: Cfg, stage: str = "all", out_dir: Path | None = None, log=print) -> dict[str, pd.DataFrame]:
    """Produces every table (name -> DataFrame) and, with out_dir, writes them as csv."""
    t0 = time.time()
    step = lambda msg: log(f"[{time.time() - t0:6.0f}s] {msg}")
    T: dict[str, pd.DataFrame] = {}
    step(f"prepare (data cut at {cfg.end})")
    D = prepare(raw, cfg)                            # from here on only D is used, never raw
    assert D["dret"].index.max() <= D["end"].end_time and D["months"].max() <= D["end"]
    step("signals")
    S, X, status = build_signals(D)
    T["status"] = status
    T["costs_flat"] = pd.DataFrame(D["cost"], index=["buy", "sell", "of_which_spread_each_way"]).T * 100
    T["coverage"] = t_coverage(D, S)
    T["benchmark"] = t_benchmark(D)
    T["exit_months"] = D["exits"].assign(known=lambda d: d["used"].notna()).set_index(["month", "stock"])
    if stage != "coverage":
        step("single-factor tests")
        P, IC = ports_and_ics(D, S)
        T["single_factor"] = t_single(D, S, P, IC)
        for u in D["U"]:
            T[f"correlation_{u}"] = t_corr(D, P, u)
        T["year_2008"] = t_year(D, P, 2008)
        main = [k for k in SLOW + FAST + ["mom"] if k in S]
        T["ic_by_month"] = t_ic_horizon(D, S, main)
        T["single_factor_sensitivity"] = t_single_sensitivity(D, S, X)
        if not D["groups"].equals(D["groups_official"]):         # the same factors ranked inside the official industries
            S0, _, _ = build_signals({**D, "groups": D["groups_official"]})
            S0 = {k: S0[k] for k in ("value", "quality", "reversal_industry", "turnover_low") if k in S0}
            P0, IC0 = ports_and_ics(D, S0)
            T["single_factor_official_industry"] = t_single(D, S0, P0, IC0)
        step("conditions and two-by-two tables")
        T.update(t_reversal(D, S, X))
        T["revenue_timing"] = t_revenue_timing(D, S)
        T["price_first"] = t_price_first(D, S, X)
        T.update(t_value_trap(D, S, X))
        step("decay and half-life")
        T["decay"] = t_decay(D, S, [k for k in S if kind(k) != "compare" or k in ("mom", "reversal_residual", "reversal_industry")])
        step("composite, cut rule, alpha")
        base = pipeline(D, S, X, P)
        if base["keep"] is not None:
            T["cut_decisions"] = base["keep"]
        Pc, ICc = ports_and_ics(D, base["C"])
        T["composite"] = t_single(D, base["C"], Pc, ICc)
        T["alpha_ic_used"] = pd.concat(base["ic_used"], axis=1).loc[lambda d: d.index.month == 12]
        step("portfolios")
        T["portfolio"], books = t_portfolio(D, base["alpha"])
        T["turnover_by_month"] = t_turnover_month(books)
        T["penalty"] = t_penalty(D, base["alpha"]["slow_wf"])
        step("sensitivity")
        T["portfolio_sensitivity"] = t_sensitivity(D, S, X, base)
        step("market state")
        T.update(t_state(D, P))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in T.items():
            v.to_csv(out_dir / f"{k}.csv", encoding="utf-8-sig")
        (out_dir / "config.json").write_text(json.dumps({**asdict(cfg), "stage": stage, "last_month": str(D["end"]),
                                                         "stocks": len(D["cols"])}, indent=2), encoding="utf-8")
    step(f"done: {len(T)} tables")
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["coverage", "all"], default="all")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--open-holdout", action="store_true", help="only for the single final run")
    ap.add_argument("--allow-partial", action="store_true", help="run although some downloads are unfinished")
    ap.add_argument("--tag", default=None, help="output folder name under output/step1")
    a = ap.parse_args()
    cfg = Cfg(end=a.end, open_holdout=a.open_holdout)
    wf.guard_holdout(cfg.end, cfg.open_holdout)
    st = download_status()
    print(st.to_string())
    if a.stage == "all" and not st["complete"].all() and not a.allow_partial:
        sys.exit("Some downloads are unfinished. The tests are meant to run once, on complete data: "
                 "wait, or use --stage coverage (no returns), or --allow-partial.")
    T = run(load_raw(), cfg, a.stage, OUT / (a.tag or f"{a.stage}_to_{cfg.end}"))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(T["status"].to_string())
    if a.stage == "coverage":
        print(T["coverage"].to_string())


if __name__ == "__main__":
    main()
