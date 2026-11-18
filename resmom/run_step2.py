"""Step 2: the two sleeves and their risk control (FACTOR_SPEC.md, 步驟二).

Stable sleeve      0050 half + residual-momentum winners half (overlay_tw.py's
                   "50/50: 0050 + score-wtd winners"), rebuilt on the cleaned data.
Aggressive sleeve  the step-1 multi-factor book (slow factors after the walk-forward cut),
                   equal weight (frozen) and weights in inverse proportion to volatility.
Risk control       volatility target (target = goal drawdown multiple x market volatility),
                   Grossman-Zhou drawdown control, both decided at month-ends with data up
                   to that day only. Monte Carlo, CVaR and correlations are reports.
Control            the winners replaced by random stocks of the same size class.

Everything runs on step 1's prepare(), so nothing after cfg.end (2023-12) is visible.

    python run_step2.py                 # about 20 minutes
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import data_tw as dt
import resmom as rm
import run_step1 as rs
import signals as sg

OUT = Path(__file__).parent / "output" / "step2"
TD = 252


@dataclass(frozen=True)
class Cfg2:
    start: str = "2009-07"                 # first month both sleeves exist
    # stable sleeve (as in overlay_tw.py)
    winners_pct: float = 0.10
    winners_cap: float = 0.03
    winners_k: int = 3
    etf_weight: float = 0.50
    etf_tax: float = 0.001
    # goals per sleeve: (return vs market in points or None, drawdown multiple of the market)
    stable_dd_mult: float = 0.7
    aggressive_dd_mult: float = 1.2
    stable_ret_gap: float = -0.02           # return >= market - 2 points
    aggressive_ret_gap: float = 0.05        # return >= market + 5 points
    # risk control
    vol_windows: tuple = (20, 60, 120)
    stable_cap: float = 1.0
    aggressive_cap: float = 1.5
    stable_dd_limit: float = 0.20          # Grossman-Zhou D
    aggressive_dd_limit: float = 0.30
    invvol_cap: float = 2.0                 # max position = this x equal weight
    # reports
    mc_draws: int = 10000
    control_draws: int = 200
    seed: int = 20261117


# ============================================================ daily series of a book
def book_daily(D: dict, res: dict) -> pd.Series:
    """Daily net return of a simulated long book from the positions it held at each
    formation day: positions drift with the daily returns, cash earns nothing, the month's
    trading cost is taken on its first day. A holding whose stock stopped trading takes the
    month's exit return (data_tw.exit_returns, as in the monthly book) on that first day."""
    dret, gone, ret = D["dret"], D["gone"], D["ret"]
    per = dret.index.to_period("M")
    out = []
    for m, lg in res["log"].items():
        days = dret.index[per == m + 1]
        if not len(days) or "held" not in lg:
            continue
        V0 = lg["value_before"]
        names = list(lg["held"])
        val = np.array([lg["held"][s] for s in names], dtype=float)
        if names:
            r = dret.loc[days, names].to_numpy(dtype=float)
            r = np.nan_to_num(r)
            g = gone.loc[m + 1, names].to_numpy(dtype=bool) if m + 1 in gone.index else np.zeros(len(names), bool)
            if g.any():
                r[:, g] = 0.0
                r[0, g] = np.nan_to_num(ret.loc[m + 1, names].to_numpy(dtype=float)[g])
            path = val * np.cumprod(1 + r, axis=0)
            tot = path.sum(axis=1) + lg["cash"]
        else:
            tot = np.full(len(days), lg["cash"])
        # held values and cash are after paying the costs, so the first day's return includes them
        dr = np.concatenate([[tot[0] / V0 - 1], tot[1:] / tot[:-1] - 1])
        out.append(pd.Series(dr, index=days))
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def monthly(d: pd.Series) -> pd.Series:
    return rs.to_monthly(d.dropna())


# ============================================================ stable sleeve
def winner_weights(score: pd.DataFrame, universe: pd.DataFrame, pct: float, cap: float) -> dict:
    """Formation month -> weights of the winners (top pct by score inside the universe),
    proportional to the score and limited to `cap` (resmom.cap_weights)."""
    out = {}
    for f, s in score.iterrows():
        s = s.where(universe.loc[f]) if f in universe.index else s
        c = rm.winners_losers(s, pct)
        if c is None:
            continue
        top = s[c[0]]
        w = rm.cap_weights(np.nan_to_num(top.to_numpy(dtype=float)).clip(min=0), cap)
        if w.sum() <= 0:
            w = np.ones(len(top))
        out[f] = pd.Series(w / w.sum(), index=top.index)
    return out


def cohort_daily(D: dict, weights: dict, k: int, cost_rt: float) -> tuple[pd.Series, pd.Series]:
    """Daily return of holding the cohorts formed in the last k months in equal parts,
    reset to target at each month-end, drifting inside the month. Turnover (one way) x
    cost_rt is charged on the first day. Returns (daily return, monthly turnover)."""
    dret, gone, ret = D["dret"], D["gone"], D["ret"]
    per = dret.index.to_period("M")
    out, turn, prev = [], {}, None
    for f in sorted(weights):
        cs = [weights.get(f - j) for j in range(k)]
        if any(c is None for c in cs):
            prev = None
            continue
        tgt = pd.concat(cs, axis=1).fillna(0).mean(axis=1)
        tgt = tgt[tgt > 0]
        days = dret.index[per == f + 1]
        if not len(days):
            continue
        tr = 0.5 * (tgt.subtract(prev, fill_value=0).abs().sum()) if prev is not None else 0.0
        names = list(tgt.index)
        r = np.nan_to_num(dret.loc[days, names].to_numpy(dtype=float))
        g = gone.loc[f + 1, names].to_numpy(dtype=bool) if f + 1 in gone.index else np.zeros(len(names), bool)
        if g.any():
            r[:, g] = 0.0
            r[0, g] = np.nan_to_num(ret.loc[f + 1, names].to_numpy(dtype=float)[g])
        path = tgt.to_numpy() * np.cumprod(1 + r, axis=0)
        tot = path.sum(axis=1)
        dr = np.concatenate([[tot[0] - 1], tot[1:] / tot[:-1] - 1])
        dr[0] -= tr * cost_rt
        out.append(pd.Series(dr, index=days))
        turn[f + 1] = tr
        prev = pd.Series(path[-1] / tot[-1], index=names)
    return (pd.concat(out).sort_index() if out else pd.Series(dtype=float)), pd.Series(turn)


def two_assets(a: pd.Series, b: pd.Series, wa: float, months=(1, 4, 7, 10), cost: float = 0.0) -> pd.Series:
    """Daily return of wa in a and 1 - wa in b, drifting, reset at the first trading day of
    the given months; |traded| x cost charged on the reset day."""
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    out, w, last_m = np.empty(len(d)), wa, None
    A, B = d["a"].to_numpy(), d["b"].to_numpy()
    for i, t in enumerate(d.index):
        c = 0.0
        if last_m is not None and t.month != last_m and t.month in months:
            c = abs(w - wa) * cost
            w = wa
        last_m = t.month
        r = w * A[i] + (1 - w) * B[i]
        out[i] = r - c
        w = w * (1 + A[i]) / (1 + r) if 1 + r != 0 else wa
    return pd.Series(out, index=d.index)


# ============================================================ risk control
def vol_est(d: pd.Series, windows) -> pd.Series:
    """Annualised volatility at each day: mean of the rolling standard deviations over the
    given windows (each needs at least 20 observations)."""
    est = [d.rolling(w, min_periods=min(w, 20)).std() for w in windows]
    return pd.concat(est, axis=1).mean(axis=1) * np.sqrt(TD)


def overlay(sleeve: pd.Series, mkt: pd.Series, rf_d: pd.Series, mult: float | None, cap: float,
            dd_limit: float | None, windows=(20, 60, 120), cost: float = 0.0) -> pd.DataFrame:
    """Risk-controlled version of a daily sleeve return.

    At each month's last trading day the exposure e for the next month is set with data up
    to that day: e = min(cap, mult x market vol / sleeve vol) (mult None: e = 1, no vol
    target), then times the Grossman-Zhou factor max(0, 1 - (1 - D) x peak / value) / D
    of the risk-controlled series itself (dd_limit None: no drawdown control). Inside the
    month the risky part drifts; the rest earns the risk-free rate (negative cash, above
    100%, pays it). |change in e| x cost is charged at the switch.
    Returns columns ret, exposure (at the start of each day)."""
    d = pd.concat([sleeve.rename("s"), mkt.rename("m"), rf_d.rename("rf")], axis=1).dropna(subset=["s", "m"])
    d["rf"] = d["rf"].fillna(0)
    vs, vm = vol_est(d["s"], windows), vol_est(d["m"], windows)
    month_end = d.index.to_series().groupby(d.index.to_period("M")).transform("max") == d.index
    S, RF = d["s"].to_numpy(), d["rf"].to_numpy()
    n = len(d)
    ret, expo = np.empty(n), np.empty(n)
    V, peak, risky = 1.0, 1.0, 1.0
    for i in range(n):
        expo[i] = risky / V
        cash = V - risky
        risky *= 1 + S[i]
        cash *= 1 + RF[i]
        newV = risky + cash
        ret[i] = newV / V - 1
        V = newV
        peak = max(peak, V)
        if month_end.iloc[i] and i < n - 1:
            e_new = 1.0
            if mult is not None:
                ratio = vm.iloc[i] / vs.iloc[i] if vs.iloc[i] > 0 else np.nan
                e_new = min(cap, mult * ratio) if np.isfinite(ratio) else 1.0
            if dd_limit is not None:
                e_new *= max(0.0, 1 - (1 - dd_limit) * peak / V) / dd_limit
            e_new = min(cap, max(0.0, e_new))
            c = abs(e_new - risky / V) * cost
            V *= 1 - c
            ret[i] = (1 + ret[i]) * (1 - c) - 1
            risky = e_new * V
    return pd.DataFrame({"ret": ret, "exposure": expo}, index=d.index)


# ============================================================ evaluation
def dd_daily(d: pd.Series) -> float:
    lv = (1 + d.dropna()).cumprod()
    return float((lv / lv.cummax() - 1).min()) if len(lv) else np.nan


def cvar(d: pd.Series, q: float = 0.95) -> float:
    x = d.dropna()
    cut = x.quantile(1 - q)
    return float(x[x <= cut].mean())


def ann(d: pd.Series) -> float:
    x = d.dropna()
    return float((1 + x).prod() ** (TD / len(x)) - 1) if len(x) else np.nan


def evaluate(d: pd.Series, mkt: pd.Series, rf_d: pd.Series, ret_gap: float | None, dd_mult: float,
             ret_mult: float | None = None) -> dict:
    """Daily series against the market over the days both have. Goal per half:
    return >= market + ret_gap (or >= ret_mult x market) and drawdown <= dd_mult x market's."""
    x = pd.concat([d.rename("p"), mkt.rename("m")], axis=1).dropna()
    if len(x) < 250:
        return {}
    k = len(x) // 2
    out, ok = {}, True
    for tag, s in (("full", x), ("h1", x.iloc[:k]), ("h2", x.iloc[k:])):
        rp, rmk, dp, dm = ann(s["p"]), ann(s["m"]), dd_daily(s["p"]), dd_daily(s["m"])
        out[f"ret_%_{tag}"], out[f"mkt_ret_%_{tag}"] = rp * 100, rmk * 100
        out[f"dd_%_{tag}"], out[f"mkt_dd_%_{tag}"] = dp * 100, dm * 100
        out[f"dd_x_mkt_{tag}"] = dp / dm if dm < 0 else np.nan
        good_r = rp >= (ret_mult * rmk if ret_mult is not None else rmk + ret_gap)
        good_d = dp / dm <= dd_mult if dm < 0 else True
        if tag != "full":
            out[f"goal_{tag}"] = bool(good_r and good_d)
            ok &= bool(good_r and good_d)
    ex = x["p"] - rf_d.reindex(x.index).fillna(0)
    out.update({"start": str(x.index[0].date()), "end": str(x.index[-1].date()),
                "vol_%": x["p"].std() * np.sqrt(TD) * 100, "mkt_vol_%": x["m"].std() * np.sqrt(TD) * 100,
                "sharpe": ex.mean() / ex.std() * np.sqrt(TD), "cvar95_%": cvar(x["p"]) * 100,
                "mkt_cvar95_%": cvar(x["m"]) * 100, "corr_mkt": x["p"].corr(x["m"]), "goal_met": ok})
    return out


def block_bootstrap(n: int, block: int, draws: int, rng) -> np.ndarray:
    """draws x n row positions: blocks of `block` consecutive months, start uniform,
    wrapping around the end (circular), cut to n."""
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(draws, -1)[:, :n]


def mc_report(x: pd.Series, mkt: pd.Series, ret_gap, dd_mult, ret_mult, block: int, draws: int, rng) -> dict:
    """Monthly returns of the series and the market resampled together in blocks. Reports
    percentiles of annual return and maximum drawdown (month-end) and the share of draws
    meeting the goal in both halves."""
    d = pd.concat([x.rename("p"), mkt.rename("m")], axis=1).dropna()
    P, Mk = d["p"].to_numpy(), d["m"].to_numpy()
    n = len(d)
    idx = block_bootstrap(n, block, draws, rng)
    p, m = P[idx], Mk[idx]

    def stats(a):
        lv = np.cumprod(1 + a, axis=1)
        dd = (lv / np.maximum.accumulate(lv, axis=1) - 1).min(axis=1)
        return lv[:, -1] ** (12 / a.shape[1]) - 1, dd

    rp, dp = stats(p)
    k = n // 2
    ok = np.ones(draws, bool)
    for sl in (slice(0, k), slice(k, None)):
        r1, d1 = stats(p[:, sl])
        r2, d2 = stats(m[:, sl])
        good_r = r1 >= (ret_mult * r2 if ret_mult is not None else r2 + ret_gap)
        ok &= good_r & (d1 / np.where(d2 < 0, d2, np.nan) <= dd_mult)
    q = lambda a, v: float(np.nanpercentile(a, v) * 100)
    return {"block": block, "ann_%_p5": q(rp, 5), "ann_%_p50": q(rp, 50), "ann_%_p95": q(rp, 95),
            "maxdd_%_p5": q(dp, 5), "maxdd_%_p50": q(dp, 50), "maxdd_%_p95": q(dp, 95),
            "p_goal_both_halves": float(ok.mean())}


def avg_corr(D: dict, res: dict, days: int = 60) -> pd.Series:
    """Average pairwise correlation of the book's holdings over the `days` days before
    each formation day."""
    dret, out = D["dret"], {}
    for m, lg in res["log"].items():
        names = list(lg.get("held", {}))
        if len(names) < 5:
            continue
        end = m.end_time
        w = dret.loc[:end, names].tail(days)
        c = w.corr().to_numpy()
        iu = np.triu_indices_from(c, 1)
        out[m] = float(np.nanmean(c[iu]))
    return pd.Series(out)


# ============================================================ run
def run(cfg: rs.Cfg, c2: Cfg2, out_dir: Path | None = None, log=print, raw: dict | None = None) -> dict:
    t0 = time.time()
    step = lambda msg: log(f"[{time.time() - t0:6.0f}s] {msg}")
    step(f"step 1 data and alpha (cut at {cfg.end})")
    raw = raw if raw is not None else rs.load_raw()
    D = rs.prepare(raw, cfg)
    S, X, _ = rs.build_signals(D)
    base = rs.pipeline(D, S, X)
    last = D["end"].end_time
    first = pd.Period(c2.start, "M").start_time
    cut = lambda d: d.loc[(d.index >= first) & (d.index <= last)]
    mkt = cut(D["mkt_d"].dropna())
    rf_d = (D["rf"].reindex(mkt.index.to_period("M")).to_numpy() / 21)
    rf_d = pd.Series(rf_d, index=mkt.index)
    etf = raw.get("etf_0050_d")
    etf = cut(etf.loc[etf.index <= last]) if etf is not None else mkt
    buy, sell, _ = D["cost"]["all"]
    T, series = {}, {}

    # ---------------- aggressive sleeve
    step("aggressive sleeve")
    alpha = base["alpha"]["slow_wf"]
    eq = rs.run_book(D, alpha, 1e6)
    dv = D["dvol"]
    elig = D["U"][rs.TRADE]
    med = dv.where(elig).median(axis=1)
    mult = (1 / dv).mul(med, axis=0).clip(upper=c2.invvol_cap)
    iv = rs.run_book(D, alpha, 1e6, size_mult=mult)
    agg = {"equal": cut(book_daily(D, eq)), "invvol": cut(book_daily(D, iv))}
    rec = {}
    for k, res in (("equal", eq), ("invvol", iv)):
        mm = res["monthly"]["net"].astype(float)
        dm = monthly(agg[k]).reindex(mm.index).dropna()
        rec[k] = {"months": len(dm), "corr_daily_vs_monthly_book": dm.corr(mm.reindex(dm.index)),
                  "mean_diff_%/mo": (dm - mm.reindex(dm.index)).mean() * 100,
                  "names": res["monthly"]["names"].mean(), "turnover_%/mo": res["monthly"]["turnover"].mean() * 100}
    T["aggressive_reconcile"] = pd.DataFrame(rec).T

    # ---------------- stable sleeve
    step("stable sleeve")
    score = S["resmom"]
    U_all, U_tr = D["listed"], D["U"][rs.TRADE]
    sleeves_st = {}
    for tag, U in (("all", U_all), ("tradable", U_tr)):
        w = winner_weights(score, U, c2.winners_pct, c2.winners_cap)
        win, turn = cohort_daily(D, w, c2.winners_k, buy + sell)
        win = cut(win)
        etf_cost = 2 * cfg.fee * cfg.fee_discount + 0.5 * (cfg.tax + c2.etf_tax)
        sleeves_st[tag] = two_assets(etf, win, c2.etf_weight, cost=etf_cost)
        series[f"winners_{tag}"] = win
        T.setdefault("stable_turnover", {})[tag] = turn.mean() * 100
    T["stable_turnover"] = pd.Series(T["stable_turnover"], name="winners_turnover_%/mo").to_frame()
    series["etf_0050"] = etf

    # ---------------- risk control variants
    step("risk control")
    W = c2.vol_windows
    sleeve_specs = {
        "stable": (sleeves_st["all"], c2.stable_dd_mult, c2.stable_cap, c2.stable_dd_limit, c2.stable_ret_gap, buy / 2 + sell / 2),
        "stable_tradable": (sleeves_st["tradable"], c2.stable_dd_mult, c2.stable_cap, c2.stable_dd_limit, c2.stable_ret_gap, buy / 2 + sell / 2),
        "aggressive": (agg["invvol"], c2.aggressive_dd_mult, c2.aggressive_cap, c2.aggressive_dd_limit, c2.aggressive_ret_gap, buy / 2 + sell / 2),
        "aggressive_equal": (agg["equal"], c2.aggressive_dd_mult, c2.aggressive_cap, c2.aggressive_dd_limit, c2.aggressive_ret_gap, buy / 2 + sell / 2),
    }
    variants = {"none": dict(mult=None, dd_limit=None), "vol": dict(dd=False), "vol+dd": dict(dd=True),
                "dd_only": dict(mult=None, dd=True)}
    for w_ in W:
        variants[f"vol_{w_}d"] = dict(dd=False, windows=(w_,))
    rows, expo = {}, {}
    for name, (sl, k, cap, D_, gap, cst) in sleeve_specs.items():
        for v, o in variants.items():
            mlt = o["mult"] if "mult" in o else k
            ov = overlay(sl, mkt, rf_d, mlt, cap, D_ if o.get("dd") else None, o.get("windows", W), cst)
            series[f"{name}|{v}"] = ov["ret"]
            expo[f"{name}|{v}"] = ov["exposure"]
            rows[name, v] = {**evaluate(ov["ret"], mkt, rf_d, gap, k), "avg_exposure": ov["exposure"].mean(),
                             "min_exposure": ov["exposure"].min()}
    rows["market", "-"] = evaluate(mkt, mkt, rf_d, 0.0, 1.0)
    rows["0050", "-"] = evaluate(etf, mkt, rf_d, 0.0, 1.0)
    T["sleeves"] = pd.DataFrame(rows).T.rename_axis(["sleeve", "risk_control"])

    # ---------------- the two sleeves together
    step("combined and Monte Carlo")
    rng = np.random.default_rng(c2.seed)
    st_main, ag_main = series["stable|vol"], series["aggressive|vol"]
    comb_rows, mc_rows = {}, {}
    nmon = len(monthly(st_main))
    block = max(1, round(nmon ** (1 / 3)))
    mk_m = monthly(mkt)
    for ws in np.round(np.arange(0, 1.01, 0.1), 1):
        cb = two_assets(st_main, ag_main, ws, months=tuple(range(1, 13)))
        series[f"combined|stable={ws}"] = cb
        comb_rows[ws] = evaluate(cb, mkt, rf_d, None, 1.1, ret_mult=1.3)
        mc_rows[ws] = mc_report(monthly(cb), mk_m, None, 1.1, 1.3, block, c2.mc_draws, rng)
    T["combined"] = pd.DataFrame(comb_rows).T.rename_axis("stable_weight")
    T["combined_monte_carlo"] = pd.DataFrame(mc_rows).T.rename_axis("stable_weight")
    mc = {}
    for name, (_, k, _, _, gap, _) in sleeve_specs.items():
        for v in ("none", "vol", "vol+dd"):
            for b in (block, 3, 12):
                mc[name, v, b] = mc_report(monthly(series[f"{name}|{v}"]), mk_m, gap, k, None, b, c2.mc_draws, rng)
    T["sleeve_monte_carlo"] = pd.DataFrame(mc).T.rename_axis(["sleeve", "risk_control", "block"])

    # ---------------- correlations
    step("correlations")
    ac = avg_corr(D, iv)
    T["correlation"] = pd.DataFrame({
        "aggressive_holdings_avg_corr_60d": ac,
        "stable_vs_aggressive_36m": monthly(sleeves_st["all"]).rolling(36).corr(monthly(agg["invvol"])),
        "aggressive_vs_market_36m": monthly(agg["invvol"]).rolling(36).corr(mk_m),
    })
    T["correlation_summary"] = pd.DataFrame({
        "stable_vs_aggressive": [monthly(sleeves_st["all"]).corr(monthly(agg["invvol"]))],
        "stable_vs_market": [monthly(sleeves_st["all"]).corr(mk_m)],
        "aggressive_vs_market": [monthly(agg["invvol"]).corr(mk_m)],
        "holdings_avg_corr_mean": [ac.mean()], "holdings_avg_corr_p90": [ac.quantile(0.9)]})

    # ---------------- control: random stocks instead of the winners
    step("control baskets")
    T["control"] = control(D, score, U_all, etf, mkt, rf_d, c2, buy + sell, rng, series)

    # ---------------- write
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in T.items():
            v.to_csv(out_dir / f"{k}.csv", encoding="utf-8-sig")
        pd.DataFrame(series).to_csv(out_dir / "daily_returns.csv")
        pd.DataFrame(expo).to_csv(out_dir / "daily_exposure.csv")
        (out_dir / "config.json").write_text(json.dumps({"step1": asdict(cfg), "step2": asdict(c2), "mc_block_months": block},
                                                        indent=2, default=str), encoding="utf-8")
    step(f"done: {len(T)} tables")
    return {"T": T, "series": series, "D": D}


def control(D, score, U, etf, mkt, rf_d, c2: Cfg2, cost_rt, rng, series) -> pd.DataFrame:
    """The stable sleeve with the winners replaced by random stocks: 'size_matched' draws,
    for every winner, a stock of the same market-cap quintile among the stocks with a
    score; 'random' draws from all of them. Same count and the same weights."""
    w_real = winner_weights(score, U, c2.winners_pct, c2.winners_cap)
    me = D["me"]
    cut = lambda d: d.loc[(d.index >= pd.Period(c2.start, "M").start_time)]
    real = series.get("winners_all")
    etf_cost = 0.0
    rows = []
    prep = {}                     # formation -> (pool, {quintile: pool of that quintile}, quintile of each winner)
    for f, ws in w_real.items():
        s = score.loc[f].where(U.loc[f]).dropna()
        pool = np.asarray(s.index, dtype=object)
        q = pd.qcut(me.loc[f, s.index].rank(method="first"), 5, labels=False)
        by_q = {k: np.asarray(q.index[q == k], dtype=object) for k in range(5)}
        prep[f] = (pool, by_q, q.reindex(ws.index).fillna(2).astype(int).to_numpy())
    for kind in ("size_matched", "random"):
        for draw in range(c2.control_draws):
            w = {}
            for f, ws in w_real.items():
                pool, by_q, wq = prep[f]
                if kind == "random":
                    pick = rng.choice(pool, size=len(ws), replace=False)
                else:
                    pick = np.empty(len(ws), dtype=object)
                    for k in range(5):
                        at = np.flatnonzero(wq == k)
                        if len(at):
                            pick[at] = rng.choice(by_q[k], size=len(at), replace=len(at) > len(by_q[k]))
                w[f] = pd.Series(ws.to_numpy(), index=pick).groupby(level=0).sum()
            basket, _ = cohort_daily(D, w, c2.winners_k, cost_rt)
            st = two_assets(etf, cut(basket), c2.etf_weight, cost=etf_cost)
            e = evaluate(st, mkt, rf_d, c2.stable_ret_gap, c2.stable_dd_mult)
            rows.append({"kind": kind, "draw": draw, "ret_%": e.get("ret_%_full"), "dd_%": e.get("dd_%_full"),
                         "basket_ann_%": ann(cut(basket)) * 100, "basket_dd_%": dd_daily(cut(basket)) * 100})
    out = pd.DataFrame(rows)
    if real is not None:
        st_real = two_assets(etf, real, c2.etf_weight)
        e = evaluate(st_real, mkt, rf_d, c2.stable_ret_gap, c2.stable_dd_mult)
        out = pd.concat([out, pd.DataFrame([{"kind": "actual_winners", "draw": -1, "ret_%": e.get("ret_%_full"),
                                             "dd_%": e.get("dd_%_full"), "basket_ann_%": ann(real) * 100,
                                             "basket_dd_%": dd_daily(real) * 100}])], ignore_index=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--open-holdout", action="store_true", help="only for the single final run")
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo and control draws")
    a = ap.parse_args()
    cfg = rs.Cfg(end=a.end, open_holdout=a.open_holdout)
    c2 = Cfg2(mc_draws=1000, control_draws=20) if a.quick else Cfg2()
    raw = rs.load_raw()
    raw["etf_0050_d"] = dt.etf_0050_daily()
    run(cfg, c2, OUT / f"to_{cfg.end}{'_quick' if a.quick else ''}", raw=raw)


if __name__ == "__main__":
    main()
