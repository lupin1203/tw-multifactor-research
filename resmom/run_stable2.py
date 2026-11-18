"""Stable sleeve, round two (FACTOR_SPEC.md, 穩定那塊第二輪, 2026-11-18).

Four changes proposed by the user, every combination (16), on version B (quality
minimum-variance book + bonds):
  bonds     US Treasury 10-30 year ladder  |  1-3 year ladder
  trend     fixed 70% stocks  |  40% stocks while the TAIEX price index closed below its
            200-day average the day before
  rebalance monthly  |  quarterly with a buffer (holdings stay candidates while they pass
            the filters and rank in the top 70%)
  risk      covariance  |  downside covariance (min(r, 0), about zero)
Selection rule, goals, Monte Carlo: as written in the spec before the run.

    python run_stable2.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import data_tw as dt
import run_step1 as rs
import run_step2 as r2
import run_stable as st
import signals as sg

OUT = Path(__file__).parent / "output" / "stable2"


@dataclass(frozen=True)
class Cfg2S:
    stock_hi: float = 0.70
    stock_lo: float = 0.40
    ma_days: int = 200
    keep_top: float = 0.70
    quarter_months: tuple = (3, 6, 9, 12)
    short_ladder: tuple = (0, 3)
    long_ladder: tuple = (10, 30)
    pick_margin: float = 0.05


def book_weights2(D: dict, Q: dict, c: st.CfgS, months, downside: bool, rebalance, keep_top: float):
    """Formation month -> target weights for every month (in months without trading the
    last target is repeated; hold_daily then keeps the drifted holdings). With rebalance
    months, current holdings that still pass the filters and rank in the top keep_top by
    score stay candidates (buffer)."""
    dret = D["dret"]
    ind = D["groups_official"].fillna("?")
    rank = sg.pct_rank(Q["score"])
    out, diag, last = {}, {}, None
    for f in months:
        if f not in Q["cand"].index:
            continue
        if rebalance is not None and last is not None and f.month not in rebalance:
            out[f] = last
            continue
        members = Q["cand"].loc[f].fillna(False).astype(bool).copy()
        kept = 0
        if rebalance is not None and last is not None:
            r = rank.loc[f].reindex(last.index)
            keep = r.index[(r > 1 - keep_top).fillna(False).to_numpy()]
            kept = int((~members[keep]).sum())
            members[keep] = True
        names = members.index[members.to_numpy()]
        win = dret.loc[:f.end_time, names].tail(c.window)
        names = names[(win.notna().sum() >= c.min_days).to_numpy()]
        if len(names) < 2:
            continue
        X = win[names].fillna(0.0).to_numpy(dtype=float)
        cov, delta = st.lw_constant_corr(np.minimum(X, 0.0), demean=False) if downside else st.lw_constant_corr(X)
        w, good = st.min_var(cov, ind.reindex(names).to_numpy(), c.stock_cap, c.industry_cap)
        s = pd.Series(w, index=names)
        s = s[s > 0]
        out[f], last = s, s
        diag[f] = {"members": len(names), "kept_by_buffer": kept, "names": len(s), "shrinkage": delta, "solver_ok": good}
    return out, pd.DataFrame(diag).T


def trend_target(taiex: pd.Series, days: pd.DatetimeIndex, hi: float, lo: float, n: int) -> pd.Series:
    """Stock weight to hold on each day: lo when the index closed below its n-day average
    on the previous trading day, else hi."""
    px = taiex.dropna()
    below = (px < px.rolling(n, min_periods=n).mean()).astype(float).where(px.rolling(n).count() >= n)
    b = below.reindex(below.index.union(days)).ffill().reindex(days)
    tgt = pd.Series(np.where(b.shift(1) == 1, lo, hi), index=days)
    return tgt


def mix_dynamic(a: pd.Series, b: pd.Series, target: pd.Series, months=(1, 4, 7, 10), cost: float = 0.0) -> pd.Series:
    """Daily return of target[t] in a and the rest in b. The weight drifts; it is reset to
    the target at the start of a day when the target changed, and on the first trading day
    of the given months. |traded| x cost is charged on that day."""
    d = pd.concat([a.rename("a"), b.rename("b"), target.rename("t")], axis=1).dropna()
    A, B, Tg = d["a"].to_numpy(), d["b"].to_numpy(), d["t"].to_numpy()
    out, w, last_m, last_t = np.empty(len(d)), Tg[0], None, Tg[0]
    for i, t in enumerate(d.index):
        c = 0.0
        if Tg[i] != last_t or (last_m is not None and t.month != last_m and t.month in months):
            c = abs(w - Tg[i]) * cost
            w = Tg[i]
        last_m, last_t = t.month, Tg[i]
        r = w * A[i] + (1 - w) * B[i]
        out[i] = r - c
        w = w * (1 + A[i]) / (1 + r) if 1 + r != 0 else Tg[i]
    return pd.Series(out, index=d.index)


def run(cfg: rs.Cfg, c: st.CfgS, c2: Cfg2S, out_dir: Path | None = None, log=print, raw: dict | None = None,
        ust: pd.DataFrame | None = None, fx: pd.DataFrame | None = None, agg: pd.Series | None = None) -> dict:
    t0 = time.time()
    step = lambda msg: log(f"[{time.time() - t0:6.0f}s] {msg}")
    step(f"step 1 data (cut at {cfg.end})")
    raw = raw if raw is not None else rs.load_raw()
    D = rs.prepare(raw, cfg)
    S, _, _ = rs.build_signals(D)
    last = D["end"].end_time
    first = pd.Period(c.start, "M").start_time
    cut = lambda d: d.loc[(d.index >= first) & (d.index <= last)]
    mkt = cut(D["mkt_d"].dropna())
    rf_d = pd.Series(D["rf"].reindex(mkt.index.to_period("M")).to_numpy() / 21, index=mkt.index)
    buy, sell, _ = D["cost"]["tradable"]
    stock_ow = (buy + sell) / 2
    T, series = {}, {}

    step("bonds")
    ust, fx = ust.loc[ust.index <= last], fx.loc[fx.index <= last]
    fx_half = float(fx["half_spread"].loc[fx.index >= first].median())
    bonds = {}
    for tag, (lo, hi) in (("long", c2.long_ladder), ("short", c2.short_ladder)):
        lad = st.ladder_usd(ust, lo, hi, c.bond_start, c.bond_cost)
        bonds[tag] = cut(st.to_twd(lad["ret"], fx["mid"], D["dret"].index))
        series[f"bond|{tag}"] = bonds[tag]
    T["bonds"] = st.bond_table(bonds, mkt)

    step("books")
    Q = st.quality_candidates(D, S, c)
    months = [m for m in D["months"] if m >= pd.Period(c.start, "M") - 1]
    books, bdiag = {}, {}
    for reb, down in itertools.product(("monthly", "quarterly"), (False, True)):
        tag = f"{reb}|{'down' if down else 'cov'}"
        step(f"book {tag}")
        rmonths = c2.quarter_months if reb == "quarterly" else None
        W, dg = book_weights2(D, Q, c, months, down, rmonths, c2.keep_top)
        d, turn = st.hold_daily(D, W, buy + sell, rmonths)
        books[tag] = cut(d)
        series[f"book|{tag}"] = books[tag]
        bdiag[tag] = {**r2.evaluate(books[tag], mkt, rf_d, c.ret_gap, c.dd_mult),
                      "turnover_%/yr": turn.loc[pd.Period(c.start, "M"):].mean() * 12 * 100,
                      "names": dg["names"].mean(), "kept_by_buffer": dg["kept_by_buffer"].mean(),
                      "solver_fail": int((~dg["solver_ok"].astype(bool)).sum())}
    T["books"] = pd.DataFrame(bdiag).T

    step("16 combinations")
    taiex = D["market"]["taiex"]
    days = mkt.index
    tgt_on = trend_target(taiex, days, c2.stock_hi, c2.stock_lo, c2.ma_days)
    tgt_off = pd.Series(c2.stock_hi, index=days)
    T["trend_switch"] = pd.DataFrame({"share_of_days_below": [(tgt_on == c2.stock_lo).mean()],
                                      "switches": [int((tgt_on.diff().abs() > 0).sum())],
                                      "switches_per_year": [(tgt_on.diff().abs() > 0).sum() / (len(days) / r2.TD)]})
    rng = np.random.default_rng(c.seed)
    mk_m = r2.monthly(mkt)
    block = max(1, round(len(mk_m) ** (1 / 3)))
    rows = {}
    for bond, trend, reb, risk in itertools.product(("long", "short"), ("off", "on"), ("monthly", "quarterly"), ("cov", "down")):
        cost = stock_ow + c.bond_cost + fx_half
        s = mix_dynamic(books[f"{reb}|{risk}"], bonds[bond], tgt_on if trend == "on" else tgt_off, cost=cost)
        key = (bond, trend, reb, risk)
        series["|".join(key)] = s
        e = r2.evaluate(s, mkt, rf_d, c.ret_gap, c.dd_mult)
        mc = r2.mc_report(r2.monthly(s), mk_m, c.ret_gap, c.dd_mult, None, block, c.mc_draws, rng)
        changes = (bond == "short") + (trend == "on") + (reb == "quarterly") + (risk == "down")
        rows[key] = {**e, "p_goal_mc": mc["p_goal_both_halves"], "mc_maxdd_p5": mc["maxdd_%_p5"], "changes": changes}
    tab = pd.DataFrame(rows).T.rename_axis(["bonds", "trend", "rebalance", "risk"])
    T["combinations"] = tab
    passed = tab[tab["goal_met"].astype(bool)]
    if len(passed):
        best_p = passed["p_goal_mc"].max()
        near = passed[passed["p_goal_mc"] >= best_p - c2.pick_margin].sort_values(["changes", "p_goal_mc"], ascending=[True, False])
        pick = near.index[0]
    else:
        pick = None
    T["pick"] = pd.DataFrame({"picked": ["|".join(pick) if pick else "none passed"], "n_passed": [len(passed)]})

    yrs = {"market": mkt}
    for key in (("long", "off", "monthly", "cov"), ("short", "on", "quarterly", "down")) + ((pick,) if pick else ()):
        yrs["|".join(key)] = series["|".join(key)]
    for k, v in bonds.items():
        yrs[f"bond_{k}"] = v
    T["calendar_years_%"] = st.calendar_years(yrs)

    if agg is not None:
        step("combined with the aggressive sleeve")
        agg = cut(agg.loc[agg.index <= last].dropna())
        best = series["|".join(pick)] if pick else series["|".join(tab["p_goal_mc"].astype(float).idxmax())]
        cb_rows = {}
        for ws in np.round(np.arange(0, 1.01, 0.1), 1):
            cb = r2.two_assets(best, agg, ws, months=tuple(range(1, 13)))
            cb_rows[ws] = {**r2.evaluate(cb, mkt, rf_d, None, 1.1, ret_mult=1.3),
                           "p_goal_mc": r2.mc_report(r2.monthly(cb), mk_m, None, 1.1, 1.3, block, c.mc_draws, rng)["p_goal_both_halves"]}
        T["combined_with_aggressive"] = pd.DataFrame(cb_rows).T.rename_axis("stable_weight")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in T.items():
            v.to_csv(out_dir / f"{k}.csv", encoding="utf-8-sig")
        pd.DataFrame(series).to_csv(out_dir / "daily_returns.csv")
        (out_dir / "config.json").write_text(json.dumps({"step1": asdict(cfg), "stable": asdict(c), "round2": asdict(c2),
                                                         "mc_block_months": block}, indent=2, default=str), encoding="utf-8")
    step(f"done: {len(T)} tables")
    return {"T": T, "series": series, "D": D}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--open-holdout", action="store_true", help="only for the single final run")
    a = ap.parse_args()
    cfg = rs.Cfg(end=a.end, open_holdout=a.open_holdout)
    raw = rs.load_raw()
    agg_file = st.AGG_FILE.parent.parent / f"to_{cfg.end}" / "daily_returns.csv"   # step 2's run with the same end
    agg = pd.read_csv(agg_file, index_col=0, parse_dates=True)["aggressive|none"] if agg_file.exists() else None
    run(cfg, st.CfgS(), Cfg2S(), OUT / f"to_{cfg.end}", raw=raw, ust=dt.ust_yields(), fx=dt.usdtwd(), agg=agg)


if __name__ == "__main__":
    main()
