"""Stable sleeve, redesigned (FACTOR_SPEC.md, 穩定那塊重新設計, 2026-11-18).

Stocks  minimum-variance quality book: big and tradable stocks that earn money, pay a
        dividend and are not among the most indebted 20% of their industry; the top half by
        a within-industry score of dividend yield, ROE and value; weights that minimise the
        variance (Ledoit-Wolf constant-correlation shrinkage, 252 days), long only, 5% per
        stock, 20% per official industry. Monthly.
Bonds   US Treasuries held directly in a 10-30 year ladder, priced from the daily yields,
        in TWD at the Bank of Taiwan spot mid rate.
Versions A book alone; B book 70% + bonds 30% (main); C 0050 70% + bonds 30%.
Risk control, goals, Monte Carlo: as in run_step2.py.

Everything runs on step 1's prepare(), so nothing after cfg.end (2023-12) is visible.

    python run_stable.py
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import data_tw as dt
import resmom as rm
import run_step1 as rs
import run_step2 as r2
import signals as sg

OUT = Path(__file__).parent / "output" / "stable"
AGG_FILE = Path(__file__).parent / "output" / "step2" / "to_2023-12" / "daily_returns.csv"


@dataclass(frozen=True)
class CfgS:
    start: str = "2009-07"
    # stock book
    lev_worst: float = 0.20               # leave out the most indebted 20% of each industry
    top: float = 0.50                     # candidates: top half by the quality score
    window: int = 252                     # days of returns for the covariance
    min_days: int = 200
    stock_cap: float = 0.05
    industry_cap: float | None = 0.20
    rebalance: tuple | None = None        # months in which the book trades; None = every month
    # bonds
    bond_lo: int = 10                     # ladder: maturities lo+1 .. hi years after each roll
    bond_hi: int = 30
    bond_cost: float = 0.001              # of the traded value
    fx_cost_mult: float = 1.0             # x the Bank of Taiwan half spread
    bond_start: str = "2008-01"
    # mix
    stock_weight: float = 0.70            # = the stable goal's drawdown multiple
    # goals and risk control (as step 2's stable sleeve)
    ret_gap: float = -0.02
    dd_mult: float = 0.7
    cap: float = 1.0
    dd_limit: float = 0.20
    vol_windows: tuple = (20, 60, 120)
    mc_draws: int = 10000
    seed: int = 20261118


# ============================================================ bonds
def bond_price(y: float, T: float, c: float) -> float:
    """Price per unit face of a bond paying c/2 every half year, the last coupon and the face
    at T years, at yield y (both annual, decimal; compounding every half year). Dirty price:
    the next coupon is included, it drops out on the day it is paid."""
    if T <= 0:
        return 0.0
    t = T - 0.5 * np.arange(int(np.ceil(2 * T - 1e-9)))
    disc = (1 + y / 2) ** (-2 * t)
    return float(c / 2 * disc.sum() + disc[0])


def coupons_left(T: float) -> int:
    return int(np.ceil(2 * T - 1e-9)) if T > 0 else 0


def ladder_usd(Y: pd.DataFrame, lo: int, hi: int, start, cost: float) -> pd.DataFrame:
    """Daily USD return of a Treasury ladder (columns ret, traded).

    Y: daily yields in %, columns = maturity in years. On the first day and on the first
    trading day of every year: bonds with less than lo + 0.5 years left are sold, a new bond
    is bought at par (coupon = that day's yield) for every maturity lo+1 .. hi not held, and
    all are reset to equal value; |traded value| x cost is paid. Yields between the given
    maturities are interpolated linearly (flat below the shortest). Coupons are reinvested in
    the bond that paid them; a bond that matures before the roll is held as cash until then."""
    Y = Y.loc[Y.index >= pd.Timestamp(start)].dropna()
    mats = np.asarray(Y.columns, dtype=float)
    yl = lambda row, T: float(np.interp(T, mats, row)) / 100
    bonds = []                       # [maturity date, coupon, face]; face 0 & cash for matured
    cash, prev_val, prev_year = 0.0, None, None
    ret, traded = np.zeros(len(Y)), np.zeros(len(Y))
    for i, (t, row) in enumerate(zip(Y.index, Y.to_numpy(dtype=float))):
        T = lambda b: (b[0] - t).days / 365.25
        # value today of what was held yesterday, with coupons and redemptions paid today
        val = cash
        for b in bonds:
            Tn = T(b)
            if b[2] == 0:
                continue
            if Tn <= 0:                                      # redeemed: face + last coupon to cash
                cash += b[2] * (1 + b[1] / 2) if b[3] > 0 else 0.0
                val += b[2] * (1 + b[1] / 2) if b[3] > 0 else 0.0
                b[2] = 0.0
                continue
            paid = b[3] - coupons_left(Tn)                   # coupons paid since yesterday
            b[3] = coupons_left(Tn)
            p = bond_price(yl(row, Tn), Tn, b[1])
            if paid > 0:                                     # reinvest in the same bond
                b[2] += b[2] * paid * b[1] / 2 / p
            val += b[2] * p
        if prev_val is not None:
            ret[i] = val / prev_val - 1
        # roll
        if prev_year is None or t.year != prev_year:
            V = val if prev_val is not None else 1.0
            keep = [b for b in bonds if b[2] > 0 and T(b) >= lo + 0.5]
            sold = V - sum(b[2] * bond_price(yl(row, T(b)), T(b), b[1]) for b in keep)
            held = {round(T(b)) for b in keep}
            for m in range(lo + 1, hi + 1):
                if m not in held:
                    keep.append([t + pd.DateOffset(years=m), yl(row, m), 0.0, 2 * m])
            n = len(keep)
            cur = np.array([b[2] * bond_price(yl(row, T(b)), T(b), b[1]) if b[2] > 0 else 0.0 for b in keep])
            tr = np.abs(V / n - cur).sum() + (sold if prev_val is not None else 0.0)
            c = tr * cost if prev_val is not None else 0.0
            V2 = V - c
            for b, v in zip(keep, cur):
                p = bond_price(yl(row, T(b)), T(b), b[1])
                b[2] = V2 / n / p
            bonds, cash = keep, 0.0
            if prev_val is not None:
                ret[i] = (1 + ret[i]) * V2 / V - 1
                traded[i] = tr / V
            val = V2
        prev_val, prev_year = val, t.year
    return pd.DataFrame({"ret": ret, "traded": traded}, index=Y.index).iloc[1:]


def to_twd(r_usd: pd.Series, fx_mid: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Daily TWD return on the Taiwan trading days `days`: the USD value as of the last US
    close BEFORE the Taiwan day (US closes after Taiwan), times that day's spot mid rate."""
    lv = (1 + r_usd).cumprod()
    lv.index = lv.index + pd.Timedelta(days=1)
    grid = lv.index.union(fx_mid.index).union(days)
    v = lv.reindex(grid).ffill() * fx_mid.reindex(grid).ffill()
    v = v.reindex(days).dropna()
    return v.pct_change().dropna()


# ============================================================ stock book
def lw_constant_corr(X: np.ndarray, demean: bool = True) -> tuple[np.ndarray, float]:
    """Ledoit and Wolf (2004, 'Honey, I shrunk the sample covariance matrix'): the sample
    covariance pulled towards the matrix with every correlation equal to the average one.
    X: T x N returns. Returns (covariance, shrinkage weight on the target).
    demean=False: second moments about zero (for the downside version, min(r, 0))."""
    T, N = X.shape
    if demean:
        X = X - X.mean(axis=0)
    S = X.T @ X / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1)) if N > 1 else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    X2 = X ** 2
    pi_mat = X2.T @ X2 / T - S ** 2
    theta = (X ** 3).T @ X / T - var[:, None] * S
    ratio = np.sqrt(np.outer(1 / var, var))          # sqrt(s_jj / s_ii)
    off = ratio * theta
    np.fill_diagonal(off, 0.0)
    rho = np.trace(pi_mat) + rbar * off.sum()
    gamma = ((F - S) ** 2).sum()
    delta = float(np.clip((pi_mat.sum() - rho) / gamma / T, 0, 1)) if gamma > 0 else 1.0
    return delta * F + (1 - delta) * S, delta


def min_var(cov: np.ndarray, groups: np.ndarray, cap: float, ind_cap: float | None) -> tuple[np.ndarray, bool]:
    """Long-only weights minimising w' cov w, each <= cap, each group's sum <= ind_cap.
    Where the limits cannot add up to 1 the weights add up to what they allow (rest cash).
    Returns (weights, solver succeeded)."""
    n = len(cov)
    gs = {g: np.flatnonzero(groups == g) for g in pd.unique(groups)}
    room = sum(min(ind_cap, cap * len(ix)) if ind_cap is not None else cap * len(ix) for ix in gs.values())
    total = min(1.0, room)
    C = cov / np.mean(np.diag(cov))                  # scale for the solver
    cons = [{"type": "eq", "fun": lambda w: w.sum() - total, "jac": lambda w: np.ones(n)}]
    if ind_cap is not None:
        for ix in gs.values():
            if cap * len(ix) > ind_cap + 1e-12:
                a = np.zeros(n)
                a[ix] = 1.0
                cons.append({"type": "ineq", "fun": lambda w, a=a: ind_cap - a @ w, "jac": lambda w, a=a: -a})
    w0 = np.full(n, total / n).clip(max=cap)
    res = minimize(lambda w: w @ C @ w, w0, jac=lambda w: 2 * C @ w, method="SLSQP",
                   bounds=[(0, cap)] * n, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, cap)
    w[w < 1e-6] = 0.0
    if w.sum() > 0:
        w *= total / w.sum()
    return w, bool(res.success)


def quality_candidates(D: dict, S: dict, c: CfgS) -> dict[str, pd.DataFrame]:
    """Monthly matrices: pool, passed (the three filters), score, candidates."""
    U = D["U"]
    pool = U["big"] & U["tradable"]
    g = D["groups"]
    ep, dp, roe, bp = S["ep"], S["dp"], S["roe"], S["bp"]
    lev = -S["leverage_low"]
    lev_rank = sg.pct_rank(lev.where(pool), g)
    passed = pool & (ep > 0) & (dp > 0) & ~(lev_rank > 1 - c.lev_worst)
    r = lambda x: sg.pct_rank(x.where(passed), g)
    value = sg.combine([r(bp), r(ep)])
    score = sg.combine([r(dp), r(roe), value], missing=0.5).reindex_like(pool).where(passed)
    cand = (sg.pct_rank(score) > 1 - c.top) & passed
    return {"pool": pool, "passed": passed, "score": score, "cand": cand}


def book_weights(D: dict, members: pd.DataFrame, c: CfgS, method: str, months) -> tuple[dict, pd.DataFrame]:
    """Formation month -> weights over `members` (boolean matrix). method: minvar, equal,
    invvol. Also a table of diagnostics per month."""
    dret = D["dret"]
    ind = D["groups_official"].fillna("?")
    out, diag = {}, {}
    for f in months:
        if f not in members.index:
            continue
        names = members.columns[members.loc[f].fillna(False).to_numpy(dtype=bool)]
        if len(names) == 0:
            continue
        win = dret.loc[:f.end_time, names].tail(c.window)
        ok = win.notna().sum() >= c.min_days
        names = names[ok.to_numpy()]
        if len(names) < 2:
            continue
        X = win[names].fillna(0.0).to_numpy(dtype=float)
        info = {"members": int(members.loc[f].sum()), "with_history": len(names)}
        if method == "equal":
            w = np.full(len(names), 1 / len(names))
        elif method == "invvol":
            w = rm.cap_weights(1 / X.std(axis=0), c.stock_cap)
        else:
            cov, delta = lw_constant_corr(X)
            w, good = min_var(cov, ind.reindex(names).to_numpy(), c.stock_cap, c.industry_cap)
            info.update(shrinkage=delta, solver_ok=good, invested=w.sum())
        s = pd.Series(w, index=names)
        s = s[s > 0]
        out[f] = s
        iw = s.groupby(ind.reindex(s.index)).sum()
        info.update(names=len(s), max_stock=s.max(), max_industry=iw.max(), top_industry=iw.idxmax())
        diag[f] = info
    return out, pd.DataFrame(diag).T


def hold_daily(D: dict, weights: dict, cost_rt: float, rebalance=None) -> tuple[pd.Series, pd.Series]:
    """Daily return of holding the target weights from each formation month-end through the
    next month (drifting inside it); a part not invested is cash at 0. rebalance: months
    (1..12) in which the book trades; in the others the drifted holdings are kept.
    One-way turnover x cost_rt is charged on the first day. Returns (daily, turnover)."""
    dret, gone, ret = D["dret"], D["gone"], D["ret"]
    per = dret.index.to_period("M")
    out, turn, prev = [], {}, None
    for f in sorted(weights):
        days = dret.index[per == f + 1]
        if not len(days):
            continue
        trade = prev is None or rebalance is None or f.month in rebalance
        tgt = weights[f] if trade else prev
        # the first build is not charged (as run_step2.cohort_daily)
        tr = 0.5 * tgt.subtract(prev, fill_value=0).abs().sum() if (trade and prev is not None) else 0.0
        names = list(tgt.index)
        r = np.nan_to_num(dret.loc[days, names].to_numpy(dtype=float))
        g = gone.loc[f + 1, names].to_numpy(dtype=bool) if f + 1 in gone.index else np.zeros(len(names), bool)
        if g.any():
            r[:, g] = 0.0
            r[0, g] = np.nan_to_num(ret.loc[f + 1, names].to_numpy(dtype=float)[g])
        cash = 1.0 - tgt.sum()
        path = tgt.to_numpy() * np.cumprod(1 + r, axis=0)
        tot = path.sum(axis=1) + cash
        dr = np.concatenate([[tot[0] - 1], tot[1:] / tot[:-1] - 1])
        dr[0] -= tr * cost_rt
        out.append(pd.Series(dr, index=days))
        turn[f + 1] = tr
        prev = pd.Series(path[-1] / tot[-1], index=names)
        prev = prev[prev > 0]
    return (pd.concat(out).sort_index() if out else pd.Series(dtype=float)), pd.Series(turn, dtype=float)


# ============================================================ reports
def calendar_years(series: dict) -> pd.DataFrame:
    return pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
                         for k, v in series.items()})


def bond_table(b: dict, mkt: pd.Series) -> pd.DataFrame:
    rows = {}
    for k, s in b.items():
        x = pd.concat([s.rename("b"), mkt.rename("m")], axis=1).dropna()
        y22 = s.loc["2022"]
        rows[k] = {"ann_%": r2.ann(s) * 100, "vol_%": s.std() * np.sqrt(r2.TD) * 100,
                   "max_dd_%": r2.dd_daily(s) * 100, "ret_2022_%": ((1 + y22).prod() - 1) * 100 if len(y22) else np.nan,
                   "corr_with_market_daily": x["b"].corr(x["m"]),
                   "corr_with_market_monthly": r2.monthly(x["b"]).corr(r2.monthly(x["m"]))}
    return pd.DataFrame(rows).T


# ============================================================ run
def run(cfg: rs.Cfg, c: CfgS, out_dir: Path | None = None, log=print, raw: dict | None = None,
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
    etf = raw.get("etf_0050_d")
    etf = cut(etf.loc[etf.index <= last]) if etf is not None else mkt
    buy, sell, _ = D["cost"]["tradable"]
    stock_ow = (buy + sell) / 2
    T, series, books = {}, {}, {}

    # ---------------- bonds (nothing after the cut)
    step("bonds")
    ust = ust.loc[ust.index <= last]
    fx = fx.loc[fx.index <= last]
    fx_half = float(fx["half_spread"].loc[fx.index >= first].median())
    bonds, bonds_usd = {}, {}
    for lo, hi in ((c.bond_lo, c.bond_hi), (0, 10)):
        for mult in (1, 2):
            lad = ladder_usd(ust, lo, hi, c.bond_start, c.bond_cost * mult)
            tag = f"ust_{lo}-{hi}" + ("" if mult == 1 else "_cost2x")
            bonds_usd[tag] = cut(lad["ret"])
            bonds[tag] = cut(to_twd(lad["ret"], fx["mid"], D["dret"].index))
    main_bond = f"ust_{c.bond_lo}-{c.bond_hi}"
    T["bonds"] = pd.concat([bond_table(bonds, mkt).add_prefix("twd_"),
                            bond_table(bonds_usd, mkt)[["ann_%", "vol_%", "max_dd_%", "ret_2022_%"]].add_prefix("usd_")], axis=1)
    for k, v in bonds.items():
        series[f"bond|{k}"] = v

    # ---------------- stock books
    step("quality filter and candidates")
    Q = quality_candidates(D, S, c)
    months = [m for m in D["months"] if m >= pd.Period(c.start, "M") - 1]
    cnt = pd.DataFrame({k: Q[k].sum(axis=1) for k in ("pool", "passed", "cand")}).loc[months[0]:]
    T["filter_counts"] = cnt
    variants = {
        "minvar": (Q["cand"], "minvar", c),
        "equal": (Q["cand"], "equal", c),
        "invvol": (Q["cand"], "invvol", c),
        "minvar_nofilter": (Q["pool"], "minvar", c),
        "minvar_ind30": (Q["cand"], "minvar", replace(c, industry_cap=0.30)),
        "minvar_indnone": (Q["cand"], "minvar", replace(c, industry_cap=None)),
        "minvar_quarterly": (Q["cand"], "minvar", replace(c, rebalance=(3, 6, 9, 12))),
    }
    diag_rows, book_rows = {}, {}
    W_main = None
    for name, (members, method, cc) in variants.items():
        step(f"book {name}")
        if name == "minvar_quarterly" and W_main is not None:
            W, dg = W_main
        else:
            W, dg = book_weights(D, members, cc, method, months)
        if name == "minvar":
            W_main = (W, dg)
        d, turn = hold_daily(D, W, buy + sell, cc.rebalance)
        books[name] = cut(d)
        series[f"book|{name}"] = books[name]
        diag_rows[name] = dg
        book_rows[name] = {**r2.evaluate(books[name], mkt, rf_d, c.ret_gap, c.dd_mult),
                           "turnover_%/mo": turn.loc[pd.Period(c.start, "M"):].mean() * 100,
                           "names": dg["names"].mean() if len(dg) else np.nan,
                           "max_industry_%": dg["max_industry"].mean() * 100 if len(dg) else np.nan}
        if "solver_ok" in dg:
            book_rows[name].update(solver_fail_months=int((~dg["solver_ok"].astype(bool)).sum()),
                                   shrinkage=dg["shrinkage"].mean(), invested=dg["invested"].mean())
    book_rows["market"] = r2.evaluate(mkt, mkt, rf_d, 0.0, 1.0)
    book_rows["0050"] = r2.evaluate(etf, mkt, rf_d, 0.0, 1.0)
    T["books"] = pd.DataFrame(book_rows).T
    dm = diag_rows["minvar"]
    T["minvar_monthly"] = dm
    W = W_main[0]
    iw = pd.DataFrame({f: w.groupby(D["groups_official"].reindex(w.index).fillna("?")).sum() for f, w in W.items()}).T.fillna(0)
    T["minvar_industry_weights"] = pd.DataFrame({"avg_%": iw.mean() * 100, "max_%": iw.max() * 100,
                                                 "months_held": (iw > 0).sum()}).sort_values("avg_%", ascending=False)
    stk = pd.DataFrame({f: w for f, w in W.items()}).T.fillna(0)
    nm = dt_names(stk.columns)
    T["minvar_top_stocks"] = pd.DataFrame({"name": nm, "avg_%": stk.mean() * 100, "months_held": (stk > 0).sum()}
                                          ).sort_values("avg_%", ascending=False).head(40)

    # ---------------- versions A, B, C and the mixes
    step("versions and risk control")
    fx_cost = fx_half * c.fx_cost_mult
    etf_ow = cfg.fee * cfg.fee_discount + 0.0005             # 0050: fee + half its 0.1% tax
    mix_cost = lambda stock_cost, bond_cost: stock_cost + bond_cost + fx_cost
    versions = {
        "A_book": books["minvar"],
        "B_book70_bond30": r2.two_assets(books["minvar"], bonds[main_bond], c.stock_weight, cost=mix_cost(stock_ow, c.bond_cost)),
        "C_0050_70_bond30": r2.two_assets(etf, bonds[main_bond], c.stock_weight, cost=mix_cost(etf_ow, c.bond_cost)),
    }
    mixes = {
        "B_stock60": r2.two_assets(books["minvar"], bonds[main_bond], 0.60, cost=mix_cost(stock_ow, c.bond_cost)),
        "B_stock80": r2.two_assets(books["minvar"], bonds[main_bond], 0.80, cost=mix_cost(stock_ow, c.bond_cost)),
        "B_bond0-10": r2.two_assets(books["minvar"], bonds["ust_0-10"], c.stock_weight, cost=mix_cost(stock_ow, c.bond_cost)),
        "B_cost2x": r2.two_assets(books["minvar"], bonds[f"{main_bond}_cost2x"], c.stock_weight,
                                  cost=mix_cost(stock_ow, 2 * c.bond_cost) + fx_cost),
        "C_bond0-10": r2.two_assets(etf, bonds["ust_0-10"], c.stock_weight, cost=mix_cost(etf_ow, c.bond_cost)),
    }
    for k, v in {**versions, **mixes}.items():
        series[k] = v
    rc = {"none": dict(mult=None, dd=None), "vol": dict(mult=c.dd_mult, dd=None),
          "vol+dd": dict(mult=c.dd_mult, dd=c.dd_limit), "dd_only": dict(mult=None, dd=c.dd_limit)}
    rows, expo = {}, {}
    for name, sl in versions.items():
        for v, o in rc.items():
            ov = r2.overlay(sl, mkt, rf_d, o["mult"], c.cap, o["dd"], c.vol_windows, stock_ow)
            series[f"{name}|{v}"] = ov["ret"]
            expo[f"{name}|{v}"] = ov["exposure"]
            rows[name, v] = {**r2.evaluate(ov["ret"], mkt, rf_d, c.ret_gap, c.dd_mult),
                             "avg_exposure": ov["exposure"].mean()}
    for name, sl in mixes.items():
        rows[name, "none"] = {**r2.evaluate(sl, mkt, rf_d, c.ret_gap, c.dd_mult), "avg_exposure": 1.0}
    rows["market", "-"] = r2.evaluate(mkt, mkt, rf_d, 0.0, 1.0)
    rows["0050", "-"] = r2.evaluate(etf, mkt, rf_d, 0.0, 1.0)
    T["stable"] = pd.DataFrame(rows).T.rename_axis(["version", "risk_control"])
    T["calendar_years_%"] = calendar_years({"market": mkt, "0050": etf, "A_book": versions["A_book"],
                                           "B_book70_bond30": versions["B_book70_bond30"],
                                           "C_0050_70_bond30": versions["C_0050_70_bond30"],
                                           main_bond: bonds[main_bond], "ust_0-10": bonds["ust_0-10"]})
    dd_when = {}
    for k in ("market", "A_book", "B_book70_bond30", "C_0050_70_bond30"):
        s = mkt if k == "market" else versions[k]
        lv = (1 + s).cumprod()
        ddv = lv / lv.cummax() - 1
        trough = ddv.idxmin()
        dd_when[k] = {"max_dd_%": ddv.min() * 100, "peak": str(lv.loc[:trough].idxmax().date()), "trough": str(trough.date())}
    T["drawdown_dates"] = pd.DataFrame(dd_when).T

    # ---------------- Monte Carlo
    step("Monte Carlo")
    rng = np.random.default_rng(c.seed)
    mk_m = r2.monthly(mkt)
    block = max(1, round(len(r2.monthly(versions["A_book"])) ** (1 / 3)))
    mc = {}
    for name in versions:
        for v in ("none", "vol"):
            for b in (block, 3, 12):
                mc[name, v, b] = r2.mc_report(r2.monthly(series[f"{name}|{v}"]), mk_m, c.ret_gap, c.dd_mult, None,
                                              b, c.mc_draws, rng)
    T["monte_carlo"] = pd.DataFrame(mc).T.rename_axis(["version", "risk_control", "block"])

    # ---------------- with the aggressive sleeve (step 2's daily returns, aggressive|none)
    if agg is not None:
        step("combined with the aggressive sleeve")
        agg = cut(agg.loc[agg.index <= last].dropna())
        cb_rows, cb_mc = {}, {}
        st = versions["B_book70_bond30"]
        for ws in np.round(np.arange(0, 1.01, 0.1), 1):
            cb = r2.two_assets(st, agg, ws, months=tuple(range(1, 13)))
            cb_rows[ws] = r2.evaluate(cb, mkt, rf_d, None, 1.1, ret_mult=1.3)
            cb_mc[ws] = r2.mc_report(r2.monthly(cb), mk_m, None, 1.1, 1.3, block, c.mc_draws, rng)
        T["combined_with_aggressive"] = pd.DataFrame(cb_rows).T.rename_axis("stable_weight")
        T["combined_monte_carlo"] = pd.DataFrame(cb_mc).T.rename_axis("stable_weight")
        T["correlation"] = pd.DataFrame({
            "with_market": {k: r2.monthly(v).corr(mk_m) for k, v in versions.items()},
            "with_aggressive": {k: r2.monthly(v).corr(r2.monthly(agg)) for k, v in versions.items()}})

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in T.items():
            v.to_csv(out_dir / f"{k}.csv", encoding="utf-8-sig")
        pd.DataFrame(series).to_csv(out_dir / "daily_returns.csv")
        pd.DataFrame(expo).to_csv(out_dir / "daily_exposure.csv")
        (out_dir / "config.json").write_text(json.dumps({"step1": asdict(cfg), "stable": asdict(c), "mc_block_months": block,
                                                         "fx_half_spread_median": fx_half}, indent=2, default=str),
                                             encoding="utf-8")
    step(f"done: {len(T)} tables")
    return {"T": T, "series": series, "D": D, "Q": Q, "W": W_main[0]}


def dt_names(ids) -> pd.Series:
    """Stock id -> short name where the stock list has it (empty otherwise)."""
    try:
        info = pd.read_parquet(dt.RAW / "stock_info_raw.parquet")
        nm = info.drop_duplicates("stock_id").set_index("stock_id")["stock_name"]
        return pd.Series(ids, index=ids).map(nm).fillna("")
    except Exception:
        return pd.Series("", index=ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--open-holdout", action="store_true", help="only for the single final run")
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo draws")
    a = ap.parse_args()
    cfg = rs.Cfg(end=a.end, open_holdout=a.open_holdout)
    c = CfgS(mc_draws=1000) if a.quick else CfgS()
    raw = rs.load_raw()
    raw["etf_0050_d"] = dt.etf_0050_daily()
    agg = pd.read_csv(AGG_FILE, index_col=0, parse_dates=True)["aggressive|none"] if AGG_FILE.exists() else None
    run(cfg, c, OUT / f"to_{cfg.end}{'_quick' if a.quick else ''}", raw=raw, ust=dt.ust_yields(), fx=dt.usdtwd(), agg=agg)


if __name__ == "__main__":
    main()
