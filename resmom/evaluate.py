"""Tests applied to every factor. Rules are in FACTOR_SPEC.md.

Signals are labelled by formation month m (see signals.py); `ret` is the month x stock
matrix of monthly returns, so the first return a signal can earn is ret[m+1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import resmom as rm

HORIZONS = (1, 5, 10, 20, 40, 60)       # trading days, for the decay curve


# --------------------------------------------------------------------------- basics
def forward_return(ret: pd.DataFrame, months: int = 1, skip: int = 0) -> pd.DataFrame:
    """Row m = compounded return of months m+1+skip .. m+skip+months (all required)."""
    lr = np.log1p(ret.clip(lower=-0.999))
    out = sum(lr.shift(-(skip + k)) for k in range(1, months + 1))
    return np.expm1(out)


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties get the average rank (same as pandas / scipy, without their overhead)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]]              # first element of each run of equal values
    start = np.r_[np.flatnonzero(new), len(x)]        # where each run starts, plus the end
    run = new.cumsum()                                # run number of each sorted element, from 1
    r = np.empty(len(x))
    r[order] = 0.5 * (start[run - 1] + start[run] + 1)
    return r


def _corr(ra: np.ndarray, rb: np.ndarray) -> float:
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else np.nan


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(_rank(a), _rank(b)) if len(a) >= 2 else np.nan


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else np.nan


def _rows(frames: list[pd.DataFrame], mask: pd.DataFrame | None, cols: pd.Index | None = None):
    """Common months and columns of the frames as float arrays, plus the mask as a boolean
    array (None if there is no mask). Months missing from the mask are dropped."""
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    if cols is None:
        cols = frames[0].columns
        for f in frames[1:]:
            cols = cols.intersection(f.columns)
    if mask is not None:
        idx = idx.intersection(mask.index)
    arrays = [f.reindex(index=idx, columns=cols).to_numpy(dtype=float) for f in frames]
    M = None if mask is None else mask.reindex(index=idx, columns=cols).fillna(False).to_numpy(dtype=bool)
    return idx, arrays, M


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, universe: pd.DataFrame | None = None,
            min_n: int = 30) -> pd.Series:
    """Spearman correlation between the signal and the forward return, month by month.
    universe: boolean month x stock matrix; only True cells are used."""
    out = {}
    idx = signal.index.intersection(fwd.index)
    cols = signal.columns.intersection(fwd.columns)
    S, F = signal.loc[idx, cols].to_numpy(dtype=float), fwd.loc[idx, cols].to_numpy(dtype=float)
    U = None if universe is None else universe.reindex(index=idx, columns=cols).fillna(False).to_numpy(dtype=bool)
    for i, m in enumerate(idx):
        ok = np.isfinite(S[i]) & np.isfinite(F[i])
        if U is not None:
            ok &= U[i]
        out[m] = _spearman(S[i][ok], F[i][ok]) if ok.sum() >= min_n else np.nan
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series, lags: int = 3) -> dict:
    x = ic.dropna()
    return {"months": len(x), "mean_ic": x.mean(), "t": rm.nw_t(x, lags),
            "share_positive": float((x > 0).mean()) if len(x) else np.nan}


def halves(x: pd.Series) -> tuple[float, float]:
    """Mean of the first and of the second half of the sample."""
    x = x.dropna()
    k = len(x) // 2
    return (float(x.iloc[:k].mean()), float(x.iloc[k:].mean())) if k else (np.nan, np.nan)


def alpha_stats(y: pd.Series, mkt: pd.Series, rf: pd.Series | None = None, lags: int = 6) -> tuple[float, float, float]:
    """Regress y (minus rf) on the market's excess return: (alpha per month, its
    Newey-West t, beta). Pass rf=None when y is already a long-short return."""
    if rf is None:
        d = pd.concat([y, mkt], axis=1).dropna()
    else:
        d = pd.concat([y - rf.reindex(y.index).fillna(0), mkt - rf.reindex(mkt.index).fillna(0)], axis=1).dropna()
    if len(d) < 12:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy()])
    yv = d.iloc[:, 0].to_numpy()
    b = np.linalg.solve(X.T @ X, X.T @ yv)
    g = X * (yv - X @ b)[:, None]
    S = g.T @ g
    for j in range(1, lags + 1):
        G = g[j:].T @ g[:-j]
        S += (1 - j / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0])), float(b[1])


# ------------------------------------------------------------ long-short portfolios
def ls_returns(signal: pd.DataFrame, ret: pd.DataFrame, universe: pd.DataFrame | None = None,
               pct: float = rm.PCT, min_stocks: int = rm.MIN_STOCKS,
               ret_short: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top `pct` minus bottom `pct` by the signal, equal weight, rebalanced monthly.
    Columns W, L, WML and one-way turnover TO_W, TO_L, indexed by holding month.
    ret_short: returns to use for the short leg (delisting months are filled
    differently for long and short positions, each to its own disadvantage)."""
    s = signal if universe is None else signal.where(universe.reindex_like(signal).fillna(False).astype(bool))
    cols = ret.columns
    S, R = s.to_numpy(dtype=float), ret.to_numpy(dtype=float)
    RS = R if ret_short is None else ret_short.reindex(index=ret.index, columns=cols).to_numpy(dtype=float)
    R0, RS0 = np.nan_to_num(R), np.nan_to_num(RS)
    where = cols.get_indexer(s.columns)               # position of each signal column in ret (-1 = absent)
    row_of = {m: i for i, m in enumerate(s.index)}
    out, prev = {}, None
    for mi, m in enumerate(ret.index):                # m = holding month, formed at m-1
        fi = row_of.get(m - 1)
        c = None if fi is None else _extremes(S[fi], pct, min_stocks)
        if c is None:
            prev = None
            continue
        row, cur = {}, []
        for tag, members, X, X0 in (("W", c[0], R, R0), ("L", c[1], RS, RS0)):
            pos = where[members]
            pos = pos[pos >= 0]
            live = pos[~np.isnan(X[mi, pos])]         # equal weight among the members with a return this month
            w = np.zeros(len(cols))
            if len(live):
                w[live] = 1 / len(live)
            row[tag] = float(w @ X0[mi]) if len(live) else np.nan
            if prev is not None:                      # turnover against last month's weights after drift
                d = prev[len(cur)] * (1 + X0[mi - 1])
                d = d / d.sum() if d.sum() > 0 else d
                row["TO_" + tag] = 0.5 * np.abs(w - d).sum()
            cur.append(w)
        row["WML"] = row["W"] - row["L"]
        out[m], prev = row, cur
    p = pd.DataFrame(out).T
    return p.reindex(columns=["W", "L", "WML", "TO_W", "TO_L"]) if len(p) else p


def _extremes(score: np.ndarray, pct: float, min_stocks: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Column positions of the top and bottom `pct` by score (same rule as
    resmom.winners_losers: stable sort, ties keep column order); None if too few."""
    idx = np.flatnonzero(~np.isnan(score))
    if len(idx) < min_stocks:
        return None
    n = max(int(round(len(idx) * pct)), 1)
    order = idx[np.argsort(score[idx], kind="stable")]
    return order[-n:], order[:n]


def net_ls(port: pd.DataFrame, buy: float, sell: float, borrow: float = 0.0) -> pd.Series:
    """WML after trading costs on both legs and a yearly borrowing fee on the short leg.
    Months without a turnover figure (the first one) use the average turnover."""
    to = (port["TO_W"] + port["TO_L"])
    return port["WML"] - to.fillna(to.mean()) * (buy + sell) - borrow / 12


def breakeven_cost(port: pd.DataFrame) -> float:
    """Round-trip cost (buy + sell, as a fraction) at which the long-short return is zero."""
    to = (port["TO_W"] + port["TO_L"]).mean()
    return float(port["WML"].mean() / to) if to > 0 else np.nan


def factor_report(signal: pd.DataFrame, ret: pd.DataFrame, buy: float, sell: float,
                  universe: pd.DataFrame | None = None, mkt: pd.Series | None = None,
                  borrow: float = 0.03, start: str | None = None, end: str | None = None,
                  lags: int = 3, ret_short: pd.DataFrame | None = None, pct: float = rm.PCT,
                  port: pd.DataFrame | None = None, ic: pd.Series | None = None) -> dict:
    """Standard single-factor numbers: gross / net long-short return with Newey-West t,
    turnover, break-even cost, both halves, rank IC and (if mkt is given) the
    market-adjusted alpha. mkt = the market return in excess of the risk-free rate.
    start / end are HOLDING months. port (from ls_returns) and ic (from rank_ic, index
    already moved to the holding month) can be passed in to avoid computing them again."""
    if port is None:
        port = ls_returns(signal, ret, universe, pct=pct, ret_short=ret_short)
    port = rm.cut(port, start, end)
    if port.empty or "WML" not in port:
        return {}
    net = net_ls(port, buy, sell, borrow)
    if ic is None:
        ic = rank_ic(signal, forward_return(ret), universe)
        ic.index = ic.index + 1                       # formation month -> holding month
    ic = rm.cut(ic, start, end)
    h1, h2 = halves(net)
    out = {"months": int(port["WML"].notna().sum()),
           "gross_%/mo": port["WML"].mean() * 100, "gross_t": rm.nw_t(port["WML"], lags),
           "turnover_%/mo_per_leg": (port["TO_W"] + port["TO_L"]).mean() * 50,
           "net_%/mo": net.mean() * 100, "net_t": rm.nw_t(net, lags),
           "net_1st_half_%/mo": h1 * 100, "net_2nd_half_%/mo": h2 * 100,
           "breakeven_roundtrip_%": breakeven_cost(port) * 100,
           "pass": bool(rm.nw_t(net, lags) > 2 and h1 > 0 and h2 > 0),
           **{f"ic_{k}": v for k, v in ic_summary(ic, lags).items() if k != "months"}}
    if mkt is not None:
        a, at, b = alpha_stats(net, mkt.reindex(net.index))
        k = len(net.dropna()) // 2
        a1, _, _ = alpha_stats(net.dropna().iloc[:k], mkt)
        a2, _, _ = alpha_stats(net.dropna().iloc[k:], mkt)
        out.update({"beta_mkt": b, "alpha_%/mo": a * 100, "alpha_t": at,
                    "alpha_1st_half_%/mo": a1 * 100, "alpha_2nd_half_%/mo": a2 * 100,
                    "pass_alpha": bool(at > 2 and a1 > 0 and a2 > 0)})
    return out


# ------------------------------------------------- conditioning tables (no single cut)
def group_returns(indicator: pd.DataFrame, fwd: pd.DataFrame, subset: pd.DataFrame | None = None,
                  n_groups: int = 3, min_n: int = 15) -> pd.DataFrame:
    """Mean forward return of `n_groups` equal-sized groups sorted on the indicator
    (group 1 = lowest), month by month, optionally only among `subset` stocks."""
    out = {}
    idx, (X, F), U = _rows([indicator, fwd], subset, indicator.columns)
    for i, m in enumerate(idx):
        ok = np.isfinite(X[i]) & ~np.isnan(F[i])
        if U is not None:
            ok &= U[i]
        n = int(ok.sum())
        if n < min_n:
            continue
        x, f = X[i][ok], F[i][ok]
        rank = np.empty(n, dtype=int)
        rank[np.argsort(x, kind="stable")] = np.arange(n)          # ties: first come, lower rank (as resmom.decile)
        g = rank * n_groups // n
        cnt = np.bincount(g, minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = np.bincount(g, weights=f, minlength=n_groups) / cnt
    cols = list(range(1, n_groups + 1))
    return pd.DataFrame.from_dict(out, orient="index", columns=cols) if out else pd.DataFrame(columns=cols, dtype=float)


def condition_report(indicator: pd.DataFrame, fwd: pd.DataFrame, subset: pd.DataFrame | None = None,
                     n_groups: int = 3, min_n: int = 15, lags: int = 3) -> dict:
    """Does the indicator sort forward returns among `subset` stocks? The rank IC is the
    main evidence; the group means show how much there is to earn."""
    ic = rank_ic(indicator, fwd, subset, min_n=min_n)
    g = group_returns(indicator, fwd, subset, n_groups, min_n)
    means = g.mean()
    d = means.diff().dropna()
    return {**ic_summary(ic, lags),
            **{f"group{k}_%/mo": means[k] * 100 for k in g.columns},
            "high_minus_low_%/mo": (g[n_groups] - g[1]).mean() * 100,
            "high_minus_low_t": rm.nw_t(g[n_groups] - g[1], lags),
            "monotonic": bool(len(d) == n_groups - 1 and ((d > 0).all() or (d < 0).all()))}


def two_by_two(a: pd.DataFrame, b: pd.DataFrame, fwd: pd.DataFrame, universe: pd.DataFrame | None = None,
               min_n: int = 40, lags: int = 3) -> dict:
    """Independent median splits on a and b. Reports the mean forward return of the
    four cells, the spread 'a high minus a low' inside each half of b, the difference
    between those two spreads, and the rank IC of a inside each half of b."""
    cells, ic_hi, ic_lo = {}, {}, {}
    idx, (A, B, F), U = _rows([a, b, fwd], universe)
    for i, m in enumerate(idx):
        ok = ~(np.isnan(A[i]) | np.isnan(B[i]) | np.isnan(F[i]))
        if U is not None:
            ok &= U[i]
        if ok.sum() < min_n:
            continue
        x, y, f = A[i][ok], B[i][ok], F[i][ok]
        xh, yh = x > np.median(x), y > np.median(y)
        cells[m] = {"a_hi_b_hi": _mean(f[xh & yh]), "a_lo_b_hi": _mean(f[~xh & yh]),
                    "a_hi_b_lo": _mean(f[xh & ~yh]), "a_lo_b_lo": _mean(f[~xh & ~yh])}
        ic_hi[m] = _spearman(x[yh], f[yh])
        ic_lo[m] = _spearman(x[~yh], f[~yh])
    c = pd.DataFrame(cells).T
    if c.empty:
        return {}
    s_hi, s_lo = c["a_hi_b_hi"] - c["a_lo_b_hi"], c["a_hi_b_lo"] - c["a_lo_b_lo"]
    return {"months": len(c), **{k + "_%/mo": c[k].mean() * 100 for k in c.columns},
            "spread_in_b_hi_%/mo": s_hi.mean() * 100, "spread_in_b_hi_t": rm.nw_t(s_hi, lags),
            "spread_in_b_lo_%/mo": s_lo.mean() * 100, "spread_in_b_lo_t": rm.nw_t(s_lo, lags),
            "difference_%/mo": (s_lo - s_hi).mean() * 100, "difference_t": rm.nw_t(s_lo - s_hi, lags),
            "ic_in_b_hi": pd.Series(ic_hi).mean(), "ic_in_b_hi_t": rm.nw_t(pd.Series(ic_hi), lags),
            "ic_in_b_lo": pd.Series(ic_lo).mean(), "ic_in_b_lo_t": rm.nw_t(pd.Series(ic_lo), lags)}


def cells_by_sign(a: pd.DataFrame, b: pd.DataFrame, fwd: pd.DataFrame, subset: pd.DataFrame | None = None,
                  min_n: int = 15, min_cell: int = 3, lags: int = 3) -> dict:
    """Four cells cut at zero (not at the median): a < 0 or >= 0, b < 0 or >= 0.
    Used for reversal condition B (a = month-on-month revenue change, b = change in the
    year-on-year growth rate). Reports each cell's mean forward return and average
    number of stocks, and 'a<0,b>=0' minus 'a<0,b<0' (seasonal dip minus real decline)."""
    names = {(True, True): "a_neg_b_neg", (True, False): "a_neg_b_pos",
             (False, True): "a_pos_b_neg", (False, False): "a_pos_b_pos"}
    rows, counts = {}, {}
    idx, (A, B, F), U = _rows([a, b, fwd], subset)
    for i, m in enumerate(idx):
        ok = ~(np.isnan(A[i]) | np.isnan(B[i]) | np.isnan(F[i]))
        if U is not None:
            ok &= U[i]
        if ok.sum() < min_n:
            continue
        x, y, f = A[i][ok], B[i][ok], F[i][ok]
        r, c = {}, {}
        for (an, bn), name in names.items():
            sel = ((x < 0) == an) & ((y < 0) == bn)
            c[name] = int(sel.sum())
            r[name] = float(f[sel].mean()) if sel.sum() >= min_cell else np.nan
        rows[m], counts[m] = r, c
    R, C = pd.DataFrame(rows).T, pd.DataFrame(counts).T
    if R.empty:
        return {}
    d = R["a_neg_b_pos"] - R["a_neg_b_neg"]
    return {"months": len(R), **{k + "_%/mo": R[k].mean() * 100 for k in names.values()},
            **{k + "_n": C[k].mean() for k in names.values()},
            "seasonal_minus_real_%/mo": d.mean() * 100, "seasonal_minus_real_t": rm.nw_t(d, lags)}


# ---------------------------------------------------- decay in trading days, half-life
def decay_daily(signal: pd.DataFrame, dret: pd.DataFrame, horizons=HORIZONS, pct: float = rm.PCT,
                universe: pd.DataFrame | None = None, min_stocks: int = rm.MIN_STOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each formation month-end: (top `pct` minus bottom `pct`) cumulative return and
    the rank IC of the cumulative return, h trading days after the signal date.
    Returns (spread, ic), both month x horizon. Days without a price count as 0%."""
    L = np.log1p(dret.fillna(0).clip(lower=-0.999)).cumsum()
    dates = pd.Series(dret.index, index=dret.index.to_period("M")).groupby(level=0).max()
    pos = {d: i for i, d in enumerate(dret.index)}
    Lv, cols = L.to_numpy(), dret.columns
    sp, ic = {}, {}
    S = signal.reindex(columns=cols).to_numpy(dtype=float)
    if universe is not None:                          # months missing from the universe are left unmasked
        U = universe.reindex(index=signal.index, columns=cols).fillna(False).to_numpy(dtype=bool)
        S = np.where(U | ~signal.index.isin(universe.index)[:, None], S, np.nan)
    for k, m in enumerate(signal.index):
        if m not in dates.index:
            continue
        i = pos[dates[m]]
        sv = S[k]
        c = _extremes(sv, pct, min_stocks)
        if c is None:
            continue
        w, l = c
        ok = np.isfinite(sv)
        rank_s = _rank(sv[ok])
        rs, ri = {}, {}
        for h in horizons:
            if i + h >= len(Lv):
                continue
            r = np.expm1(Lv[i + h] - Lv[i])
            rs[h] = r[w].mean() - r[l].mean()
            ri[h] = _corr(rank_s, _rank(r[ok])) if ok.sum() >= 2 else np.nan
        sp[m], ic[m] = rs, ri
    return pd.DataFrame(sp).T.reindex(columns=list(horizons)), pd.DataFrame(ic).T.reindex(columns=list(horizons))


def half_life(mean_spread: pd.Series) -> float:
    """Trading days until the signal's daily payoff has halved.

    mean_spread: average cumulative long-short return by horizon (index = trading
    days, must include 5). The starting pace is the return of the first 5 days / 5.
    The half-life is the midpoint of the first later interval whose pace per day is at
    most half the starting pace; inf if that never happens within the horizons given;
    nan if the starting pace is not positive.
    """
    s = mean_spread.dropna().sort_index()
    if 5 not in s.index:
        return np.nan
    base = s[5] / 5
    if not base > 0:
        return np.nan
    hs = [h for h in s.index if h >= 5]
    for a, b in zip(hs[:-1], hs[1:]):
        if (s[b] - s[a]) / (b - a) <= base / 2:
            return (a + b) / 2
    return np.inf


# -------------------------------------------------------------- market-state checks
def trailing_pctile(x: pd.Series, window: int = 750, min_obs: int = 500) -> pd.Series:
    """Share of the previous `window` observations (today excluded) that are below today's value."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - window):i]
        w = w[np.isfinite(w)]
        if len(w) >= min_obs and np.isfinite(v[i]):
            out[i] = (w < v[i]).mean()
    return pd.Series(out, index=x.index)


def state_flag(indicator: pd.Series, tail: float = 0.10, low_is_bad: bool = True,
               window: int = 750) -> pd.Series:
    """True on the days the indicator is in the worst `tail` of its own previous `window`
    days (today excluded); missing until there is enough history."""
    p = trailing_pctile(indicator, window, min_obs=min(500, window * 2 // 3))
    bad = (p <= tail) if low_is_bad else (p >= 1 - tail)
    return bad.astype(float).where(p.notna())         # 1 / 0 / missing


def _forward(mkt_dret: pd.Series, horizon: int, delay: int) -> tuple[pd.Series, pd.Series]:
    """(return, worst loss) of the market over the `horizon` trading days that start at
    the close of day t + delay."""
    lr = np.log1p(mkt_dret).cumsum()
    base = lr.shift(-delay)
    fwd = np.expm1(lr.shift(-(horizon + delay)) - base)
    lo = lr.rolling(horizon, min_periods=horizon).min().shift(-(horizon + delay))   # lowest close inside the window
    return fwd, np.expm1(lo - base).clip(upper=0)


def flag_report(flag: pd.Series, mkt_dret: pd.Series, horizon: int = 20, delay: int = 1) -> dict:
    """Market over the next `horizon` trading days on flagged days versus the others.
    The difference's t uses Newey-West with `horizon` lags (the windows overlap)."""
    fwd, worst = _forward(mkt_dret, horizon, delay)
    d = pd.DataFrame({"fwd": fwd, "worst": worst, "bad": flag.reindex(mkt_dret.index)}).dropna()
    if d.empty:
        return {}
    f = d["bad"].astype(bool)
    if not f.any() or f.all():
        return {"days": len(d), "bad_days": int(f.sum()), "episodes": int(f.any())}
    # mean(fwd | bad) - mean(fwd | other) as the mean of one series, so that nw_t applies
    z = d["fwd"] * (f / f.mean() - (~f) / (~f).mean())
    return {"days": len(d), "bad_days": int(f.sum()),
            "episodes": int((f & ~f.shift(1, fill_value=False)).sum()),
            "fwd_%_bad": d.loc[f, "fwd"].mean() * 100, "fwd_%_other": d.loc[~f, "fwd"].mean() * 100,
            "diff_t": rm.nw_t(z, horizon),
            "worst_%_bad": d.loc[f, "worst"].mean() * 100, "worst_%_other": d.loc[~f, "worst"].mean() * 100}


def state_report(indicator: pd.Series, mkt_dret: pd.Series, horizon: int = 20, tail: float = 0.10,
                 low_is_bad: bool = True, window: int = 750, delay: int = 1) -> dict:
    """How the market did in the `horizon` trading days after the indicator was in its
    worst `tail` of the previous `window` days, versus the other days; plus the
    threshold-free rank correlation between the indicator and that forward return.
    delay: the indicator of day t is published after the close, so the forward window
    starts at the close of day t + delay."""
    x = indicator.reindex(mkt_dret.index)
    out = flag_report(state_flag(x, tail, low_is_bad, window), mkt_dret, horizon, delay)
    if not out:
        return {}
    fwd, _ = _forward(mkt_dret, horizon, delay)
    d = pd.DataFrame({"x": x, "fwd": fwd, "p": trailing_pctile(x, window, min_obs=min(500, window * 2 // 3))}).dropna()
    ic = _spearman(d["x"].to_numpy(), d["fwd"].to_numpy())
    n_eff = max(len(d) / horizon, 3)                                           # overlapping windows
    return {**out, "rank_corr": ic, "rank_corr_t_approx": ic * np.sqrt((n_eff - 2) / max(1 - ic ** 2, 1e-9))}
