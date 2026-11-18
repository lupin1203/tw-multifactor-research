"""Walk-forward pieces: everything estimated from data uses only what was known at the
time. Rules are in FACTOR_SPEC.md (驗證方式, 砍因子規則, 成本與換股規則).

  cum_ic            expanding (or rolling) estimate of a signal's rank IC, summed over
                    the months it stays informative
  expected_return   Grinold: alpha = IC x volatility x score, in return units
  cut_decisions     each year-end, drop a factor that is both weak and unstable so far
  wf_composite      equal-weight composite of the factors kept for that year
  guard_holdout     data from 2024-01 on stays closed until the final run
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import evaluate as ev
import resmom as rm
import signals as sg

HOLDOUT_START = pd.Period("2024-01", "M")


def guard_holdout(end: str | pd.Period, open_holdout: bool = False) -> pd.Period:
    """Returns the last month allowed. Raises if `end` reaches into the holdout period
    while it is still closed."""
    end = pd.Period(end, "M")
    if end >= HOLDOUT_START and not open_holdout:
        raise ValueError(f"{end} is inside the holdout period (from {HOLDOUT_START}); "
                         "pass open_holdout=True only for the single final run")
    return end


def known_mean(x: pd.Series, lag: int, index: pd.PeriodIndex, min_months: int = 36,
               window: int | None = None) -> pd.Series:
    """Average of the observations already known at each month of `index`.

    x[s] becomes known at month s + lag. window=None uses everything known so far
    (expanding); otherwise only what became known in the last `window` months."""
    k = x.dropna().copy()
    k.index = k.index + lag
    k = k.reindex(index.union(k.index).sort_values())
    if window is None:
        m = k.expanding(min_periods=1).mean()
        n = k.notna().cumsum()
    else:
        m = k.rolling(window, min_periods=1).mean()
        n = k.notna().rolling(window, min_periods=1).sum()
    return m.where(n >= min_months).reindex(index)


def cum_ic(signal: pd.DataFrame, ret: pd.DataFrame, horizons: int = 6,
           universe: pd.DataFrame | None = None, min_months: int = 36,
           window: int | None = None) -> pd.DataFrame:
    """Column h (1..horizons) = estimate, at each formation month, of the rank IC between
    the signal and the return h months later; column 'cum' = their sum.
    The IC of signal month s with the return of month s+h is only used from month s+h."""
    out = {}
    for h in range(1, horizons + 1):
        ic = ev.rank_ic(signal, ret.shift(-h), universe)
        out[h] = known_mean(ic, h, signal.index, min_months, window)
    d = pd.DataFrame(out)
    d["cum"] = d[list(range(1, horizons + 1))].sum(axis=1, min_count=horizons)
    return d


def expected_return(score: pd.DataFrame, ic: pd.Series, vol: pd.DataFrame) -> pd.DataFrame:
    """alpha = IC x stock volatility x standardised score (Grinold), per formation month.
    ic: one number per month (e.g. cum_ic(...)['cum']); vol: monthly residual sd."""
    z = sg.normal_score(score)
    v = vol.reindex(index=z.index, columns=z.columns)
    return z.mul(v).mul(ic.reindex(z.index), axis=0)


def cut_decisions(net: pd.DataFrame, min_months: int = 60, t_cut: float = 1.0, lags: int = 3,
                  window: int | None = None) -> pd.DataFrame:
    """Year x factor table: True = the factor is used during that year.

    net: holding month x factor, long-short return after costs (for low volatility pass
    the market-adjusted return). At the end of year Y-1, with the months up to December
    Y-1 (the last `window` months if given), a factor is dropped for year Y when its
    Newey-West t is below `t_cut` AND one of the two halves is negative. With fewer
    than `min_months` months there is no decision and the factor stays in.
    """
    years = range(net.index.min().year + 1, net.index.max().year + 2)
    out = {}
    for y in years:
        d = net[net.index <= pd.Period(f"{y - 1}-12", "M")]
        if window is not None:
            d = d.iloc[-window:]
        row = {}
        for f in net.columns:
            x = d[f].dropna()
            if len(x) < min_months:
                row[f] = True
                continue
            h1, h2 = ev.halves(x)
            row[f] = not (rm.nw_t(x, lags) < t_cut and (h1 < 0 or h2 < 0))
        out[y] = row
    return pd.DataFrame(out).T.astype(bool)


def wf_composite(factors: dict[str, pd.DataFrame], keep: pd.DataFrame | None = None,
                 min_factors: int = 1, missing: float | None = None) -> pd.DataFrame:
    """Equal-weight composite. With `keep` (from cut_decisions) the portfolio formed at
    month m, held from m+1, uses the factors kept for the year of m+1. If every factor
    is dropped in some year, all of them are used that year."""
    if keep is None:
        return sg.composite(factors, min_factors=min_factors, missing=missing)
    idx = None
    for v in factors.values():
        idx = v.index if idx is None else idx.union(v.index)
    parts = []
    for y, g in pd.Series(idx, index=idx).groupby((idx + 1).year):
        names = [f for f in factors if (y not in keep.index) or bool(keep.loc[y].get(f, True))]
        names = names or list(factors)
        sub = {f: factors[f].reindex(g.index) for f in names}
        parts.append(sg.composite(sub, min_factors=min(min_factors, len(names)), missing=missing))
    return pd.concat(parts).sort_index()
