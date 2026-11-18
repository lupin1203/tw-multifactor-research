"""Residual momentum (Blitz, Huij & Martens 2011) for any country.

Core calculations behind Table 2 / 4 / 6 of Lin, "Residual momentum in international
stock markets": winner-minus-loser returns for K = 3/6/9/12 for residual momentum and
for ordinary total-return momentum, with Newey-West t statistics.

Timing (t = first holding month):
  regression window   t-36 .. t-1   excess return on RMRF, SMB, HML (36 months required)
  score               sum(residuals t-12 .. t-2) / std(residuals t-12 .. t-2)
  total momentum      compounded return t-12 .. t-2
  portfolios          sorted on the score, long top `pct` / short bottom `pct`
                      (10% = deciles, 30% = Table 4), equal or value weighted
  holding             K months, overlapping: month t averages the cohorts formed for
                      t, t-1, ..., t-K+1

Inputs (long format, returns in decimals: 0.05 = 5%):
  returns   columns  id, month, ret   [+ me_lag = market cap at the end of month-1]
  factors   columns  month, RMRF, SMB, HML   [+ RF]   (RF absent -> treated as 0)

See report.py for the full set of tables and charts, and its command line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 36                 # regression window, months
FIRST, LAST = 12, 2         # score uses months t-12 .. t-2
HOLD = (3, 6, 9, 12)
RET_CLIP = (-0.95, 1.00)    # monthly return limits
MIN_STOCKS = 30             # per country and month
PCT = 0.10                  # winner / loser fraction
FACTORS = ["RMRF", "SMB", "HML"]


# --------------------------------------------------------------------------- inputs
def clip_returns(ret, lo: float = RET_CLIP[0], hi: float = RET_CLIP[1]):
    """Returns above +100% become +100%, below -95% become -95%."""
    return ret.clip(lower=lo, upper=hi)


def to_wide(long: pd.DataFrame, value: str = "ret", id_col: str = "id",
            month_col: str = "month", clip: bool | None = None) -> pd.DataFrame:
    """Long table -> month x stock matrix on a gap-free monthly index.

    Returns are clipped (clip defaults to True when value == "ret").
    """
    d = long[[id_col, month_col, value]].copy()
    d[month_col] = pd.PeriodIndex(d[month_col].astype(str), freq="M")
    w = d.pivot(index=month_col, columns=id_col, values=value)
    w = w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="M"))
    if clip is None:
        clip = value == "ret"
    return clip_returns(w) if clip else w


def prep_factors(fac: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    f = fac.copy()
    if month_col in f.columns:
        f = f.set_index(month_col)
    f.index = pd.PeriodIndex(f.index.astype(str), freq="M")
    if "RF" not in f.columns:
        f["RF"] = 0.0
    return f[FACTORS + ["RF"]].dropna().sort_index()


# --------------------------------------------------------------------------- scores
def scores(ret: pd.DataFrame, fac: pd.DataFrame, window: int = WINDOW,
           first: int = FIRST, last: int = LAST,
           min_stocks: int = MIN_STOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual and total-return momentum scores.

    Row label = first holding month t; the row only uses data up to month t-1.
    Stocks without all `window` returns are skipped; months with fewer than
    `min_stocks` usable stocks produce no row.
    """
    res, tot = {}, {}
    lo, hi = window - first, window - last + 1          # rows for t-first .. t-last
    for i in range(window, len(ret.index) + 1):
        win = ret.index[i - window:i]                   # t-window .. t-1
        if not win.isin(fac.index).all():
            continue
        r = ret.loc[win]
        r = r.loc[:, r.notna().all()]
        if r.shape[1] < min_stocks:
            continue
        f = fac.loc[win]
        y = r.to_numpy() - f["RF"].to_numpy()[:, None]
        X = np.column_stack([np.ones(window), f[FACTORS].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = (y - X @ beta)[lo:hi]
        sd = e.std(0, ddof=1)
        ok = sd > 1e-12                                 # exact fit -> no score
        t = win[-1] + 1
        res[t] = pd.Series(e.sum(0)[ok] / sd[ok], index=r.columns[ok])
        tot[t] = pd.Series(np.expm1(np.log1p(r.to_numpy()[lo:hi]).sum(0)), index=r.columns)
    cols = ret.columns
    return (pd.DataFrame(res).T.reindex(columns=cols),
            pd.DataFrame(tot).T.reindex(columns=cols))


# ----------------------------------------------------------------------- portfolios
def decile(score: pd.Series, n_groups: int = 10) -> pd.Series:
    """Group 1 = lowest score ... n_groups = highest; equal-sized by rank."""
    s = score.dropna()
    rank = s.rank(method="first").to_numpy() - 1
    return pd.Series((rank * n_groups // len(s)).astype(int) + 1, index=s.index)


def winners_losers(score: pd.Series, pct: float = PCT,
                   min_stocks: int = MIN_STOCKS) -> tuple[pd.Index, pd.Index] | None:
    """Top and bottom `pct` of stocks by score; None if fewer than `min_stocks`."""
    s = score.dropna().sort_values(kind="stable")
    if len(s) < min_stocks:
        return None
    n = max(int(round(len(s) * pct)), 1)
    return s.index[-n:], s.index[:n]


def cap_weights(w: np.ndarray, cap: float | None) -> np.ndarray:
    """Normalise weights to sum 1 with no weight above `cap`; the excess is spread
    over the other names in proportion to their weights."""
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    if cap is None or cap * (w > 0).sum() < 1:          # cap impossible -> leave as is
        return w
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        fixed = w >= cap - 1e-12
        w[fixed] = cap
        free = ~fixed & (w > 0)
        w[free] *= (1 - cap * fixed.sum()) / w[free].sum()
    return w


def wml(score: pd.DataFrame, ret: pd.DataFrame, hold=HOLD, pct: float = PCT,
        min_stocks: int = MIN_STOCKS, weight: pd.DataFrame | None = None,
        score_weight: bool = False, cap: float | None = None) -> pd.DataFrame:
    """Monthly winner (W), loser (L), winner-minus-loser (WML) returns for each K,
    plus the one-way turnover of each leg (TO_W, TO_L).

    The cohort labelled t is first held in month t. Month m holds the cohorts
    m, m-1, ..., m-K+1 in equal parts; months without all K cohorts are left out.
    Inside a cohort stocks are equal weighted, or weighted by `weight` (a month x
    stock matrix known before month m, e.g. market cap at the end of m-1).
    With score_weight=True a stock's weight inside its cohort is proportional to how
    extreme its score was at formation (score for winners, -score for losers),
    limited to `cap` per stock.
    Turnover = half the sum of |target weight - drifted weight from last month|,
    so 1.0 means the whole leg was replaced.
    """
    cols = ret.columns
    R = ret.to_numpy(dtype=float)
    Wt = None if weight is None else weight.reindex(index=ret.index, columns=cols).to_numpy(dtype=float)
    pos = {m: i for i, m in enumerate(ret.index)}
    cohort = {}
    for t, s in score.iterrows():
        c = winners_losers(s, pct, min_stocks)
        if c is None:
            continue
        legs = []
        for members, sign in ((c[0], 1.0), (c[1], -1.0)):
            mk = cols.isin(members).astype(float)
            if score_weight:
                sw = np.where(mk > 0, np.nan_to_num(sign * s.reindex(cols).to_numpy(dtype=float)), 0.0)
                sw = cap_weights(sw, cap)
                mk = sw if sw.sum() > 0 else mk
            legs.append(mk)
        cohort[t] = tuple(legs)

    def leg(mi: int, masks: list[np.ndarray]) -> tuple[float, np.ndarray]:
        r = R[mi]
        base = np.ones(len(cols)) if Wt is None else np.nan_to_num(Wt[mi], nan=0.0).clip(min=0)
        rets, ws = [], []
        for mk in masks:
            w = np.where(~np.isnan(r), mk * base, 0.0)
            tot = w.sum()
            if tot > 0:
                w = w / tot
                rets.append(float(w @ np.nan_to_num(r)))
                ws.append(w)
        if not rets:
            return np.nan, np.zeros(len(cols))
        return float(np.mean(rets)), np.mean(ws, axis=0)

    out, prev = {}, {}
    for m in ret.index:
        mi, row = pos[m], {}
        for K in hold:
            cs = [cohort.get(m - k) for k in range(K)]
            if any(c is None for c in cs):
                prev.pop(K, None)
                continue
            (w, ww), (l, wl) = leg(mi, [c[0] for c in cs]), leg(mi, [c[1] for c in cs])
            row[f"W{K}"], row[f"L{K}"], row[f"WML{K}"] = w, l, w - l
            if K in prev:
                r0 = np.nan_to_num(R[mi - 1])
                for tag, new, old in (("W", ww, prev[K][0]), ("L", wl, prev[K][1])):
                    d = old * (1 + r0)
                    d = d / d.sum() if d.sum() > 0 else d
                    row[f"TO_{tag}{K}"] = 0.5 * np.abs(new - d).sum()
            prev[K] = (ww, wl)
        if row:
            out[m] = row
    return pd.DataFrame(out).T


def net_of_cost(port: pd.DataFrame, K: int, buy: float, sell: float) -> pd.DataFrame:
    """W, L (as a long position) and WML after trading costs.

    Replacing a fraction TO of a leg costs TO * (buy + sell): the old names are closed
    and the new ones opened. `buy` / `sell` are one-way costs as fractions (fee, tax,
    half spread). Stock-borrowing fees for the short leg are not included.
    The first month of each series has no turnover figure and is dropped.
    """
    rt = buy + sell
    d = port[[f"W{K}", f"L{K}", f"WML{K}", f"TO_W{K}", f"TO_L{K}"]].dropna()
    return pd.DataFrame({
        "W": d[f"W{K}"] - d[f"TO_W{K}"] * rt,
        "L": d[f"L{K}"] - d[f"TO_L{K}"] * rt,
        "WML": d[f"WML{K}"] - (d[f"TO_W{K}"] + d[f"TO_L{K}"]) * rt,
    })


# ----------------------------------------------------------------------- statistics
def nw_t(x, lags: int) -> float:
    """t statistic of the mean with Newey-West (Bartlett kernel) standard error."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    u = x - x.mean()
    v = (u @ u) / n
    for j in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def cut(d, start: str | None = None, end: str | None = None):
    if start:
        d = d[d.index >= pd.Period(start, "M")]
    if end:
        d = d[d.index <= pd.Period(end, "M")]
    return d


def summarize(res: pd.DataFrame, tot: pd.DataFrame, hold=HOLD,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Mean % per month and Newey-West t (lag = K) of WML, residual vs total."""
    res, tot = cut(res, start, end), cut(tot, start, end)
    empty = pd.Series(dtype=float)
    rows = []
    for K in hold:
        a, b = res.get(f"WML{K}", empty), tot.get(f"WML{K}", empty)
        rows.append({"K": K, "months": int(a.notna().sum()),
                     "residual_mean_%": a.mean() * 100, "residual_t": nw_t(a, K),
                     "total_mean_%": b.mean() * 100, "total_t": nw_t(b, K)})
    return pd.DataFrame(rows).set_index("K")


def max_drawdown(x: pd.Series) -> tuple[float, pd.Period | None, pd.Period | None]:
    """Largest peak-to-trough loss of the compounded series: (loss, peak, trough).

    The peak is the last month-end before the losses begin."""
    x = x.dropna()
    if x.empty:
        return np.nan, None, None
    cum = (1 + x).cumprod()
    cum = pd.concat([pd.Series([1.0], index=[x.index[0] - 1]), cum])
    dd = cum / cum.cummax() - 1
    trough = dd.idxmin()
    if dd[trough] == 0:
        return 0.0, None, None
    return float(dd[trough]), cum.loc[:trough].idxmax(), trough


def perf(x: pd.Series, rf: pd.Series | None = None) -> dict:
    """Annualised return (compounded), volatility, Sharpe ratio and max drawdown.

    Sharpe uses the series itself for long-short returns, or the excess over `rf`
    when rf is given (long-only portfolios)."""
    x = x.dropna()
    if len(x) < 2:
        return {}
    ex = x if rf is None else x - rf.reindex(x.index).fillna(0)
    dd, peak, trough = max_drawdown(x)
    return {"months": len(x),
            "ann_return_%": ((1 + x).prod() ** (12 / len(x)) - 1) * 100,
            "ann_vol_%": x.std() * np.sqrt(12) * 100,
            "sharpe": ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else np.nan,
            "max_dd_%": dd * 100, "dd_peak": str(peak) if peak else "", "dd_trough": str(trough) if trough else ""}


def table2(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Paper-style table: one row per country, mean %/month with (t) for each K."""
    rows = {}
    for name, s in summaries.items():
        row = {}
        for kind, label in [("residual", "Residual"), ("total", "Total")]:
            for K in s.index:
                row[(label, f"K={K}")] = f"{s.loc[K, kind + '_mean_%']:.3f} ({s.loc[K, kind + '_t']:.2f})"
        row[("", "months")] = int(s["months"].max())
        rows[name] = row
    t = pd.DataFrame(rows).T
    t.columns = pd.MultiIndex.from_tuples(t.columns)
    return t
