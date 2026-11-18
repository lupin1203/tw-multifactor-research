"""Factor values for the multi-factor tests. Rules are in FACTOR_SPEC.md.

Conventions
  monthly matrix     PeriodIndex (M) x stock id
  quarterly matrix   PeriodIndex (Q) x stock id, single-quarter figures
  daily matrix       DatetimeIndex x stock id
  formation month m  a monthly signal row labelled m uses only what is known at the
                     close of the last trading day of month m; it is paired with the
                     returns of month m+1 onward. (resmom.scores labels rows by the
                     first HOLDING month instead; use to_formation() to convert.)

Every function here is pure (matrices in, matrices out) so it can be unit-tested on
small simulated data. Nothing reads files; see the build script for the data layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

import resmom as rm

P = pd.Period


# ------------------------------------------------------------------ ranking helpers
def _rank01(x: pd.DataFrame) -> pd.DataFrame:
    n = x.notna().sum(axis=1)
    return (x.rank(axis=1, method="average") - 0.5).div(n.where(n > 0), axis=0)


def pct_rank(x: pd.DataFrame, groups: pd.Series | None = None, min_group: int = 5) -> pd.DataFrame:
    """Percentile rank within each row, in (0, 1): (rank - 0.5) / n, ties averaged.

    groups (stock id -> industry): rank inside the industry instead. Where an industry
    has fewer than `min_group` stocks with a value that month, or the stock has no
    industry, the market-wide rank is used.
    """
    mkt = _rank01(x)
    if groups is None:
        return mkt
    g = groups.reindex(x.columns)
    out = mkt.copy()
    for _, cols in g.groupby(g).groups.items():
        sub = x[cols]
        ok = sub.notna().sum(axis=1) >= min_group
        out[cols] = _rank01(sub).where(ok, mkt[cols], axis=0)
    return out


def combine(parts: list[pd.DataFrame], min_parts: int = 1, missing: float | None = None) -> pd.DataFrame:
    """Mean of the parts for each stock-month (parts are percentile ranks); a stock-month
    needs at least `min_parts` of them. missing=None: the mean of the parts that exist.
    missing=0.5: a part that does not exist counts as that rank and the sum is divided by
    the number of parts, so a stock known on few parts is pulled towards the middle."""
    idx = parts[0].index
    cols = parts[0].columns
    for p in parts[1:]:
        idx, cols = idx.union(p.index), cols.union(p.columns)
    a = np.stack([p.reindex(index=idx, columns=cols).to_numpy(dtype=float) for p in parts])
    n = (~np.isnan(a)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(a, 0) / n if missing is None else (np.nansum(a, 0) + missing * (len(parts) - n)) / len(parts)
        m = np.where(n >= max(min_parts, 1), mean, np.nan)
    return pd.DataFrame(m, index=idx, columns=cols)


def normal_score(x: pd.DataFrame) -> pd.DataFrame:
    """Row-wise standard-normal scores of the percentile rank (mean 0, sd about 1)."""
    r = pct_rank(x)
    return pd.DataFrame(ndtri(r.to_numpy(dtype=float)), index=r.index, columns=r.columns)


def composite(factors: dict[str, pd.DataFrame], names: list[str] | None = None,
              min_factors: int = 1, missing: float | None = None) -> pd.DataFrame:
    """Equal-weight composite: each factor is re-ranked market-wide, then averaged
    (see combine for `min_factors` and `missing`)."""
    names = list(factors) if names is None else names
    return combine([pct_rank(factors[k]) for k in names], min_parts=min_factors, missing=missing)


def orthogonal(x: pd.DataFrame, base: pd.DataFrame, universe: pd.DataFrame | None = None, min_n: int = 20) -> pd.DataFrame:
    """The part of x that `base` does not explain, month by month: both are turned into
    normal scores over the stocks that have both (inside `universe`), x is regressed on
    base across those stocks, and the residual is returned. Same-month data only."""
    b = base.reindex(index=x.index, columns=x.columns)
    both = x.notna() & b.notna()
    if universe is not None:
        both &= universe.reindex(index=x.index, columns=x.columns).fillna(False).astype(bool)
    zx, zb = normal_score(x.where(both)).to_numpy(), normal_score(b.where(both)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.nansum(zx * zb, 1) / np.nansum(zb * zb, 1)
    res = zx - beta[:, None] * zb
    res[both.to_numpy().sum(1) < min_n] = np.nan
    return pd.DataFrame(res, index=x.index, columns=x.columns)


def to_formation(score: pd.DataFrame) -> pd.DataFrame:
    """Rows labelled by first holding month -> rows labelled by formation month."""
    out = score.copy()
    out.index = out.index - 1
    return out


def mask(x: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """Set the listed stocks (e.g. financials) to missing."""
    if drop_cols is None or len(drop_cols) == 0:
        return x
    out = x.copy()
    out.loc[:, out.columns.isin(drop_cols)] = np.nan
    return out


def within(x: pd.DataFrame, universe: pd.DataFrame | None) -> pd.DataFrame:
    """Keep only the cells where `universe` (same labels, boolean) is True, so that
    ranks are taken among the stocks that can actually be held (e.g. listed ones)."""
    if universe is None:
        return x
    u = universe.reindex(index=x.index, columns=x.columns)
    return x.where(u.fillna(False).astype(bool))


# -------------------------------------------------------------------- daily helpers
def month_ends(idx: pd.DatetimeIndex) -> pd.Series:
    """Last trading day of each month: PeriodIndex (M) -> Timestamp."""
    return pd.Series(idx, index=idx.to_period("M")).groupby(level=0).max()


def at_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    me = month_ends(daily.index)
    out = daily.loc[me.to_numpy()]
    out.index = me.index
    return out


def daily_vol(dret: pd.DataFrame, months: int = 12, min_days: int = 120) -> pd.DataFrame:
    """Standard deviation of daily returns over the `months` calendar months ending at
    each month-end. Low volatility = good, so the factor score is the NEGATIVE of this."""
    out = {}
    for m, d in month_ends(dret.index).items():
        w = dret.loc[(m - months + 1).to_timestamp(how="start"):d]
        out[m] = w.std(ddof=1).where(w.notna().sum() >= min_days)
    return pd.DataFrame(out).T.reindex(columns=dret.columns)


def month_return_skip_last(dret: pd.DataFrame, skip: int = 1, min_days: int = 10) -> pd.DataFrame:
    """Compounded return inside each month, leaving out its last `skip` trading days
    (short-term reversal ranks on this and trades at the month-end close)."""
    per = dret.index.to_period("M")
    from_end = pd.Series(0, index=dret.index).groupby(per).cumcount(ascending=False)
    keep = dret.loc[(from_end >= skip).to_numpy()]
    kp = keep.index.to_period("M")
    lr = np.log1p(keep.clip(lower=-0.999))
    out = np.expm1(lr.groupby(kp).sum(min_count=1)).where(keep.notna().groupby(kp).sum() >= min_days)
    return out


def high_52w(dret: pd.DataFrame, months: int = 12, min_days: int = 120) -> pd.DataFrame:
    """Month-end adjusted price / highest adjusted close of the last `months` months."""
    px = (1 + dret.fillna(0)).cumprod().where(dret.notna())
    out = {}
    for m, d in month_ends(dret.index).items():
        w = px.loc[(m - months + 1).to_timestamp(how="start"):d]
        out[m] = (px.loc[d] / w.max()).where(w.notna().sum() >= min_days)
    return pd.DataFrame(out).T.reindex(columns=dret.columns)


def turnover_stats(volume: pd.DataFrame, shares: pd.DataFrame | None = None,
                   months: int = 12, min_months: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(abnormal, long_run) from daily share volume.

    abnormal   average daily turnover in month m / average daily turnover of the
               previous `months` months (reversal condition C; 2.0 = twice as busy)
    long_run   average daily turnover over the `months` months ending in m
    shares     monthly shares outstanding; if given, volume is divided by it so that
               new share issues do not look like a volume surge.
    """
    per = volume.index.to_period("M")
    v = volume
    if shares is not None:
        s = shares.reindex(index=per.unique(), columns=volume.columns)
        v = volume / s.reindex(per).to_numpy()
    S = v.groupby(per).sum(min_count=1)
    D = v.notna().groupby(per).sum()
    D = D.where(D > 0)
    prior = S.shift(1).rolling(months, min_periods=min_months).sum() / D.shift(1).rolling(months, min_periods=min_months).sum()
    abnormal = ((S / D) / prior).where(prior > 0)
    long_run = S.rolling(months, min_periods=min_months).sum() / D.rolling(months, min_periods=min_months).sum()
    return abnormal, long_run


def rolling_at_month_end(daily: pd.DataFrame, days: int = 20, min_days: int = 15) -> pd.DataFrame:
    """Sum of the last `days` trading days, read at each month-end."""
    return at_month_end(daily.rolling(days, min_periods=min_days).sum())


def change_at_month_end(level: pd.DataFrame, days: int = 20) -> pd.DataFrame:
    """Level minus its value `days` trading days earlier, read at each month-end."""
    return at_month_end(level - level.shift(days))


# ------------------------------------------------------------------- FF3 residuals
def ff3_fit(ret: pd.DataFrame, fac: pd.DataFrame, window: int = rm.WINDOW,
            min_stocks: int = rm.MIN_STOCKS):
    """Rolling 36-month three-factor regressions.

    Yields (last window month, stock ids, slopes [3 x n], residual sd [n]) for every
    month with a full window. Stocks without all `window` returns are skipped.
    """
    for i in range(window, len(ret.index) + 1):
        win = ret.index[i - window:i]
        if not win.isin(fac.index).all():
            continue
        r = ret.loc[win]
        r = r.loc[:, r.notna().all()]
        if r.shape[1] < min_stocks:
            continue
        f = fac.loc[win]
        y = r.to_numpy() - f["RF"].to_numpy()[:, None]
        X = np.column_stack([np.ones(window), f[rm.FACTORS].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        yield win[-1], r.columns, beta[1:], e.std(0, ddof=X.shape[1])


def residual_vol(ret: pd.DataFrame, fac: pd.DataFrame, window: int = rm.WINDOW) -> pd.DataFrame:
    """Monthly residual standard deviation from the 36-month regression ending in the
    formation month (used as each stock's volatility when turning scores into expected
    returns, and as the comparison for the low-volatility factor)."""
    out = {m: pd.Series(sd, index=cols) for m, cols, _, sd in ff3_fit(ret, fac, window)}
    return pd.DataFrame(out).T.reindex(columns=ret.columns)


def residual_of(target: pd.DataFrame, ret: pd.DataFrame, fac: pd.DataFrame,
                window: int = rm.WINDOW) -> pd.DataFrame:
    """Month-m value of `target` (a monthly return) minus what the three factors explain:
    target - RF - slopes . factors, with slopes from months m-window .. m-1 only."""
    out = {}
    for last, cols, b, _ in ff3_fit(ret, fac, window):
        m = last + 1
        if m not in fac.index or m not in target.index:
            continue
        f = fac.loc[m]
        out[m] = target.loc[m].reindex(cols) - f["RF"] - pd.Series(f[rm.FACTORS].to_numpy() @ b, index=cols)
    return pd.DataFrame(out).T.reindex(columns=ret.columns)


def industry_relative(x: pd.DataFrame, groups: pd.Series, min_group: int = 5) -> pd.DataFrame:
    """Value minus the same-month mean of the stock's industry (market mean when the
    industry has fewer than `min_group` values or is unknown)."""
    g = groups.reindex(x.columns)
    out = x.sub(x.mean(axis=1), axis=0)
    for _, cols in g.groupby(g).groups.items():
        sub = x[cols]
        ok = sub.notna().sum(axis=1) >= min_group
        out[cols] = sub.sub(sub.mean(axis=1), axis=0).where(ok, out[cols], axis=0)
    return out


# ------------------------------------------------------------------ monthly revenue
def _same_basis(index: pd.PeriodIndex, span: int, basis_break) -> pd.Series:
    """True where a comparison that reaches back `span` periods does not straddle
    `basis_break` (the first period reported on a new basis)."""
    if basis_break is None:
        return pd.Series(True, index=index)
    b = pd.Period(basis_break, index.freq)
    return pd.Series(np.asarray((index < b) | (index - span >= b)), index=index)


def revenue_signals(rev: pd.DataFrame, basis_break=None) -> dict[str, pd.DataFrame]:
    """rev: revenue-month x stock. Output rows are FORMATION months: revenue of month r
    is published by the 10th of r+1, so it is first usable at the end of r+1.

    basis_break: first month reported on a different basis (Taiwan: 2013-01, when
    monthly revenue changed from parent-only to consolidated). Growth rates that
    compare across it are meaningless and are left out.

    yoy3    last 3 months / same 3 months a year earlier - 1     (needs 15 months)
    accel   yoy3 - (last 12 months / previous 12 months - 1)     (needs 24 months)
    mom     latest month / previous month - 1
    d_yoy   latest month's year-on-year growth minus the previous month's
    A base that is zero or negative gives no value.
    """
    rev = rev.reindex(pd.period_range(rev.index.min(), rev.index.max(), freq="M"))
    r3, r12 = rev.rolling(3, min_periods=3).sum(), rev.rolling(12, min_periods=12).sum()
    b3, b12, b1, prev = r3.shift(12), r12.shift(12), rev.shift(12), rev.shift(1)
    yoy3 = (r3 / b3 - 1).where(b3 > 0)
    yoy12 = (r12 / b12 - 1).where(b12 > 0)
    yoy1 = (rev / b1 - 1).where(b1 > 0)
    i = rev.index
    yoy3 = yoy3.where(_same_basis(i, 14, basis_break), axis=0)
    yoy12 = yoy12.where(_same_basis(i, 23, basis_break), axis=0)
    yoy1 = yoy1.where(_same_basis(i, 12, basis_break), axis=0)
    mom = (rev / prev - 1).where(prev > 0).where(_same_basis(i, 1, basis_break), axis=0)
    out = {"yoy3": yoy3, "accel": yoy3 - yoy12, "mom": mom, "d_yoy": yoy1 - yoy1.shift(1)}
    for k, v in out.items():
        v = v.replace([np.inf, -np.inf], np.nan)
        v.index = v.index + 1
        out[k] = v
    return out


# ------------------------------------------------------------- quarterly statements
def usable_month(q: pd.Period, late: bool = False) -> pd.Period:
    """First formation month (month-end) at which the statements of quarter q may be used.

    Securities and Exchange Act art. 36. Until fiscal 2012: first and third quarter
    within one month (4/30, 10/31), half-year within two months (8/31), annual within
    four, later three, months. From fiscal 2013: each of the first three quarters
    within 45 days (5/15, 8/14, 11/14), annual within three months (3/31). A deadline
    on the last day of a month only counts from the following month-end.
    late=True: financial companies and foreign primary listings get longer (second
    quarter 8/31, financial holding companies' third quarter about 11/30), so their
    first three quarters are used one month later.
    """
    y, k = q.year, q.quarter
    if late:
        return {1: P(f"{y}-06", "M"), 2: P(f"{y}-09", "M"), 3: P(f"{y}-12", "M"), 4: P(f"{y + 1}-05", "M")}[k]
    if y >= 2013:
        return {1: P(f"{y}-05", "M"), 2: P(f"{y}-08", "M"), 3: P(f"{y}-11", "M"), 4: P(f"{y + 1}-04", "M")}[k]
    return {1: P(f"{y}-05", "M"), 2: P(f"{y}-09", "M"), 3: P(f"{y}-11", "M"), 4: P(f"{y + 1}-05", "M")}[k]


def full_quarters(q: pd.DataFrame) -> pd.DataFrame:
    return q.reindex(pd.period_range(q.index.min(), q.index.max(), freq="Q"))


def q_to_m(q: pd.DataFrame, months: pd.PeriodIndex, max_age: int = 8, late=None) -> pd.DataFrame:
    """Point-in-time monthly view of a quarterly matrix: at formation month m each stock
    shows its latest quarter that was usable by m, dropped once it is `max_age` months
    past the month it became usable. late: stock ids on the later filing schedule."""
    def view(x: pd.DataFrame, is_late: bool) -> pd.DataFrame:
        d = x.copy()
        d.index = pd.PeriodIndex([usable_month(p, is_late) for p in x.index], freq="M")
        d = d.groupby(level=0).last()                 # two quarters usable together: the newer one
        full = pd.period_range(min(d.index.min(), months.min()), max(d.index.max(), months.max()), freq="M")
        return d.reindex(full).ffill(limit=max_age).reindex(months)

    if late is None or len(late) == 0:
        return view(q, False)
    is_late = q.columns.isin(list(late))
    out = pd.concat([view(q.loc[:, ~is_late], False), view(q.loc[:, is_late], True)], axis=1)
    return out.reindex(columns=q.columns)


def ttm(q: pd.DataFrame) -> pd.DataFrame:
    """Sum of the last four quarters (all four required)."""
    return full_quarters(q).rolling(4, min_periods=4).sum()


def sue(ni: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Earnings surprise: (net income - same quarter last year) / sd of that change
    over the last `window` quarters including the current one (needs window + 4)."""
    d = full_quarters(ni)
    d = d - d.shift(4)
    sd = d.rolling(window, min_periods=window).std(ddof=1)
    return (d / sd).where(sd > 0)


def margin_change(rev: pd.DataFrame, profits: list[pd.DataFrame], basis_break=None) -> pd.DataFrame:
    """Average year-on-year change of the profit margins (gross, operating, net) of the
    latest quarter. Quarters with revenue <= 0 give no margin; quarters compared across
    `basis_break` (Taiwan: 2013Q1, parent-only -> consolidated statements) are left out."""
    rev = full_quarters(rev)
    ch = []
    for p in profits:
        mg = (p.reindex(index=rev.index, columns=rev.columns) / rev).where(rev > 0)
        ch.append(mg - mg.shift(4))
    return combine(ch).replace([np.inf, -np.inf], np.nan).where(_same_basis(rev.index, 4, basis_break), axis=0)


# ---------------------------------------------------------------- factor assemblies
def score_revenue(rev: pd.DataFrame, financials=None, basis_break=None, universe=None) -> dict[str, pd.DataFrame]:
    """Monthly revenue momentum = mean market-wide rank of yoy3 and accel; no financials.
    universe (formation month x stock, boolean): rank among these stocks only; the same
    argument does the same in every score_* function below."""
    s = {k: within(v, universe) for k, v in revenue_signals(mask(rev, financials), basis_break).items()}
    s["score"] = combine([pct_rank(s["yoy3"]), pct_rank(s["accel"])])
    return s


def score_earnings(ni: pd.DataFrame, rev: pd.DataFrame, gross: pd.DataFrame, oper: pd.DataFrame,
                   months: pd.PeriodIndex, financials=None, basis_break=None, late=None,
                   universe=None) -> dict[str, pd.DataFrame]:
    """Earnings momentum = mean market-wide rank of surprise, acceleration and margin
    change (financials: the first two only). basis_break applies to the margins only:
    parent-only net income equals consolidated income attributable to the parent, so
    the surprise is comparable across the change."""
    s = sue(ni)
    parts = {"sue": q_to_m(s, months, late=late), "accel": q_to_m(s - s.shift(1), months, late=late),
             "margin": mask(q_to_m(margin_change(rev, [gross, oper, ni], basis_break), months, late=late), financials)}
    parts = {k: within(v, universe) for k, v in parts.items()}
    parts["score"] = combine([pct_rank(v) for v in parts.values()], min_parts=2)
    return parts


def score_value(pbr: pd.DataFrame, ni: pd.DataFrame, me: pd.DataFrame, div_yield: pd.DataFrame,
                groups: pd.Series | None, late=None, universe=None) -> dict[str, pd.DataFrame]:
    """Value = mean within-industry rank of B/P, E/P and dividend yield.

    B/P = 1 / official PBR (negative book value ranks last); E/P = net income of the
    last four quarters / market cap (same as EPS / price, but not distorted by stock
    dividends or capital reductions); the yield is the exchange's month-end figure.
    """
    months = me.index
    bp = (1 / pbr).where(pbr > 0)
    bp = bp.mask(pbr < 0, -np.inf)
    ep = (q_to_m(ttm(ni), months, late=late) / me).where(me > 0)
    parts = {"bp": bp.reindex(months), "ep": ep, "dp": div_yield.reindex(months)}
    parts = {k: within(v, universe) for k, v in parts.items()}
    parts["score"] = combine([pct_rank(v, groups) for v in parts.values()], min_parts=2)
    parts["score_market_wide"] = combine([pct_rank(parts[k]) for k in ("bp", "ep", "dp")], min_parts=2)
    return parts


def score_quality(ni: pd.DataFrame, cfo: pd.DataFrame, assets: pd.DataFrame, liab: pd.DataFrame,
                  me: pd.DataFrame, pbr: pd.DataFrame, groups: pd.Series | None,
                  financials=None, late=None, universe=None) -> dict[str, pd.DataFrame]:
    """Quality = mean within-industry rank of ROE (high), accruals (low), ROE stability
    (low sd) and liabilities / assets (low). Financials: ROE and stability only.

    Book equity = market cap / official PBR, so ROE needs no balance sheet.
    """
    months = me.index
    book = (me / pbr).where(pbr > 0)
    roe = (q_to_m(ttm(ni), months, late=late) / book).where(book > 0)
    qe = book[book.index.month % 3 == 0]
    qe.index = qe.index.asfreq("Q")
    niq = full_quarters(ni)
    roe_q = (niq / qe.reindex(index=niq.index, columns=niq.columns)).replace([np.inf, -np.inf], np.nan)
    roe_sd = roe_q.rolling(20, min_periods=12).std(ddof=1)
    acc = ((ttm(ni) - ttm(cfo)) / full_quarters(assets)).where(full_quarters(assets) > 0)
    lev = (liab / assets).where(assets > 0)
    parts = {"roe": roe, "accruals": mask(q_to_m(acc, months, late=late), financials),
             "roe_sd": q_to_m(roe_sd, months, late=late), "leverage": mask(q_to_m(lev, months, late=late), financials)}
    parts = {k: within(v, universe) for k, v in parts.items()}
    signs = {"roe": 1, "accruals": -1, "roe_sd": -1, "leverage": -1}
    parts["score"] = combine([pct_rank(signs[k] * parts[k], groups) for k in signs], min_parts=2)
    return parts


def score_flows(foreign_net: pd.DataFrame, trust_net: pd.DataFrame, margin_bal: pd.DataFrame,
                sbl_bal: pd.DataFrame | None, shares: pd.DataFrame, days: int = 20,
                lag: int = 1, universe=None) -> dict[str, pd.DataFrame]:
    """Stock-level flows = mean market-wide rank of foreign buying (high), investment-
    trust buying (high) and the change in margin balance (low). All inputs in shares.

    Foreign buying is net of short covering: net purchases plus the change in the
    borrowed-share short balance (a falling balance is subtracted).
    lag: these figures are published after the close, so the portfolio formed at the
    month-end close can only use them up to `lag` trading day(s) earlier.
    At least two of the three parts are required.
    """
    foreign_net, trust_net, margin_bal = foreign_net.shift(lag), trust_net.shift(lag), margin_bal.shift(lag)
    sbl_bal = None if sbl_bal is None else sbl_bal.shift(lag)
    f_raw = rolling_at_month_end(foreign_net, days)
    sh = shares.reindex(index=f_raw.index, columns=f_raw.columns)
    f_adj = f_raw if sbl_bal is None else f_raw.add(
        change_at_month_end(sbl_bal, days).reindex(index=f_raw.index, columns=f_raw.columns).fillna(0))
    parts = {"foreign": f_adj / sh, "foreign_raw": f_raw / sh,
             "trust": rolling_at_month_end(trust_net, days).reindex(index=f_raw.index, columns=f_raw.columns) / sh,
             "margin": change_at_month_end(margin_bal, days).reindex(index=f_raw.index, columns=f_raw.columns) / sh}
    parts = {k: within(v.replace([np.inf, -np.inf], np.nan), universe) for k, v in parts.items()}
    parts["score"] = combine([pct_rank(parts["foreign"]), pct_rank(parts["trust"]), pct_rank(-parts["margin"])],
                             min_parts=2)
    return parts


def score_reversal(dret: pd.DataFrame, ret: pd.DataFrame, fac: pd.DataFrame,
                   groups: pd.Series | None, universe=None, skip: int = 1) -> dict[str, pd.DataFrame]:
    """Short-term reversal, three versions; high score = fell the most.

    The ranking return leaves out the last trading day of the month; the position is
    opened at that day's close.
    """
    r = within(month_return_skip_last(dret, skip).clip(*rm.RET_CLIP), universe)
    out = {"raw": -r, "residual": -residual_of(r, ret, fac)}
    if groups is not None:
        out["industry"] = -industry_relative(r, groups)
    return out
