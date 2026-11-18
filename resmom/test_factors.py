"""Unit tests for signals.py, evaluate.py, portfolio.py and walkforward.py on small
simulated data: hand-checked values, timing (no look-ahead) and the trading rules.

Run: python -m pytest test_factors.py -q
"""
import numpy as np
import pandas as pd
import pytest

import data_tw as dt
import evaluate as ev
import portfolio as pf
import resmom as rm
import signals as sg
import walkforward as wf


def M(start, n):
    return pd.period_range(start, periods=n, freq="M")


def Q(start, n):
    return pd.period_range(start, periods=n, freq="Q")


def P(month):
    return pd.Period(month, "M")


def ids(n):
    return [f"s{i:02d}" for i in range(n)]


def factors(months, seed=0):
    r = np.random.default_rng(seed)
    f = pd.DataFrame(r.normal(0, 0.04, (len(months), 3)), index=months, columns=rm.FACTORS)
    f["RF"] = 0.001
    return f


# ============================================================================ signals
def test_pct_rank_market_wide_and_ties():
    x = pd.DataFrame([[1, 2, 3, 4], [4, np.nan, 2, 2]], index=M("2020-01", 2), columns=list("abcd"), dtype=float)
    r = sg.pct_rank(x)
    assert r.iloc[0].tolist() == [0.125, 0.375, 0.625, 0.875]
    assert r.iloc[1, 0] == pytest.approx(2.5 / 3) and np.isnan(r.iloc[1, 1])
    assert r.iloc[1, 2] == r.iloc[1, 3] == pytest.approx(1 / 3)


def test_pct_rank_within_industry_and_small_group_fallback():
    c = ids(14)
    x = pd.DataFrame([np.arange(14.0)], index=M("2020-01", 1), columns=c)
    g = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 2, index=c)
    r = sg.pct_rank(x, g, min_group=5)
    assert r.iloc[0, :6].tolist() == pytest.approx([(i + 0.5) / 6 for i in range(6)])
    assert r.iloc[0, 6:12].tolist() == pytest.approx([(i + 0.5) / 6 for i in range(6)])
    assert r.iloc[0, 12:].tolist() == pytest.approx([12.5 / 14, 13.5 / 14])      # too small: market-wide


def test_combine_uses_available_parts():
    i, c = M("2020-01", 1), ["a", "b"]
    p1 = pd.DataFrame([[0.2, np.nan]], index=i, columns=c)
    p2 = pd.DataFrame([[0.4, 0.6]], index=i, columns=c)
    assert sg.combine([p1, p2]).iloc[0].tolist() == pytest.approx([0.3, 0.6])
    out = sg.combine([p1, p2], min_parts=2).iloc[0]
    assert out["a"] == pytest.approx(0.3) and np.isnan(out["b"])


def test_composite_reranks_each_factor():
    i, c = M("2020-01", 1), ids(4)
    a = pd.DataFrame([[1, 2, 3, 4]], index=i, columns=c, dtype=float)
    b = pd.DataFrame([[1000, 10, 1, 0.1]], index=i, columns=c)        # scale must not matter
    assert sg.composite({"a": a, "b": b}).iloc[0].tolist() == pytest.approx([0.5] * 4)


def test_missing_parts_count_as_the_middle_rank():
    i = pd.period_range("2020-01", periods=1, freq="M")
    p1 = pd.DataFrame([[0.9, 0.9, np.nan, np.nan]], index=i, columns=list("abcd"))
    p2 = pd.DataFrame([[0.7, np.nan, 0.1, np.nan]], index=i, columns=list("abcd"))
    p3 = pd.DataFrame([[0.8, np.nan, np.nan, np.nan]], index=i, columns=list("abcd"))
    out = sg.combine([p1, p2, p3], missing=0.5).iloc[0]
    assert out[:3].tolist() == pytest.approx([0.8, (0.9 + 0.5 + 0.5) / 3, (0.1 + 0.5 + 0.5) / 3]) and np.isnan(out["d"])
    old = sg.combine([p1, p2, p3]).iloc[0]                      # mean of what exists: one part alone can be extreme
    assert old[:3].tolist() == pytest.approx([0.8, 0.9, 0.1])
    assert np.isnan(sg.combine([p1, p2, p3], min_parts=2, missing=0.5).iloc[0][["b", "c", "d"]]).all()


def test_orthogonal_removes_what_the_base_explains_month_by_month():
    rng = np.random.default_rng(3)
    i, cols = pd.period_range("2020-01", periods=6, freq="M"), [f"s{k}" for k in range(300)]
    base = pd.DataFrame(rng.normal(size=(6, 300)), index=i, columns=cols)
    own = pd.DataFrame(rng.normal(size=(6, 300)), index=i, columns=cols)
    x = 0.8 * base + 0.6 * own
    x.iloc[0, :50] = np.nan
    res = sg.orthogonal(x, base)
    rank_corr = lambda a, b: a.rank().corr(b.rank())
    for m in i:
        assert abs(rank_corr(res.loc[m], base.loc[m])) < 0.05 and rank_corr(res.loc[m], own.loc[m]) > 0.9
    assert res.iloc[0, :50].isna().all() and res.iloc[0, 50:].notna().all()
    later = base.copy()
    later.iloc[3:] = rng.normal(size=(3, 300))                  # other months do not matter
    pd.testing.assert_frame_equal(sg.orthogonal(x, later).iloc[:3], res.iloc[:3])
    u = pd.DataFrame(False, index=i, columns=cols)
    u.iloc[:, :10] = True                                       # too few stocks: nothing
    assert sg.orthogonal(x, base, u).isna().all().all()


def _daily(start="2019-01-01", end="2020-12-31"):
    return pd.bdate_range(start, end)


def test_daily_vol_value_window_and_min_days():
    d = _daily()
    x = pd.DataFrame({"a": np.where(np.arange(len(d)) % 2, 0.01, -0.01), "b": 0.0, "c": np.nan}, index=d)
    x.loc[d[-50:], "c"] = 0.02
    v = sg.daily_vol(x, months=12, min_days=120)
    m = pd.Period("2020-12", "M")
    assert v.loc[m, "a"] == pytest.approx(x.loc["2020-01-01":"2020-12-31", "a"].std(ddof=1))
    assert v.loc[m, "b"] == 0 and np.isnan(v.loc[m, "c"])
    early = sg.daily_vol(x.loc[:"2020-06-30"])
    assert early.loc[pd.Period("2020-06", "M"), "a"] == v.loc[pd.Period("2020-06", "M"), "a"]   # no look-ahead


def test_month_return_skip_last_ignores_last_day():
    d = _daily("2020-01-01", "2020-02-28")
    x = pd.DataFrame({"a": 0.01}, index=d)
    jan = d[d.month == 1]
    x.loc[jan[-1], "a"] = 0.50
    r = sg.month_return_skip_last(x)
    assert r.loc[pd.Period("2020-01", "M"), "a"] == pytest.approx(1.01 ** (len(jan) - 1) - 1)
    assert sg.month_return_skip_last(x, skip=0).loc[pd.Period("2020-01", "M"), "a"] == pytest.approx(
        1.01 ** (len(jan) - 1) * 1.5 - 1)


def test_high_52w():
    d = _daily()
    x = pd.DataFrame({"up": 0.001, "drop": 0.0}, index=d)
    x.loc[d[-5], "drop"] = -0.20
    h = sg.high_52w(x)
    m = pd.Period("2020-12", "M")
    assert h.loc[m, "up"] == pytest.approx(1.0) and h.loc[m, "drop"] == pytest.approx(0.8)
    assert h.loc[pd.Period("2020-11", "M"), "drop"] == pytest.approx(1.0)


def test_turnover_abnormal_and_share_adjustment():
    d = _daily()
    v = pd.DataFrame({"a": 100.0, "b": 100.0}, index=d)
    last = d.to_period("M") == pd.Period("2020-12", "M")
    v.loc[last, :] = 200.0
    ab, lr = sg.turnover_stats(v)
    m = pd.Period("2020-12", "M")
    assert ab.loc[m, "a"] == pytest.approx(2.0) and ab.loc[pd.Period("2020-11", "M"), "a"] == pytest.approx(1.0)
    assert np.isnan(ab.loc[pd.Period("2019-06", "M"), "a"])                    # needs 10 earlier months
    sh = pd.DataFrame(1000.0, index=pd.period_range("2019-01", "2020-12", freq="M"), columns=["a", "b"])
    sh.loc[m, "b"] = 2000.0                                                     # b doubled its shares
    ab2, _ = sg.turnover_stats(v, sh)
    assert ab2.loc[m, "a"] == pytest.approx(2.0) and ab2.loc[m, "b"] == pytest.approx(1.0)


def test_revenue_signals_values_and_publication_lag():
    i = M("2018-01", 30)
    rev = pd.DataFrame({"flat": 100.0, "jump": 100.0, "zero": 0.0}, index=i)
    rev.loc[i[12:], "jump"] = 150.0
    rev.loc[i[12:], "zero"] = 50.0
    s = sg.revenue_signals(rev)
    # revenue of 2019-03 (15th month) is first usable at the end of 2019-04
    assert s["yoy3"].loc[pd.Period("2019-04", "M"), "jump"] == pytest.approx(0.5)
    assert np.isnan(s["yoy3"].loc[pd.Period("2019-03", "M"), "jump"])
    assert s["yoy3"].loc[pd.Period("2019-04", "M"), "flat"] == pytest.approx(0.0)
    assert s["yoy3"].loc[:pd.Period("2020-01", "M"), "zero"].isna().all()       # base (same 3 months a year ago) was zero
    assert s["yoy3"].loc[pd.Period("2020-02", "M"), "zero"] == pytest.approx(2.0)   # base 0 + 0 + 50, now 150
    assert s["mom"].loc[pd.Period("2019-02", "M"), "jump"] == pytest.approx(0.5)
    # 24 months needed for acceleration: revenue month 2019-12 -> formation 2020-01
    assert np.isnan(s["accel"].loc[pd.Period("2019-12", "M"), "jump"])
    assert s["accel"].loc[pd.Period("2020-01", "M"), "jump"] == pytest.approx(0.5 - 0.5)
    assert np.isnan(s["d_yoy"].loc[pd.Period("2019-02", "M"), "jump"])          # Dec-18 has no year-ago month
    assert s["d_yoy"].loc[pd.Period("2020-02", "M"), "jump"] == pytest.approx(-0.5)  # Jan-20 yoy 0% vs Dec-19 50%


def test_revenue_basis_break_masks_cross_comparisons():
    i = M("2010-01", 72)
    rev = pd.DataFrame({"a": 100.0}, index=i)
    rev.loc[pd.Period("2013-01", "M"):, "a"] = 150.0            # consolidated from 2013: looks like +50%
    raw = sg.revenue_signals(rev)
    s = sg.revenue_signals(rev, basis_break="2013-01")
    f = lambda k, m: s[k].loc[pd.Period(m, "M"), "a"]           # formation month = revenue month + 1
    assert raw["yoy3"].loc[pd.Period("2013-04", "M"), "a"] == pytest.approx(0.5)     # the false growth
    assert f("yoy3", "2013-01") == 0 and np.isnan(f("yoy3", "2013-02")) and np.isnan(f("yoy3", "2014-03"))
    assert f("yoy3", "2014-04") == 0                             # revenue 2014-03 vs 2013-03: same basis again
    assert np.isnan(f("mom", "2013-02")) and f("mom", "2013-03") == 0
    assert np.isnan(f("accel", "2014-12")) and f("accel", "2015-01") == 0
    assert np.isnan(f("d_yoy", "2014-02")) and f("d_yoy", "2014-03") == 0
    q = Q("2012Q1", 12)
    one = pd.DataFrame({"a": 100.0}, index=q)
    gp = pd.DataFrame({"a": [30.0] * 4 + [45.0] * 8}, index=q)
    mc = sg.margin_change(one, [gp], basis_break="2013Q1")["a"]
    assert mc.loc["2013Q1":"2013Q4"].isna().all() and mc.loc[pd.Period("2014Q1", "Q")] == 0
    assert sg.margin_change(one, [gp])["a"].loc[pd.Period("2013Q1", "Q")] == pytest.approx(0.15)


def test_revenue_no_look_ahead():
    r = np.random.default_rng(1)
    i = M("2015-01", 60)
    rev = pd.DataFrame(r.uniform(50, 150, (60, 5)), index=i, columns=ids(5))
    full, cut = sg.revenue_signals(rev), sg.revenue_signals(rev.iloc[:40])
    for k in full:
        pd.testing.assert_frame_equal(full[k].loc[cut[k].index], cut[k])


def test_usable_month_deadlines():
    u = lambda s: str(sg.usable_month(pd.Period(s, "Q")))
    assert [u("2015Q1"), u("2015Q2"), u("2015Q3"), u("2015Q4")] == ["2015-05", "2015-08", "2015-11", "2016-04"]
    assert [u("2010Q1"), u("2010Q2"), u("2010Q3"), u("2010Q4")] == ["2010-05", "2010-09", "2010-11", "2011-05"]


def test_q_to_m_point_in_time_expiry_and_collision():
    q = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=Q("2019Q4", 3))
    m = sg.q_to_m(q, pd.period_range("2020-01", "2021-06", freq="M"), max_age=8)
    assert m.loc["2020-01":"2020-03", "a"].isna().all()
    assert m.loc[pd.Period("2020-04", "M"), "a"] == 1 and m.loc[pd.Period("2020-05", "M"), "a"] == 2
    assert m.loc[pd.Period("2020-07", "M"), "a"] == 2 and m.loc[pd.Period("2020-08", "M"), "a"] == 3
    assert m.loc[pd.Period("2021-04", "M"), "a"] == 3 and np.isnan(m.loc[pd.Period("2021-05", "M"), "a"])
    q2 = pd.DataFrame({"a": [10.0, 20.0], "b": [10.0, np.nan]}, index=Q("2012Q4", 2))   # both usable 2013-05
    m2 = sg.q_to_m(q2, pd.period_range("2013-04", "2013-06", freq="M"))
    assert np.isnan(m2.loc[pd.Period("2013-04", "M"), "a"])
    assert m2.loc[pd.Period("2013-05", "M")].tolist() == [20.0, 10.0]


def test_ttm_and_sue_need_full_history():
    ni = pd.DataFrame({"a": [0.0] * 4 + [1, 2, 3, 4] + [2, 4, 6, 8]}, index=Q("2018Q1", 12))
    t = sg.ttm(ni)
    assert np.isnan(t.iloc[2, 0]) and t.iloc[3, 0] == 0 and t.iloc[7, 0] == 10
    s = sg.sue(ni)
    assert s.iloc[:11, 0].isna().all()                                          # 12 quarters needed
    assert s.iloc[11, 0] == pytest.approx(4 / np.std([1, 2, 3, 4, 1, 2, 3, 4], ddof=1))


def test_margin_change():
    i = Q("2019Q1", 5)
    rev = pd.DataFrame({"a": 100.0, "b": [100, 100, 100, 100, 0.0]}, index=i)
    gp = pd.DataFrame({"a": [30, 30, 30, 30, 40.0], "b": 30.0}, index=i)
    op = pd.DataFrame({"a": 10.0, "b": 10.0}, index=i)
    ni = pd.DataFrame({"a": [5, 5, 5, 5, 8.0], "b": 5.0}, index=i)
    mc = sg.margin_change(rev, [gp, op, ni])
    assert mc.iloc[4]["a"] == pytest.approx((0.10 + 0.0 + 0.03) / 3)
    assert np.isnan(mc.iloc[4]["b"]) and mc.iloc[:4]["a"].isna().all()


def _ff_panel(n_m=60, n_s=40, seed=3):
    r = np.random.default_rng(seed)
    mo = M("2015-01", n_m)
    f = factors(mo, seed)
    beta = r.uniform(0.5, 1.5, n_s)
    ret = pd.DataFrame(f["RF"].to_numpy()[:, None] + np.outer(f["RMRF"], beta) + r.normal(0, 0.002, (n_m, n_s)),
                       index=mo, columns=ids(n_s))
    return ret, f


def test_residual_of_removes_factor_part_and_has_no_look_ahead():
    ret, f = _ff_panel()
    m = ret.index[50]
    ret.loc[m, "s00"] += 0.20
    res = sg.residual_of(ret, ret, f)
    assert res.loc[m, "s00"] == pytest.approx(0.20, abs=0.02)
    assert res.loc[m].drop("s00").abs().max() < 0.02
    assert res.index.min() == ret.index[36]                                     # 36 earlier months needed
    later = ret.copy()
    later.iloc[51:] += 0.5
    assert sg.residual_of(later, later, f).loc[m].equals(res.loc[m])
    vol = sg.residual_vol(ret, f)
    assert vol.index.min() == ret.index[35] and vol.loc[ret.index[45]].between(0.001, 0.004).all()


def test_industry_relative():
    c = ids(12)
    x = pd.DataFrame([np.r_[np.arange(6.0), np.arange(6.0) + 100]], index=M("2020-01", 1), columns=c)
    g = pd.Series(["A"] * 6 + ["B"] * 6, index=c)
    out = sg.industry_relative(x, g).iloc[0]
    assert out.iloc[:6].tolist() == pytest.approx(out.iloc[6:].tolist())
    assert out.iloc[0] == pytest.approx(-2.5)


def _fundamentals(n=12):
    mo = M("2017-01", 60)
    c = ids(n)
    qs = Q("2015Q1", 28)
    ni = pd.DataFrame(np.tile(np.linspace(-5, 6, n), (len(qs), 1)), index=qs, columns=c)       # s00 loses most
    me = pd.DataFrame(100.0, index=mo, columns=c)
    pbr = pd.DataFrame(np.tile(np.linspace(0.5, 3.0, n), (len(mo), 1)), index=mo, columns=c)
    return mo, c, qs, ni, me, pbr


def test_score_value_directions():
    mo, c, qs, ni, me, pbr = _fundamentals()
    pbr["s11"] = -1.0                                                           # negative book value
    dy = pd.DataFrame(np.tile(np.r_[0.0, np.linspace(1, 6, len(c) - 1)], (len(mo), 1)), index=mo, columns=c)
    v = sg.score_value(pbr, ni, me, dy, None)
    row = v["ep"].loc[mo[-1]]
    assert row.idxmin() == "s00" and row.idxmax() == "s11" and row["s00"] < 0   # bigger loss ranks lower
    assert v["bp"].loc[mo[-1], "s11"] == -np.inf
    assert sg.pct_rank(v["bp"]).loc[mo[-1]].idxmin() == "s11"
    assert v["ep"].loc[mo[-1], "s11"] == pytest.approx(4 * 6 / 100)             # four quarters / market cap
    assert sg.pct_rank(v["dp"]).loc[mo[-1]].idxmin() == "s00"


def test_score_quality_directions_and_financials():
    mo, c, qs, ni, me, pbr = _fundamentals()
    pbr[:] = 1.0
    cfo = ni.copy()
    cfo["s05"] = ni["s05"] - 10                                                 # profit not backed by cash
    assets = pd.DataFrame(100.0, index=qs, columns=c)
    liab = pd.DataFrame(np.tile(np.linspace(10, 90, len(c)), (len(qs), 1)), index=qs, columns=c)
    qv = sg.score_quality(ni, cfo, assets, liab, me, pbr, None, financials=["s02"])
    m = mo[-1]
    assert qv["roe"].loc[m, "s11"] == pytest.approx(4 * 6 / 100) and qv["roe"].loc[m].idxmax() == "s11"
    assert qv["accruals"].loc[m, "s05"] == pytest.approx(0.40) and qv["accruals"].loc[m, "s04"] == 0
    assert np.isnan(qv["accruals"].loc[m, "s02"]) and np.isnan(qv["leverage"].loc[m, "s02"])
    assert qv["leverage"].loc[m, "s00"] == pytest.approx(0.10)
    assert qv["roe_sd"].loc[m].max() < 1e-12                                    # constant earnings
    assert not np.isnan(qv["score"].loc[m, "s02"])


def test_score_earnings_parts():
    mo, c, qs, ni, me, pbr = _fundamentals()
    r = np.random.default_rng(5)
    ni = ni + r.normal(0, 1, ni.shape)
    ni.iloc[-2, 3] += 50                                                        # s03 surprises in 2021Q3
    rev = pd.DataFrame(100.0, index=qs, columns=c)
    e = sg.score_earnings(ni, rev, ni * 3, ni * 2, mo, financials=["s02"])
    m = sg.usable_month(qs[-2])                                                 # 2021-11
    assert e["sue"].loc[m].idxmax() == "s03" and e["accel"].loc[m].idxmax() == "s03"
    assert e["sue"].loc[m - 1, "s03"] != e["sue"].loc[m, "s03"]                 # not visible a month earlier
    assert np.isnan(e["margin"].loc[m, "s02"]) and not np.isnan(e["score"].loc[m, "s02"])


def test_score_flows_nets_out_short_covering():
    d = pd.bdate_range("2020-01-01", "2020-03-31")
    c = ["a", "b"]
    fn = pd.DataFrame({"a": 10.0, "b": 10.0}, index=d)
    tn = pd.DataFrame({"a": 0.0, "b": 5.0}, index=d)
    mb = pd.DataFrame({"a": 100.0, "b": 100.0}, index=d)
    mb.loc[d[-1], "b"] = 150.0
    sbl = pd.DataFrame({"a": 500.0, "b": 500.0}, index=d)
    sbl.loc[d[-10]:, "a"] = 400.0                                               # shorts covered 100 shares
    sh = pd.DataFrame(1000.0, index=pd.period_range("2020-01", "2020-03", freq="M"), columns=c)
    f = sg.score_flows(fn, tn, mb, sbl, sh)
    m = pd.Period("2020-03", "M")
    assert f["foreign_raw"].loc[m, "a"] == pytest.approx(0.2) and f["foreign"].loc[m, "a"] == pytest.approx(0.1)
    assert f["foreign"].loc[m, "b"] == pytest.approx(0.2) and f["trust"].loc[m, "b"] == pytest.approx(0.1)
    assert f["margin"].loc[m, "b"] == 0                       # the last day's figure is published after the close
    assert sg.score_flows(fn, tn, mb, sbl, sh, lag=0)["margin"].loc[m, "b"] == pytest.approx(0.05)
    assert f["score"].loc[m, "a"] < f["score"].loc[m, "b"]


def test_score_reversal_sign():
    d = _daily("2019-01-01", "2019-03-29")
    x = pd.DataFrame({"a": 0.01, "b": -0.01}, index=d)
    mo = pd.period_range("2019-01", "2019-03", freq="M")
    r = sg.score_reversal(x, pd.DataFrame(0.0, index=mo, columns=["a", "b"]), factors(mo), None)
    assert (r["raw"]["b"] > 0).all() and (r["raw"]["a"] < 0).all()


# ========================================================================= data layer
def test_clean_daily_returns_listed_only_and_price_limit():
    d = pd.bdate_range("2015-05-04", "2015-07-31")
    close = pd.DataFrame(100.0, index=d, columns=["old", "new", "otc"])
    ret = pd.DataFrame(0.01, index=d, columns=close.columns)
    ret.loc["2015-05-20", "old"] = 0.09                 # above the 7% limit of that time
    ret.loc["2015-06-10", "old"] = 0.09                 # fine under the 10% limit
    ret.loc["2015-07-10", "old"] = -0.46                # a missing split adjustment
    ret.loc["2015-07-02", "new"] = 0.80                 # 2nd day after listing: no limit
    ret.loc["2015-07-20", "new"] = 0.30                 # later: impossible
    mo = pd.period_range("2015-05", "2015-07", freq="M")
    listed = pd.DataFrame({"old": True, "new": [False, False, True], "otc": False}, index=mo)
    r, n = dt.clean_daily_returns(ret, close, listed)
    assert n == 3 and r["otc"].isna().all() and r.loc[:"2015-06-30", "new"].isna().all()
    assert np.isnan(r.loc["2015-05-20", "old"]) and r.loc["2015-06-10", "old"] == 0.09
    assert np.isnan(r.loc["2015-07-10", "old"]) and r.loc["2015-07-02", "new"] == 0.80
    assert np.isnan(r.loc["2015-07-20", "new"]) and r.loc["2015-07-21", "new"] == 0.01


def test_monthly_from_daily_needs_previous_month():
    d = pd.bdate_range("2020-01-01", "2020-04-30")
    r = pd.DataFrame({"a": 0.001, "b": 0.001}, index=d)
    r.loc[:"2020-02-29", "b"] = np.nan                  # b starts in March
    m = dt.monthly_from_daily(r)
    n_mar = (d.to_period("M") == pd.Period("2020-03", "M")).sum()
    assert m.loc[pd.Period("2020-03", "M"), "a"] == pytest.approx(1.001 ** n_mar - 1)
    assert np.isnan(m.loc[pd.Period("2020-01", "M"), "a"])                      # no earlier month
    assert np.isnan(m.loc[pd.Period("2020-03", "M"), "b"]) and not np.isnan(m.loc[pd.Period("2020-04", "M"), "b"])
    big = pd.DataFrame({"a": [0.0] * 30 + [5.0]}, index=pd.bdate_range("2020-01-01", periods=31))
    assert dt.monthly_from_daily(big).iloc[-1, 0] == 1.0                        # clipped at +100%


def test_exit_returns_actual_where_known_and_only_then():
    d = pd.bdate_range("2020-01-01", "2020-08-31")
    m = d.to_period("M")
    close = pd.DataFrame(10.0, index=d, columns=["dead", "back", "cut4", "end"])
    close.loc[m >= "2020-03", "dead"] = np.nan                           # never trades again
    close.loc[(m >= "2020-03") & (m <= "2020-04"), ["back", "cut4"]] = np.nan
    close.loc[m >= "2020-05", "back"] = 6.0                               # back after two months, 40% lower
    close.loc[m >= "2020-05", "cut4"] = 40.0                              # capital reduction: price x 4 ...
    close.loc[m >= "2020-08", "end"] = np.nan                             # stops in the last month of the data
    dadj = close.pct_change(fill_method=None)
    first_back = d[m == "2020-05"][0]
    dadj.loc[first_back, "back"], dadj.loc[first_back, "cut4"] = -0.40, 3.0     # ... and the adjustment data misses it
    i = M("2020-01", 8)
    ret = dt.monthly_from_daily(dadj.where(dadj.abs() < 0.1)).reindex(i)
    shares = pd.DataFrame(100.0, index=i, columns=close.columns)
    shares.loc[P("2020-05"):, "cut4"] = 25.0                              # ... shares / 4
    gone, actual, events = dt.exit_returns(ret, close, dadj, shares)
    assert gone.sum().to_dict() == {"dead": 1, "back": 1, "cut4": 1, "end": 0}      # the data's last month is never an exit
    assert gone.loc[P("2020-03")].tolist() == [True, True, True, False] and not gone.loc[P("2020-04")].any()
    assert np.isnan(actual.loc[P("2020-03"), "dead"])                     # unknown: left to the conservative fill
    assert actual.loc[P("2020-03"), "back"] == pytest.approx(-0.40)
    assert np.isnan(actual.loc[P("2020-03"), "cut4"])                     # +300% by price, 0% by value: not known
    e = events.set_index("stock")
    assert e.loc["cut4", "price_based"] == pytest.approx(3.0) and e.loc["cut4", "cap_based"] == pytest.approx(0.0)
    assert e.loc["back", "next_trade"] == first_back and e.loc["back", "cap_based"] == pytest.approx(-0.40)
    assert pd.isna(e.loc["dead", "next_trade"]) and np.isnan(ret.loc[P("2020-05"), "back"])     # no double counting
    assert dt.exit_returns(ret, close, dadj, shares, incomplete_last=False)[0]["end"].sum() == 1
    # cut the data at April: nobody knows yet that two of them come back
    early = d[m <= "2020-04"]
    g2, a2, _ = dt.exit_returns(ret.loc[:P("2020-04")], close.loc[early], dadj.loc[early], shares.loc[:P("2020-04")],
                                incomplete_last=False)
    assert g2.loc[P("2020-03")].tolist() == [True, True, True, False] and a2.isna().all().all()
    # a gap longer than max_gap is a different company using the code: not an actual return
    assert dt.exit_returns(ret, close, dadj, shares, max_gap=2)[1].isna().all().all()

    # merger exits: only a stock that never comes back, with a delisting dated near its last trade
    last = d[m == "2020-02"][-1]
    T = pd.Timestamp
    mg = pd.DataFrame({"stock": ["dead", "back", "cut4", "other"],
                       "delisted": [last + pd.Timedelta(days=9), last + pd.Timedelta(days=9), T("2015-01-01"), last]})
    flag = lambda mergers, by="2020-12-31", **k: dict(zip(events["stock"], dt.merger_flags(events, mergers, T(by), **k)))
    assert flag(mg) == {"dead": True, "back": False, "cut4": False}       # back: it traded again; cut4: wrong date
    assert flag(mg, by=str((last + pd.Timedelta(days=8)).date())) == {"dead": False, "back": False, "cut4": False}   # not yet known
    assert not any(flag(None).values()) and not any(flag(mg.iloc[:0]).values())
    far = mg.assign(delisted=last + pd.Timedelta(days=200))               # an old code used again much later
    assert not any(flag(far).values()) and flag(far, window=250)["dead"]
    assert len(dt.merger_exits()) > 100 and dt.merger_exits()["delisted"].notna().all()
    assert not dt.merger_exits().duplicated(["stock", "delisted"]).any()


def test_late_filers_are_used_one_month_later():
    q = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "bank": [1.0, 2.0, 3.0, 4.0]}, index=Q("2020Q1", 4))
    m = sg.q_to_m(q, pd.period_range("2020-04", "2021-06", freq="M"), late=["bank"])
    at = lambda s, c: m.loc[pd.Period(s, "M"), c]
    assert at("2020-05", "a") == 1 and np.isnan(at("2020-05", "bank")) and at("2020-06", "bank") == 1
    assert at("2020-08", "a") == 2 and at("2020-08", "bank") == 1 and at("2020-09", "bank") == 2
    assert at("2020-11", "a") == 3 and at("2020-11", "bank") == 2 and at("2020-12", "bank") == 3
    assert at("2021-04", "a") == 4 and at("2021-04", "bank") == 3 and at("2021-05", "bank") == 4
    assert list(m.columns) == ["a", "bank"]


def test_ls_returns_separate_short_leg_returns():
    c, i = ids(40), M("2020-01", 3)
    s = pd.DataFrame([np.arange(40.0)] * 3, index=i, columns=c)
    ret = pd.DataFrame(0.0, index=i, columns=c)
    lo, sh = ret.copy(), ret.copy()
    lo.iloc[1, :4], sh.iloc[1, :4] = -0.30, 0.0          # the four losers stop trading
    lo.iloc[1, -4:], sh.iloc[1, -4:] = -0.30, 0.0        # and so do the four winners
    p = ev.ls_returns(s.iloc[[0]], lo, ret_short=sh)
    assert p["W"].iloc[0] == pytest.approx(-0.30) and p["L"].iloc[0] == pytest.approx(0.0)
    assert p["WML"].iloc[0] == pytest.approx(-0.30)      # neither leg benefits from the unknown ending


def test_ytd_to_quarter():
    q = Q("2019Q1", 8)
    ytd = pd.DataFrame({"a": [10, 25, 45, 70, 5, 5, np.nan, 40.0]}, index=q)
    out = dt.ytd_to_quarter(ytd)["a"]
    assert out.iloc[:6].tolist() == [10, 15, 20, 25, 5, 0]
    assert out.iloc[6:].isna().all()                    # Q3 missing, so Q4 alone cannot be derived


# =========================================================================== evaluate
def test_forward_return():
    ret = pd.DataFrame({"a": [0.0, 0.1, 0.2, -0.5]}, index=M("2020-01", 4))
    f1, f2 = ev.forward_return(ret), ev.forward_return(ret, 2)
    assert f1["a"].iloc[:3].tolist() == pytest.approx([0.1, 0.2, -0.5]) and np.isnan(f1["a"].iloc[3])
    assert f2["a"].iloc[1] == pytest.approx(1.2 * 0.5 - 1) and f2["a"].iloc[2:].isna().all()
    assert ev.forward_return(ret, 1, skip=1)["a"].iloc[0] == pytest.approx(0.2)


def test_rank_ic_perfect_reversed_universe_and_min_n():
    c, i = ids(40), M("2020-01", 2)
    s = pd.DataFrame([np.arange(40.0)] * 2, index=i, columns=c)
    f = s.copy()
    f.iloc[1] = f.iloc[1].to_numpy()[::-1]
    ic = ev.rank_ic(s, f)
    assert ic.iloc[0] == pytest.approx(1) and ic.iloc[1] == pytest.approx(-1)
    assert ev.rank_ic(s.iloc[:, :10], f.iloc[:, :10]).isna().all()
    f2 = s.copy()
    f2.iloc[0, 20:] = -f2.iloc[0, 20:]
    u = pd.DataFrame(False, index=i, columns=c)
    u.iloc[:, 20:] = True
    assert ev.rank_ic(s, f2, u, min_n=10).iloc[0] == pytest.approx(-1)


def test_ls_returns_uses_next_month_only():
    c, i = ids(40), M("2020-01", 6)
    s = pd.DataFrame([np.arange(40.0)] * 6, index=i, columns=c)
    ret = pd.DataFrame(0.0, index=i, columns=c)
    ret.iloc[3, -4:], ret.iloc[3, :4] = 0.10, -0.10
    ret.iloc[2] = np.linspace(0.9, -0.9, 40)                                    # formation month: must not count
    p = ev.ls_returns(s.iloc[[2]], ret)
    assert list(p.index) == [i[3]] and p["WML"].iloc[0] == pytest.approx(0.20)
    assert p["W"].iloc[0] == pytest.approx(0.10) and p["L"].iloc[0] == pytest.approx(-0.10)


def test_net_ls_and_breakeven():
    port = pd.DataFrame({"WML": [0.02, 0.02], "TO_W": [np.nan, 0.5], "TO_L": [np.nan, 0.5]}, index=M("2020-01", 2))
    net = ev.net_ls(port, 0.001, 0.004, borrow=0.03)
    assert net.tolist() == pytest.approx([0.02 - 1.0 * 0.005 - 0.0025] * 2)
    assert ev.breakeven_cost(port) == pytest.approx(0.02)


def _planted(n_m=24, n_s=60, seed=7):
    r = np.random.default_rng(seed)
    i, c = M("2018-01", n_m), ids(n_s)
    x = pd.DataFrame(r.normal(size=(n_m, n_s)), index=i, columns=c)
    return r, i, c, x


def test_condition_report_finds_planted_relation():
    r, i, c, x = _planted()
    fwd = x.rank(axis=1) * 0.001 + r.normal(0, 0.01, x.shape)
    sub = pd.DataFrame(True, index=i, columns=c)
    sub.iloc[:, 45:] = False
    rep = ev.condition_report(x, fwd, sub)
    assert rep["mean_ic"] > 0.3 and rep["t"] > 3 and rep["monotonic"] and rep["high_minus_low_%/mo"] > 0
    g = ev.group_returns(x, fwd, sub)
    assert list(g.columns) == [1, 2, 3] and len(g) == 24
    none = ev.condition_report(x, pd.DataFrame(r.normal(0, 0.01, x.shape), index=i, columns=c), sub)
    assert abs(none["t"]) < 3


def test_two_by_two_effect_only_where_b_is_low():
    r, i, c, a = _planted(seed=8)
    b = pd.DataFrame(r.normal(size=a.shape), index=i, columns=c)
    low = b.lt(b.median(axis=1), axis=0) | b.eq(b.median(axis=1), axis=0)
    fwd = (a * 0.02).where(low, 0.0) + r.normal(0, 0.002, a.shape)
    t = ev.two_by_two(a, b, fwd)
    assert t["spread_in_b_lo_%/mo"] > 2 and abs(t["spread_in_b_hi_%/mo"]) < 0.3
    assert t["difference_t"] > 3 and t["ic_in_b_lo"] > 0.8 and abs(t["ic_in_b_hi"]) < 0.2


def test_half_life_definition():
    fast = pd.Series({1: 0.002, 5: 0.010, 10: 0.011, 20: 0.012, 40: 0.012, 60: 0.012})
    assert ev.half_life(fast) == 7.5
    assert ev.half_life(pd.Series({h: 0.002 * h for h in ev.HORIZONS})) == np.inf
    assert np.isnan(ev.half_life(-fast)) and np.isnan(ev.half_life(fast.drop(5)))
    slow = pd.Series({1: 0.001, 5: 0.005, 10: 0.010, 20: 0.020, 40: 0.025, 60: 0.027})
    assert ev.half_life(slow) == 30


def test_decay_daily_fast_signal():
    d = pd.bdate_range("2020-01-01", "2020-12-31")
    c = ids(40)
    dret = pd.DataFrame(0.0, index=d, columns=c)
    me = sg.month_ends(d)
    sig = pd.DataFrame([np.arange(40.0)] * 8, index=me.index[:8], columns=c)
    pos = {x: k for k, x in enumerate(d)}
    for m in sig.index:                                         # +0.5% a day for 5 days after the signal, top 4 only
        k = pos[me[m]]
        dret.iloc[k + 1:k + 6, -4:] += 0.005
    sp, ic = ev.decay_daily(sig, dret)
    mean = sp.mean()
    assert mean[1] == pytest.approx(0.005) and mean[5] == pytest.approx(1.005 ** 5 - 1)
    assert mean[10] == pytest.approx(mean[5]) and ev.half_life(mean) == 7.5
    assert (ic[5] > 0).all()


def test_trailing_pctile_excludes_today():
    x = pd.Series(np.r_[np.arange(600.0), -1.0])
    p = ev.trailing_pctile(x, window=750, min_obs=500)
    assert np.isnan(p.iloc[499]) and p.iloc[500] == 1.0 and p.iloc[-1] == 0.0


def test_alpha_stats_recovers_alpha_and_beta():
    r = np.random.default_rng(2)
    i = M("2010-01", 120)
    mkt = pd.Series(r.normal(0.01, 0.05, 120), index=i)
    y = 0.004 + 0.5 * mkt + pd.Series(r.normal(0, 0.001, 120), index=i)
    a, t, b = ev.alpha_stats(y, mkt)
    assert a == pytest.approx(0.004, abs=5e-4) and b == pytest.approx(0.5, abs=0.01) and t > 10
    rf = pd.Series(0.001, index=i)
    a2, _, b2 = ev.alpha_stats(y + rf, mkt + rf, rf)
    assert a2 == pytest.approx(a) and b2 == pytest.approx(b)


def test_factor_report_on_planted_factor():
    r = np.random.default_rng(4)
    i, c = M("2012-01", 61), ids(60)
    s = pd.DataFrame(r.normal(size=(61, 60)), index=i, columns=c)
    ret = (s.shift(1) * 0.01 + r.normal(0, 0.02, s.shape)).fillna(0)
    rep = ev.factor_report(s, ret, buy=0.0005, sell=0.0035)
    assert rep["months"] == 60 and rep["gross_t"] > 5 and rep["ic_mean_ic"] > 0.2
    assert rep["net_%/mo"] < rep["gross_%/mo"] and rep["pass"]
    assert rep["turnover_%/mo_per_leg"] > 50                                    # a fresh random signal each month
    assert rep["breakeven_roundtrip_%"] > 0.4


def test_state_report_counts_episodes():
    r = np.random.default_rng(9)
    d = pd.bdate_range("2015-01-01", "2020-12-31")
    mkt = pd.Series(r.normal(0.0003, 0.01, len(d)), index=d)
    ind = pd.Series(r.normal(size=len(d)), index=d)
    rep = ev.state_report(ind, mkt, window=250)
    assert rep["bad_days"] > 0 and rep["episodes"] <= rep["bad_days"] and abs(rep["rank_corr"]) < 0.3


# ========================================================================== portfolio
def test_tick_size():
    p = [5, 10, 49.9, 50, 99, 100, 499, 500, 999, 1000]
    assert pf.tick_size(p).tolist() == [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 5.0]


def test_limit_flags_10_and_7_percent():
    new = pd.DataFrame({"a": [100.0, 110.0, 99.0], "b": [100.0, 107.0, 107.0]}, index=pd.bdate_range("2020-03-02", periods=3))
    up, dn = pf.limit_flags(new)
    assert up["a"].tolist() == [False, True, False] and dn["a"].tolist() == [False, False, True]
    assert not up["b"].any() and not dn["b"].any()
    old = pd.DataFrame({"a": [100.0, 107.0], "b": [100.0, 106.5]}, index=pd.bdate_range("2014-03-03", periods=2))
    up, _ = pf.limit_flags(old)
    assert up["a"].tolist() == [False, True] and not up["b"].any()


def test_costs_components():
    c = pf.Costs()
    b = c.buy(10_000, 100, 1e9, 0.02)
    assert float(b) == pytest.approx(0.000285 + 0.5 * 0.5 / 100 + 0.02 * np.sqrt(1e4 / 1e9))
    assert float(c.sell(10_000, 100, 1e9, 0.02)) == pytest.approx(float(b) + 0.003)
    assert float(pf.Costs(spread_ticks=0, impact_k=0).buy(1_000, 100, 1e9, 0.02)) == pytest.approx(0.001)   # NT$1 minimum
    big, small = c.buy(1e8, 100, 1e8, 0.02), c.buy(1e4, 100, 1e8, 0.02)
    assert float(big) - float(small) == pytest.approx(0.02 * (1 - 0.01) - 0.0, abs=1e-3)


FREE = pf.Costs(fee=0, min_fee=0, tax=0, spread_ticks=0, impact_k=0, borrow=0)
PLAIN = pf.Costs(spread_ticks=0, impact_k=0)             # fee both ways + tax on sells only


def _sim_inputs(cols, n_m, alpha_rows):
    i = M("2020-01", n_m)
    alpha = pd.DataFrame(alpha_rows, index=i[:len(alpha_rows)], columns=cols, dtype=float)
    const = lambda v: pd.DataFrame(v, index=i, columns=cols)
    return i, alpha, const


def test_simulate_accounting_without_and_with_costs():
    i, alpha, const = _sim_inputs(["a"], 3, [[0.05], [0.05]])
    ret = pd.DataFrame({"a": [0.0, 0.10, -0.05]}, index=i)
    out = pf.simulate(alpha, ret, const(100.0), const(1e9), const(0.0), n=1, costs=FREE)["monthly"]
    assert out["net"].tolist() == pytest.approx([0.10, -0.05]) and out["names"].tolist() == [1, 1]
    assert out["turnover"].tolist() == pytest.approx([0.5, 0.0])
    out = pf.simulate(alpha, ret, const(100.0), const(1e9), const(0.0), n=1, costs=PLAIN)["monthly"]
    v = 1e6 / 1.000285                                                          # bought after paying the fee
    assert out["cost"].iloc[0] == pytest.approx(v * 0.000285 / 1e6) and out["cost"].iloc[1] == 0
    assert out["net"].iloc[0] == pytest.approx(v * 0.10 / 1e6 - v * 0.000285 / 1e6)


def test_soft_penalty_blocks_marginal_swap_and_allows_strong_one():
    rows = [[0.05, 0.04], [0.04, 0.0405], [0.04, 0.10]]
    i, alpha, const = _sim_inputs(["A", "B"], 4, rows)
    ret = pd.DataFrame(0.0, index=i, columns=["A", "B"])
    args = (alpha, ret, const(100.0), const(1e9), const(0.0))
    res = pf.simulate(*args, n=1, gamma=1.0, costs=PLAIN)
    log = res["log"]
    assert log[i[0]]["bought"] == ["A"]
    assert log[i[1]]["bought"] == [] and log[i[1]]["blocked"] == ["B"] and log[i[1]]["kept"] == ["A"]
    assert log[i[2]]["sold"] == ["A"] and log[i[2]]["bought"] == ["B"]
    free = pf.simulate(*args, n=1, gamma=0.0, costs=PLAIN)["log"]
    assert free[i[1]]["bought"] == ["B"] and free[i[1]]["blocked"] == []
    # the threshold is exactly gamma x (sell cost of the old + buy cost of the new)
    thr = (0.000285 + 0.003) + 0.000285
    for diff, swapped in ((thr * 0.9, False), (thr * 1.1, True)):
        a2 = alpha.copy()
        a2.iloc[1] = [0.04, 0.04 + diff]
        lg = pf.simulate(a2, ret, const(100.0), const(1e9), const(0.0), n=1, costs=PLAIN)["log"]
        assert (lg[i[1]]["bought"] == ["B"]) is swapped


def test_position_cap_makes_large_capital_hold_more_names():
    c = ids(100)
    i, alpha, const = _sim_inputs(c, 2, [np.arange(100.0) / 1000])
    ret = pd.DataFrame(0.0, index=i, columns=c)
    args = (alpha, ret, const(100.0), const(1e8), const(0.02))
    small = pf.simulate(*args, capital=1e6, n=30)["monthly"]
    large = pf.simulate(*args, capital=1e9, n=30)["monthly"]
    assert small["names"].iloc[0] == 30 and large["names"].iloc[0] == 50        # 1e9 / (20% x 1e8)
    assert large["cost"].iloc[0] > 3 * small["cost"].iloc[0]                    # impact grows with size


def test_industry_cap_and_trading_flags():
    c = ids(40)
    i, alpha, const = _sim_inputs(c, 3, [np.arange(40.0) / 1000] * 2)
    ret = pd.DataFrame(0.0, index=i, columns=c)
    ind = pd.Series([f"y{k}" for k in range(20)] + ["X"] * 20, index=c)          # the 20 best are all in X
    args = (alpha, ret, const(100.0), const(1e9), const(0.0))
    log = pf.simulate(*args, n=10, industry=ind, costs=FREE)["log"]
    held = log[i[0]]["bought"]
    assert len(held) == 10 and sum(ind[s] == "X" for s in held) == 3
    assert set(s for s in held if ind[s] == "X") == {"s39", "s38", "s37"}
    nb = const(False)
    nb["s39"] = True                                                            # limit-up: cannot buy
    held = pf.simulate(*args, n=5, no_buy=nb, costs=FREE)["log"][i[0]]["bought"]
    assert "s39" not in held and "s38" in held
    a2 = alpha.copy()
    a2.loc[i[1], "s39"] = -1.0                                                  # collapses, but limit-down: cannot sell
    ns = const(False)
    ns.loc[i[1], "s39"] = True
    lg = pf.simulate(a2, ret, const(100.0), const(1e9), const(0.0), n=5, no_sell=ns, costs=FREE)["log"]
    assert "s39" not in lg[i[1]]["sold"]
    lg = pf.simulate(a2, ret, const(100.0), const(1e9), const(0.0), n=5, costs=FREE)["log"]
    assert "s39" in lg[i[1]]["sold"]
    el = const(True)
    el["s38"] = False                                                           # fails the liquidity floor
    assert "s38" not in pf.simulate(*args, n=5, eligible=el, costs=FREE)["log"][i[0]]["bought"]


def test_short_book_sign_and_borrow_fee():
    i, alpha, const = _sim_inputs(["A", "B"], 2, [[0.05, -0.05]])
    ret = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, -0.10]}, index=i)
    out = pf.simulate(alpha, ret, const(100.0), const(1e9), const(0.0), n=1, short=True,
                      costs=pf.Costs(fee=0, min_fee=0, tax=0, spread_ticks=0, impact_k=0, borrow=0.03))
    row = out["monthly"].iloc[0]
    assert out["log"][i[0]]["bought"] == ["B"]
    assert row["gross"] == pytest.approx(0.10) and row["borrow"] == pytest.approx(0.0025)
    assert row["net"] == pytest.approx(0.10 - 0.0025)


def test_missing_return_counts_as_zero_and_position_is_closed():
    i, alpha, const = _sim_inputs(["A", "B"], 3, [[0.05, 0.01], [np.nan, 0.01]])
    ret = pd.DataFrame({"A": [0.0, np.nan, np.nan], "B": [0.0, 0.0, 0.02]}, index=i)
    res = pf.simulate(alpha, ret, const(100.0), const(1e9), const(0.0), n=1, costs=FREE)
    assert res["monthly"]["net"].tolist() == pytest.approx([0.0, 0.02])
    assert res["log"][i[1]]["sold"] == ["A"] and res["log"][i[1]]["bought"] == ["B"]


def test_blocked_and_penalty_reports():
    i = M("2020-01", 5)
    ret = pd.DataFrame({"x": [0, 0.3, 0.2, 0.1, 0], "y": [0, 0.0, 0.0, 0.0, 0.0]}, index=i)
    log = {i[0]: {"bought": [], "sold": [], "blocked": ["x"], "kept": ["y"]}}
    rep = pf.blocked_report(log, ret)
    assert rep["blocked_n"] == 1 and rep["blocked_next3m_%"] == pytest.approx((1.3 * 1.2 * 1.1 - 1) * 100)
    assert rep["blocked_rockets"] == 1 and rep["kept_rockets"] == 0 and rep["kept_next3m_%"] == 0
    rows = [[0.05, 0.04], [0.04, 0.0405], [0.04, 0.0405]]
    j, alpha, const = _sim_inputs(["A", "B"], 8, rows)
    r2 = pd.DataFrame(0.0, index=j, columns=["A", "B"])
    args = (alpha, r2, const(100.0), const(1e9), const(0.0))
    pen = pf.penalty_report(pf.simulate(*args, n=1, costs=PLAIN), pf.simulate(*args, n=1, gamma=0, costs=PLAIN), r2)
    assert pen["cost_saved_%/mo"] > 0 and pen["turnover_with_%/mo"] < pen["turnover_without_%/mo"]
    assert pen["blocked_n"] == 2


# ======================================================================== walkforward
def test_known_mean_respects_lag_min_months_and_window():
    i = M("2020-01", 10)
    x = pd.Series(np.arange(1.0, 11.0), index=i)
    k = wf.known_mean(x, lag=2, index=i, min_months=3)
    assert np.isnan(k.iloc[3]) and k.iloc[4] == pytest.approx(2.0) and k.iloc[9] == pytest.approx(4.5)
    kw = wf.known_mean(x, lag=2, index=i, min_months=2, window=2)
    assert kw.iloc[5] == pytest.approx(3.5)


def test_cum_ic_planted_and_no_look_ahead():
    r = np.random.default_rng(11)
    i, c = M("2012-01", 72), ids(50)
    s = pd.DataFrame(r.normal(size=(72, 50)), index=i, columns=c)
    ret = (s.shift(1) * 0.01).fillna(0) + r.normal(0, 0.0001, s.shape)          # predicts next month only
    ci = wf.cum_ic(s, ret, horizons=3, min_months=24)
    assert ci[1].iloc[:24].isna().all() and ci[1].iloc[30] > 0.95
    assert abs(ci[2].iloc[40]) < 0.1 and ci["cum"].iloc[40] == pytest.approx(ci[[1, 2, 3]].iloc[40].sum())
    later = ret.copy()
    later.iloc[50:] = r.normal(0, 0.05, later.iloc[50:].shape)
    pd.testing.assert_frame_equal(wf.cum_ic(s, later, horizons=3, min_months=24).iloc[:50], ci.iloc[:50])
    roll = wf.cum_ic(s, later, horizons=1, min_months=12, window=12)
    assert roll[1].iloc[45] > 0.95 and roll[1].iloc[-1] < 0.3                   # the rolling estimate forgets


def test_expected_return_scale():
    i, c = M("2020-01", 1), ids(40)
    score = pd.DataFrame([np.arange(40.0)], index=i, columns=c)
    vol = pd.DataFrame(0.10, index=i, columns=c)
    a = wf.expected_return(score, pd.Series(0.05, index=i), vol).iloc[0]
    assert a.mean() == pytest.approx(0, abs=1e-12) and a.is_monotonic_increasing
    assert a.std() == pytest.approx(0.005, rel=0.08) and a.iloc[-1] == pytest.approx(0.005 * 2.24, rel=0.01)


def test_cut_decisions_weak_and_unstable_only_and_only_past_data():
    r = np.random.default_rng(12)
    i = M("2010-01", 108)
    good = pd.Series(0.01 + r.normal(0, 0.01, 108), index=i)
    bad = pd.Series(np.r_[np.full(30, 0.01), np.full(30, -0.01), np.full(48, 0.05)] + r.normal(0, 0.002, 108), index=i)
    weak_stable = pd.Series(0.0005 + r.normal(0, 0.0001, 108), index=i)
    keep = wf.cut_decisions(pd.DataFrame({"good": good, "bad": bad, "weak_stable": weak_stable}))
    assert keep["good"].all() and keep["weak_stable"].all()
    assert keep.loc[2014, "bad"]                                                # only 48 months: no decision yet
    assert not keep.loc[2015, "bad"]                                            # 60 months: t < 1 and 2nd half < 0
    assert keep.loc[2019, "bad"]                                                # later strength is seen only later
    assert list(keep.index) == list(range(2011, 2020))


def test_wf_composite_applies_decision_from_january():
    c = ids(4)
    i = pd.period_range("2014-11", "2015-02", freq="M")
    a = pd.DataFrame([[1, 2, 3, 4]] * 4, index=i, columns=c, dtype=float)
    b = pd.DataFrame([[4, 3, 2, 1]] * 4, index=i, columns=c, dtype=float)
    keep = pd.DataFrame({"a": [True, True], "b": [True, False]}, index=[2014, 2015])
    comp = wf.wf_composite({"a": a, "b": b}, keep)
    assert comp.loc[pd.Period("2014-11", "M")].tolist() == pytest.approx([0.5] * 4)        # held in Dec 2014
    assert comp.loc[pd.Period("2014-12", "M")].tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])
    none = pd.DataFrame({"a": [False], "b": [False]}, index=[2015])
    assert wf.wf_composite({"a": a, "b": b}, none).loc[pd.Period("2015-01", "M")].tolist() == pytest.approx([0.5] * 4)


def test_holdout_guard():
    assert wf.guard_holdout("2023-12") == pd.Period("2023-12", "M")
    with pytest.raises(ValueError):
        wf.guard_holdout("2024-01")
    assert wf.guard_holdout("2026-10", open_holdout=True) == pd.Period("2026-10", "M")
