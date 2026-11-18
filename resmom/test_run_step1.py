"""Tests of the step-1 batch (run_step1.py) on hand-made and simulated data.

The simulated market (fake_data.py) has a planted effect whose timing is written from
the raw publication rule, so these tests check the whole chain: holdout cut -> signals
-> composite -> alpha -> portfolio, with nothing from the real data.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import data_tw as dt
import evaluate as ev
import fake_data as fd
import resmom as rm
import run_step1 as rs
import signals as sg

M = lambda s, n: pd.period_range(s, periods=n, freq="M")
P = lambda s: pd.Period(s, "M")


# ------------------------------------------------------------------ evaluate: fast paths
def test_rank_matches_pandas_average_ranks():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, 50).astype(float)                    # many ties
    assert np.allclose(ev._rank(x), pd.Series(x).rank().to_numpy())
    assert ev._spearman(np.array([1.0]), np.array([2.0])) != ev._spearman(np.array([1.0]), np.array([2.0]))   # nan


def test_ls_returns_equals_the_original_portfolio_routine():
    rng = np.random.default_rng(1)
    i, cols = M("2015-01", 30), [f"s{k:02d}" for k in range(60)]
    sig = pd.DataFrame(rng.normal(size=(30, 60)).round(1), index=i, columns=cols)       # rounded: ties
    ret = pd.DataFrame(rng.normal(0, 0.1, (30, 60)), index=i, columns=cols)
    sig[rng.random(sig.shape) < 0.2] = np.nan
    ret[rng.random(ret.shape) < 0.1] = np.nan
    sig.iloc[11] = np.nan                                       # a month without a portfolio breaks the turnover chain
    short = ret.fillna(0.0)
    uni = pd.DataFrame(rng.random(sig.shape) < 0.9, index=i, columns=cols)
    for u, rs_ in ((None, None), (uni, None), (uni, short)):
        got = ev.ls_returns(sig, ret, u, pct=0.2, min_stocks=20, ret_short=rs_)
        s = (sig if u is None else sig.where(u)).copy()
        s.index = s.index + 1
        ref = rm.wml(s, ret, hold=(1,), pct=0.2, min_stocks=20).rename(columns=lambda c: c[:-1])
        if rs_ is not None:
            q = rm.wml(s, rs_, hold=(1,), pct=0.2, min_stocks=20).rename(columns=lambda c: c[:-1])
            ref["L"], ref["TO_L"] = q["L"], q["TO_L"]
            ref["WML"] = ref["W"] - ref["L"]
        pd.testing.assert_frame_equal(got, ref[got.columns], check_exact=False, rtol=1e-12, check_freq=False)
        assert got["TO_W"].isna().sum() == 2                    # first month, and the month after the gap


def test_group_returns_equals_decile_groupby():
    rng = np.random.default_rng(2)
    i, cols = M("2020-01", 6), list("abcdefghijklmnopqrst")
    x = pd.DataFrame(rng.integers(0, 5, (6, 20)).astype(float), index=i, columns=cols)
    f = pd.DataFrame(rng.normal(size=(6, 20)), index=i, columns=cols)
    x.iloc[0, :3] = np.nan
    f.iloc[1, 5] = np.nan
    sub = pd.DataFrame(rng.random((6, 20)) < 0.8, index=i, columns=cols)
    got = ev.group_returns(x, f, sub, n_groups=3, min_n=5)
    for m in i:
        ok = x.loc[m].notna() & f.loc[m].notna() & sub.loc[m]
        ref = f.loc[m][ok].groupby(rm.decile(x.loc[m][ok], 3)).mean()
        assert np.allclose(got.loc[m].to_numpy(), ref.reindex([1, 2, 3]).to_numpy(), equal_nan=True)
    assert ev.group_returns(x, f, sub, 3, min_n=99).empty


def test_two_by_two_hand_numbers():
    i, cols = M("2020-01", 1), list("abcdefgh")
    a = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8.0]], index=i, columns=cols)       # high a: e f g h
    b = pd.DataFrame([[1, 8, 2, 7, 3, 6, 4, 5.0]], index=i, columns=cols)       # high b: b d f h
    f = pd.DataFrame([[.01, .02, .03, .04, .05, .06, .07, .08]], index=i, columns=cols)
    r = ev.two_by_two(a, b, f, min_n=8)
    assert r["a_hi_b_hi_%/mo"] == pytest.approx(7.0) and r["a_lo_b_hi_%/mo"] == pytest.approx(3.0)      # f,h | b,d
    assert r["a_hi_b_lo_%/mo"] == pytest.approx(6.0) and r["a_lo_b_lo_%/mo"] == pytest.approx(2.0)      # e,g | a,c
    assert r["difference_%/mo"] == pytest.approx(0.0) and r["ic_in_b_hi"] == pytest.approx(1.0)


def test_cells_by_sign_cuts_at_zero():
    i, cols = M("2020-01", 3), [f"s{k}" for k in range(16)]
    a = pd.DataFrame([[-1.0] * 8 + [1.0] * 8] * 3, index=i, columns=cols)
    b = pd.DataFrame([([-1.0] * 4 + [1.0] * 4) * 2] * 3, index=i, columns=cols)
    f = pd.DataFrame([[0.01] * 4 + [0.03] * 4 + [0.0] * 4 + [0.02] * 4] * 3, index=i, columns=cols)
    r = ev.cells_by_sign(a, b, f)
    assert r["a_neg_b_neg_%/mo"] == pytest.approx(1.0) and r["a_neg_b_pos_%/mo"] == pytest.approx(3.0)
    assert r["a_pos_b_neg_%/mo"] == pytest.approx(0.0) and r["a_pos_b_pos_n"] == 4
    assert r["seasonal_minus_real_%/mo"] == pytest.approx(2.0)
    sub = pd.DataFrame(False, index=i, columns=cols)
    assert ev.cells_by_sign(a, b, f, sub) == {}


def test_flag_report_and_state_flag():
    d = pd.bdate_range("2020-01-01", periods=400)
    mkt = pd.Series(0.001, index=d)
    flag = pd.Series(0.0, index=d)
    flag.iloc[100:110] = 1.0
    flag.iloc[200:205] = 1.0
    mkt.iloc[101:131] = -0.002                                  # falls right after the first flagged stretch starts
    r = ev.flag_report(flag, mkt, horizon=20, delay=1)
    assert r["bad_days"] == 15 and r["episodes"] == 2 and r["fwd_%_bad"] < r["fwd_%_other"]
    x = pd.Series(np.arange(400.0), index=d)                    # always a new high: never in the worst tail
    assert ev.state_flag(x, 0.1, True, 250).dropna().sum() == 0
    assert ev.state_flag(-x, 0.1, True, 250).dropna().mean() == 1 and ev.state_flag(x, 0.1, True, 250).iloc[:100].isna().all()


# ----------------------------------------------------------------------- run helpers
def test_vw_daily_uses_previous_month_end_caps_and_drifts():
    d = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    dret = pd.DataFrame({"a": [0.0, 0.0, 0.10, 0.0], "b": [0.0, 0.0, 0.0, 0.10], "c": [np.nan] * 4}, index=d)
    me = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 5.0], "c": [9.0, 9.0]}, index=M("2020-01", 2))
    r = rs.vw_daily(dret, me)
    assert r.iloc[:2].isna().all()                              # no cap before the first month
    assert r.iloc[2] == pytest.approx(0.1 * 1 / 4)              # c never trades: left out
    assert r.iloc[3] == pytest.approx(4.4 / 4.1 - 1)            # weights have drifted to 1.1 : 3
    assert rs.to_monthly(r).loc[P("2020-02")] == pytest.approx(4.4 / 4 - 1)


def test_entry_dates_follow_the_deadline_and_holidays():
    d = pd.bdate_range("2021-01-01", "2022-03-31")
    d = d[~((d >= "2021-02-10") & (d <= "2021-02-16"))]         # a long holiday over the 10th
    e = rs.entry_dates(d)
    assert e[P("2022-01")] == pd.Timestamp("2022-01-11")        # the 10th is a Monday
    assert e[P("2021-04")] == pd.Timestamp("2021-04-13")        # the 10th is a Saturday -> Monday the 12th -> next day
    assert e[P("2021-02")] == pd.Timestamp("2021-02-18")        # market reopens on the 17th


def test_returns_between_pairs_like_monthly_returns():
    d = pd.bdate_range("2021-01-01", "2021-04-30")
    dret = pd.DataFrame({"a": 0.01, "b": 0.01}, index=d)
    dret.loc[d > "2021-02-20", "b"] = np.nan
    dates = pd.Series({P("2021-01"): pd.Timestamp("2021-01-12"), P("2021-02"): pd.Timestamp("2021-02-11"),
                       P("2021-03"): pd.Timestamp("2021-03-11")})
    r = rs.returns_between(dret, dates)
    k = len(d[(d > "2021-01-12") & (d <= "2021-02-11")])
    assert list(r.index) == [P("2021-02"), P("2021-03")]
    assert r.loc[P("2021-02"), "a"] == pytest.approx(1.01 ** k - 1)
    assert np.isnan(r.loc[P("2021-03"), "b"])                   # fewer than 10 returns in the window
    me = rs.returns_between(dret, sg.month_ends(d))             # month-end dates reproduce monthly returns
    assert me.loc[P("2021-02"), "a"] == pytest.approx(dt.monthly_from_daily(dret).loc[P("2021-02"), "a"])


def test_flat_costs():
    i = M("2020-01", 2)
    px = pd.DataFrame({"a": [20.0, 20.0], "b": [200.0, 200.0], "c": [5.0, 5.0]}, index=i)
    u = pd.DataFrame({"a": True, "b": True, "c": False}, index=i)
    buy, sell, spread = rs.flat_costs(px, u, rs.Cfg())
    assert spread == pytest.approx(0.00125)                     # half of 0.05 / 20 and half of 0.5 / 200
    assert buy == pytest.approx(0.001425 * 0.2 + 0.00125) and sell == pytest.approx(buy + 0.003)
    assert rs.flat_costs(px, u, replace(rs.Cfg(), spread_ticks=0.0, fee_discount=1.0))[0] == pytest.approx(0.001425)


def test_basket_and_extreme():
    i, cols = M("2020-01", 3), list("abcdefghij")
    ret = pd.DataFrame(np.tile(np.arange(10) / 100, (3, 1)), index=i, columns=cols)
    sig = pd.DataFrame(np.tile(np.arange(10.0), (3, 1)), index=i, columns=cols)
    uni = pd.DataFrame(True, index=i, columns=cols)
    uni["j"] = False
    top = rs.extreme(sig, uni, 0.2, top=True)
    assert list(top.columns[top.iloc[0]]) == ["h", "i"] and list(rs.extreme(sig, uni, 0.2, False).columns[top.iloc[0] * 0 == 0][:0]) == []
    assert list(rs.extreme(sig, uni, 0.2, top=False).iloc[0][lambda s: s].index) == ["a", "b"]
    b = rs.basket(top, ret)
    assert list(b.index) == [P("2020-02"), P("2020-03")] and b.iloc[0] == pytest.approx(0.075)


def test_market_adjusted_uses_only_earlier_months():
    rng = np.random.default_rng(3)
    i = M("2010-01", 80)
    m = pd.Series(rng.normal(0.01, 0.05, 80), index=i)
    x = 0.5 * m + 0.002
    adj = rs.market_adjusted(x, m, min_months=36)
    assert adj.index[0] == i[36] and np.allclose(adj, 0.002 - 0.5 * 0 + (0.5 - 0.5) * m.reindex(adj.index), atol=1e-12)
    x2, m2 = x.copy(), m.copy()
    x2.iloc[60:], m2.iloc[60:] = 9.0, -9.0                      # the future changes ...
    assert np.allclose(rs.market_adjusted(x2, m2, 36).iloc[:24], adj.iloc[:24])     # ... the past does not


def test_goal_needs_both_halves():
    i = M("2010-01", 48)
    mkt = pd.Series(0.01, index=i)
    mkt.iloc[[5, 30]] = -0.10
    good, lev = mkt + 0.006, mkt * 1.6
    g, l = rs.goal(good, mkt), rs.goal(lev, mkt)
    assert g["goal_met"] and g["ret_x_mkt_h1"] > 1.3 and g["dd_x_mkt_h2"] < 1.0
    assert l["ret_x_mkt_full"] > 1.3 and l["dd_x_mkt_full"] > 1.1 and not l["goal_met"]
    late = good.copy()
    late.iloc[:24] = mkt.iloc[:24]                              # only the second half is better
    assert not rs.goal(late, mkt)["goal_met"]


def test_trend_flag_and_month_flag_report():
    i = M("2010-01", 30)
    mkt = pd.Series([0.02] * 12 + [-0.05] * 4 + [0.03] * 14, index=i)
    f = rs.trend_flag(mkt, 10)
    assert f.iloc[:9].isna().all() and f.loc[P("2010-12")] == 0 and f.loc[P("2011-02")] == 1
    r = rs.month_flag_report(f, mkt)
    assert r["episodes"] == 1 and r["bad_months"] == int(f.shift(1).sum())
    y = pd.Series(np.where(f.shift(1) == 1, 0.05, 0.01), index=i)
    assert rs.month_flag_report(f, y)["next_%_bad"] == pytest.approx(5.0)


def test_clean_industry():
    lab = pd.Series(["金融業", "金融保險", "其他電子類", "創新板股票", "其他", "半導體業"])
    assert dt.clean_industry(lab).tolist()[:3] == ["金融保險", "金融保險", "其他電子業"]
    assert dt.clean_industry(lab).isna().tolist() == [False, False, False, True, True, False]


# ------------------------------------------------------------------ prepare / holdout
@pytest.fixture(scope="module")
def raw_small():
    return fd.fake_raw(seed=5, n=60, start="2015-01", end="2025-06")


def _walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        yield path, x


def test_prepare_cuts_everything_at_the_end(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    end, seen = P("2023-12"), 0
    for path, x in _walk(D):
        i = x.index
        if len(i) == 0:
            continue
        if isinstance(i, pd.DatetimeIndex):
            assert i.max() <= end.end_time, path
        elif isinstance(i, pd.PeriodIndex) and i.freqstr.startswith("Q"):
            assert all(sg.usable_month(q) <= end for q in i), path
        elif isinstance(i, pd.PeriodIndex):
            assert i.max() <= end, path
        else:
            continue
        seen += 1
    assert seen > 30 and D["rev"].index.max() == P("2023-11")     # November revenue is the last one published by December
    assert D["months"][-1] == end and D["dret"].index.max().year == 2023


def test_holdout_needs_the_flag(raw_small):
    with pytest.raises(ValueError, match="holdout"):
        rs.prepare(raw_small, rs.Cfg(end="2024-06"))
    with pytest.raises(ValueError, match="holdout"):
        rs.run(raw_small, rs.Cfg(end="2024-01"), "coverage")
    D = rs.prepare(raw_small, rs.Cfg(end="2024-06", open_holdout=True))
    assert D["months"][-1] == P("2024-06")


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, obj=path)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, obj=path)
    elif isinstance(a, pd.Index):
        assert a.equals(b), path
    elif isinstance(a, (list, tuple)):
        assert list(a) == list(b), path
    else:
        assert a == b, path


def test_data_after_the_cut_cannot_change_anything(raw_small):
    """Everything the tables are computed from (D) is identical when every later number
    is replaced by garbage; run() uses nothing but D after prepare()."""
    junk = fd.scramble_after(raw_small, "2023-12")
    assert not junk["dret"].loc["2024":].equals(raw_small["dret"].loc["2024":])
    assert not junk["rev"].loc[P("2023-12")].equals(raw_small["rev"].loc[P("2023-12")])
    _same(rs.prepare(raw_small, rs.Cfg()), rs.prepare(junk, rs.Cfg()))


def test_derive_changes_only_what_depends_on_the_setting(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    merged = D["merged"]                                        # delisted by a merger: paid out
    unknown = D["gone"] & D["gap_ret"].isna() & ~merged         # stopped trading and never seen again
    known = D["gone"] & D["gap_ret"].notna()
    assert unknown.sum().sum() > 0 and known.sum().sum() == 2 and merged.sum().sum() > 0
    assert not (merged & ~D["gone"]).any().any() and not (merged & known).any().any()
    D2 = rs.derive(D, replace(D["cfg"], delist_long=-1.0, adv_floor=1e9))
    diff = (D2["ret"] != D["ret"]) & ~(D2["ret"].isna() & D["ret"].isna())
    at = lambda x, where=unknown: x.to_numpy()[where.to_numpy()]
    assert diff.equals(unknown) and (at(D2["ret"]) == -1.0).all()
    assert (at(D["ret"]) == -0.30).all() and (at(D["ret_short"]) == 0.0).all() and np.isnan(at(D["ret_raw"])).all()
    assert np.array_equal(at(D["ret"], known), at(D["gap_ret"], known))             # the actual outcome, for both sides
    assert np.array_equal(at(D["ret_short"], known), at(D["gap_ret"], known))
    assert (at(D["ret"], merged) == 0.0).all() and (at(D["ret_short"], merged) == 0.0).all()   # paid the last price
    assert (at(D2["ret"], merged) == 0.0).all()                                               # not a delisting loss
    D3 = rs.derive(D, replace(D["cfg"], merger_exit=-0.30))
    diff3 = (D3["ret"] != D["ret"]) & ~(D3["ret"].isna() & D["ret"].isna())
    assert diff3.equals(merged) and (at(D3["ret_short"], merged) == -0.30).all()
    assert D2["U"]["tradable"].sum().sum() < D["U"]["tradable"].sum().sum() and D["cfg"].adv_floor == 3e7
    assert D2["dret"] is D["dret"]


def test_the_market_is_the_official_index_when_there_is_one(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    level = raw_small["mkt_level"]
    m_end = level.groupby(level.index.to_period("M")).last()
    assert D["mkt"].loc[P("2023-06")] == pytest.approx(m_end[P("2023-06")] / m_end[P("2023-05")] - 1)
    day = D["dret"].index[-5]
    assert D["mkt_d"].loc[day] == pytest.approx(level.loc[day] / level.loc[:day].iloc[-2] - 1)
    assert D["mkt_x"].equals(D["mkt"] - D["rf"]) and D["mkt"].corr(D["mkt_own"]) > 0.8
    assert not D["mkt"].dropna().equals(D["mkt_own"].dropna())
    D0 = rs.prepare({**raw_small, "mkt_level": None}, rs.Cfg())                 # none given: the self-built one
    assert D0["mkt"].equals(D0["mkt_own"]) and D0["mkt_own"].equals(D["mkt_own"])
    b = rs.t_benchmark(D)
    assert b.loc["market_in_use", "tracking_error_%/yr"] == pytest.approx(0) and b.loc["self_built_all_stocks", "corr_monthly"] > 0.8


def test_a_missing_factor_counts_as_the_middle_and_fast_signals_add_only_their_own_part(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    S, X, _ = rs.build_signals(D)
    P_, _ = rs.ports_and_ics(D, S, rs.SLOW, [rs.PRIMARY])
    cp = rs.composites(D, S, P_)
    slow = [S[k] for k in rs.SLOW]
    n = sum(s.notna() for s in slow)
    c = cp["C"]["slow_all"]
    assert c.notna().equals(n >= 1) and (n[c.notna()] < len(slow)).any().any()     # every stock with one factor has a score
    few, full = (n == 1) & c.notna(), n == len(slow)
    assert (c[few] - 0.5).abs().max().max() <= 0.5 / len(slow) + 1e-12             # one factor cannot move it far
    assert (c[full] - 0.5).abs().max().max() > 2 * 0.5 / len(slow)
    old = rs.composites(rs.derive(D, replace(D["cfg"], missing_rank=None)), S, P_)["C"]["slow_all"]
    assert old.notna().equals(n >= 3) and np.allclose(old[full], c[full], equal_nan=True)
    assert cp["fast"] == [k for k in rs.FAST if k in S] and "slow_fast" in cp["C"]  # no half-life rule
    base = rs.pipeline(D, S, X, P_)
    assert {"slow_wf", "slow_all", "slow_wf+fast", "slow_wf+fast_raw"} == set(base["alpha"])
    assert not base["alpha"]["slow_wf+fast"].equals(base["alpha"]["slow_wf+fast_raw"])


def test_a_merger_exit_needs_a_delisting_dated_inside_the_sample(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    ex, mg = D["exits"], raw_small["mergers"]
    assert ex["merger"].sum() == D["merged"].sum().sum() > 0
    assert set(ex.loc[ex["merger"], "stock"]) <= set(mg["stock"]) and ex.loc[ex["merger"], "next_trade"].isna().all()
    # cut the sample just before one merger's delisting date: that exit is no longer in the data,
    # and no delisting dated after the cut is used
    first = mg.sort_values("delisted").iloc[0]
    cut = str(first["delisted"].to_period("M") - 2)
    Dc = rs.prepare(raw_small, rs.Cfg(end=cut))
    assert not Dc["merged"].any().any()
    # without the list every one of them is an unknown outcome again
    D0 = rs.prepare({**raw_small, "mergers": None}, rs.Cfg())
    assert not D0["merged"].any().any()
    w = D["merged"].to_numpy()
    assert (D0["ret"].to_numpy()[w] == -0.30).all() and (D["ret"].to_numpy()[w] == 0.0).all()


def test_a_stock_that_comes_back_gets_its_actual_gap_return(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    for s, h in raw_small["halts"].items():
        days = raw_small["close"].index
        m = days[days < h["stops"]][-1].to_period("M") + 1        # first whole month without trading
        # the capital reduction (price x 4, shares / 4) cannot be told from a gain: conservative fill
        lo, sh = (-0.30, 0.0) if h["jump"] > 1 else (h["jump"], h["jump"])
        assert bool(D["gone"].loc[m, s]) and D["ret"].loc[m, s] == pytest.approx(lo, abs=1e-9)
        assert D["ret_short"].loc[m, s] == pytest.approx(sh, abs=1e-9)
        back = h["resumes"].to_period("M")
        assert np.isnan(D["ret"].loc[m + 1:back, s]).all()        # the jump is not counted a second time
        assert not D["listed"].loc[m, s] and not D["no_sell"].loc[m, s] and D["no_buy"].loc[m, s]
    ex = D["exits"]
    assert len(ex) == int(D["gone"].sum().sum()) and ex["used"].notna().sum() == 2


def test_a_delisted_holding_can_be_closed_but_a_halted_one_cannot(raw_small):
    D = rs.prepare(raw_small, rs.Cfg())
    g = D["gone"].to_numpy()                                     # first month without trading: no longer listed
    assert not D["no_sell"].to_numpy()[g].any() and not D["no_cover"].to_numpy()[g].any()
    assert D["no_buy"].to_numpy()[g].all() and D["no_short"].to_numpy()[g].all()
    before = D["gone"].shift(-1, fill_value=False).to_numpy()    # last month with trading, still listed
    last_day = sg.at_month_end(D["close"].isna()).reindex(index=D["months"], columns=D["cols"]).to_numpy()
    stopped_early = before & last_day & D["listed"].to_numpy()
    assert stopped_early.any() and D["no_sell"].to_numpy()[stopped_early].all()
    # in the simulated book: the delisting loss is taken once and the position leaves the book
    s = (D["gone"] & D["gap_ret"].isna()).any()[lambda x: x].index[0]
    m = D["gone"][s].idxmax()
    alpha = pd.DataFrame(0.0, index=D["months"], columns=D["cols"]).where(D["listed"])
    alpha.loc[: m - 1, s] = 9.0                                  # held until it stops trading
    res = rs.pf.simulate(alpha, D["ret"], D["close_m"], D["adv"], D["dvol"], no_buy=D["no_buy"], no_sell=D["no_sell"],
                         capital=1e6, n=5, start=str(m - 3), end=str(m + 2))
    log = res["log"]
    assert s in log[m - 4]["bought"] and s in log[m]["sold"] and all(s not in log[k]["sold"] for k in log if k < m)


# -------------------------------------------------------------------- signal timing
def test_planted_effect_shows_up_in_the_right_month_only():
    """Revenue of month r moves the return of month r+2. The signal row for formation
    month r+1 must line up with the NEXT month's return, not with its own month."""
    raw = fd.fake_raw(seed=7, n=150, start="2015-01", end="2023-12", plant_mom=0.03, season=0.0)
    D = rs.prepare(raw, rs.Cfg())
    S, X, status = rs.build_signals(D)
    ic = lambda shift: ev.rank_ic(X["rev_mom"], D["ret_raw"].shift(-shift), D["listed"]).mean()
    assert ic(1) > 0.15 and abs(ic(0)) < 0.03 and abs(ic(2)) < 0.03
    for k, v in S.items():                                       # every signal is restricted to listed stocks
        assert not (v.notna() & ~D["listed"]).any().any(), k
    assert status["built"].drop("fcf_ev").all() and set(rs.SLOW + rs.FAST) <= set(S)


# ------------------------------------------------------------------ the whole batch
@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    raw = fd.fake_raw(seed=1, n=140, start="2010-01", end="2024-06", plant_mom=0.02, plant_yoy=0.02)
    out = tmp_path_factory.mktemp("step1")
    return rs.run(raw, rs.Cfg(), "all", out, log=lambda *_: None), out


def test_every_table_is_produced_and_written(batch):
    T, out = batch
    expected = {"status", "costs_flat", "coverage", "exit_months", "single_factor", "correlation_all", "correlation_big",
                "correlation_tradable", "year_2008", "ic_by_month", "single_factor_sensitivity", "reversal_A_baseline",
                "reversal_BE_conditions", "reversal_B_cells", "revenue_timing", "price_first", "value_trap_cells",
                "value_trap_screen", "decay", "cut_decisions", "composite", "alpha_ic_used", "portfolio", "penalty",
                "portfolio_sensitivity", "state_daily", "state_monthly", "benchmark", "turnover_by_month",
                "single_factor_official_industry"}
    assert set(T) == expected
    # the simulated market starts in 2010 and has too few losers a month for the four-cell table
    empty = [k for k, v in T.items() if v.empty and k not in ("year_2008", "reversal_B_cells")]
    assert not empty
    assert all((out / f"{k}.csv").exists() for k in T) and (out / "config.json").exists()


def test_planted_effect_reaches_factor_composite_and_portfolio(batch):
    T, _ = batch
    sf = T["single_factor"]
    rev = sf.loc[("revenue", "all", "full")]
    assert rev["ic_mean_ic"] > 0.05 and rev["gross_t"] > 4 and bool(rev["pass"])
    assert abs(sf.loc[("flows", "all", "full"), "gross_t"]) < 3.5              # nothing planted there
    assert T["composite"].loc[("slow_all", "all", "full"), "gross_t"] > 3
    lo = T["portfolio"].loc[("slow_wf", 1e6, "long_only")]
    assert lo["alpha_t"] > 2 and lo["names"] <= 30 and 0 < lo["turnover_%/mo"] < 60
    assert T["alpha_ic_used"][("slow_wf", "composite")].dropna().gt(0).all()
    assert "monthly_ok" not in T["decay"].columns and "half_life_days" in T["decay"].columns
    tm = T["turnover_by_month"]
    assert list(tm.columns) == [f"m{m:02d}" for m in range(1, 13)] and ("slow_wf+fast", 1e6) in tm.index
    assert tm.loc[("slow_wf", 1e6)].mean() == pytest.approx(lo["turnover_%/mo"], rel=0.15)


def test_more_capital_means_more_names(batch):
    p = T = batch[0]["portfolio"]
    names = lambda cap: p.loc[("slow_wf", cap, "long_only"), "names"]
    assert names(1e9) > names(1e6)
    assert {"long_only", "short_book", "long_short"} <= set(p.index.get_level_values("book"))
    pen = batch[0]["penalty"]
    assert (pen["turnover_with_%/mo"] <= pen["turnover_without_%/mo"]).all()


def test_revenue_timing_arms_and_sensitivity_rows(batch):
    T, _ = batch
    rt = T["revenue_timing"]
    assert rt.loc[("all", "month_end"), "entry_day_of_month"] > 27
    assert 10 < rt.loc[("all", "after_deadline"), "entry_day_of_month"] < 15
    sens = T["portfolio_sensitivity"]
    settings = {s for s, *_ in rs.SENSITIVITY}
    assert settings <= set(sens.index.get_level_values("setting"))
    g = sens.xs("gamma", level="setting")
    assert g["turnover_%/mo"].iloc[-1] <= g["turnover_%/mo"].iloc[1]           # gamma 2.0 trades less than 0.5
