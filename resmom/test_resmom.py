"""Unit tests on small simulated data. Run: python -m pytest test_resmom.py -q"""
import numpy as np
import pandas as pd
import pytest

import resmom as rm


def simulate(n_stocks=60, n_months=72, seed=0):
    """Wide returns and factors: ret = rf + beta * factors + noise."""
    g = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")
    fac = pd.DataFrame(g.normal(0, 0.04, (n_months, 3)), index=idx, columns=rm.FACTORS)
    fac["RF"] = 0.002
    beta = g.normal([1, 0.3, 0.2], 0.3, (n_stocks, 3))
    ret = fac[rm.FACTORS].to_numpy() @ beta.T + g.normal(0, 0.06, (n_months, n_stocks)) + 0.002
    return pd.DataFrame(ret, index=idx, columns=[f"S{i:03d}" for i in range(n_stocks)]), fac


# ------------------------------------------------------------------------ clipping
def test_clip_returns():
    s = pd.Series([-0.99, -0.95, 0.0, 1.0, 3.5, np.nan])
    out = rm.clip_returns(s)
    assert out.tolist()[:5] == [-0.95, -0.95, 0.0, 1.0, 1.0] and np.isnan(out.iloc[5])


def test_to_wide_clips_returns_and_fills_month_gaps():
    long = pd.DataFrame({"id": ["A", "A", "A"], "month": ["2020-01", "2020-02", "2020-04"],
                         "ret": [2.5, -0.99, 0.1], "me_lag": [5.0, 6.0, 7.0]})
    w = rm.to_wide(long)
    assert [str(m) for m in w.index] == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert w["A"].iloc[0] == 1.0 and w["A"].iloc[1] == -0.95 and np.isnan(w["A"].iloc[2])
    assert rm.to_wide(long, "me_lag")["A"].iloc[0] == 5.0       # only returns are clipped


# ------------------------------------------------------------------------ grouping
def test_decile_groups_are_equal_sized_and_ordered():
    s = pd.Series(np.arange(100.0), index=[f"S{i}" for i in range(100)]).sample(frac=1, random_state=1)
    d = rm.decile(s)
    assert d.value_counts().tolist() == [10] * 10
    assert d["S0"] == 1 and d["S9"] == 1 and d["S10"] == 2 and d["S99"] == 10


def test_winners_losers_top_and_bottom_ten_percent():
    s = pd.Series(np.arange(51.0), index=[f"S{i}" for i in range(51)])
    s["S7"] = np.nan                                           # no score -> never selected
    w, l = rm.winners_losers(s)
    assert set(w) == {"S46", "S47", "S48", "S49", "S50"}       # 50 scored stocks -> 5 each
    assert set(l) == {"S0", "S1", "S2", "S3", "S4"}       # S7 has no score but is not in the bottom 5 anyway
    d = rm.decile(s)
    assert (d[w] == 10).all() and (d[l] == 1).all()


def test_thirty_percent_cutoff():
    s = pd.Series(np.arange(100.0), index=[f"S{i}" for i in range(100)])
    w, l = rm.winners_losers(s, pct=0.30)
    assert set(w) == {f"S{i}" for i in range(70, 100)} and set(l) == {f"S{i}" for i in range(30)}


def test_fewer_than_30_stocks_builds_no_portfolio():
    assert rm.winners_losers(pd.Series(np.arange(29.0))) is None
    assert rm.winners_losers(pd.Series(np.arange(30.0))) is not None


# ---------------------------------------------------------------- overlapping hold
def hand_case():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    cols = list("ABCDEFGHIJ")
    g = np.random.default_rng(3)
    ret = pd.DataFrame(g.normal(0, 0.1, (6, 10)), index=idx, columns=cols)
    score = pd.DataFrame(np.nan, index=idx[:4], columns=cols)
    score.loc[idx[0]] = np.arange(10.0)                      # winner J, loser A
    score.loc[idx[1]] = np.arange(10.0)[::-1]                # winner A, loser J
    score.loc[idx[2]] = [5.0, 9, 0, 1, 2, 3, 4, 6, 7, 8]     # winner B, loser C
    score.loc[idx[3]] = np.arange(10.0)                      # winner J, loser A
    return idx, ret, score


def test_overlapping_holding_matches_hand_calculation():
    idx, ret, score = hand_case()
    out = rm.wml(score, ret, hold=(1, 3), min_stocks=1)

    # K=1: month t holds only the cohort labelled t
    assert out.loc[idx[1], "WML1"] == pytest.approx(ret.loc[idx[1], "A"] - ret.loc[idx[1], "J"])
    # K=3: first complete month is the third; equal thirds of the three live cohorts
    assert np.isnan(out.loc[idx[1], "WML3"]) and not np.isnan(out.loc[idx[2], "WML3"])
    m = idx[2]
    w = (ret.loc[m, "J"] + ret.loc[m, "A"] + ret.loc[m, "B"]) / 3
    l = (ret.loc[m, "A"] + ret.loc[m, "J"] + ret.loc[m, "C"]) / 3
    assert out.loc[m, "W3"] == pytest.approx(w) and out.loc[m, "L3"] == pytest.approx(l)
    assert out.loc[m, "WML3"] == pytest.approx(w - l)
    m = idx[3]                                    # cohorts labelled idx[3], idx[2], idx[1]
    assert out.loc[m, "W3"] == pytest.approx((ret.loc[m, "J"] + ret.loc[m, "B"] + ret.loc[m, "A"]) / 3)
    # after the last cohort there is no complete set of three any more
    assert idx[4] not in out.index and idx[5] not in out.index


def test_turnover_matches_hand_calculation():
    idx, ret, score = hand_case()
    out = rm.wml(score, ret, hold=(1, 3), min_stocks=1)
    # K=1: the single winner changes every month -> the whole leg is replaced
    assert out.loc[idx[1], "TO_W1"] == pytest.approx(1.0)
    assert np.isnan(out.loc[idx[0], "TO_W1"])                 # no previous portfolio
    # K=3, month 4: held J, A, B (1/3 each); target J, B, A again -> only drift is traded
    r = ret.loc[idx[2], ["J", "A", "B"]].to_numpy()
    drift = (1 + r) / (1 + r).sum()
    assert out.loc[idx[3], "TO_W3"] == pytest.approx(0.5 * np.abs(1 / 3 - drift).sum())
    assert out.loc[idx[3], "TO_W3"] < 0.2


def test_cohorts_are_equal_weighted_and_skip_delisted():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    cols = [f"S{i:02d}" for i in range(40)]
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    ret.loc[idx[1], ["S36", "S37", "S38", "S39"]] = [0.10, 0.20, np.nan, 0.30]   # S38 delisted
    score = pd.DataFrame([np.arange(40.0)], index=idx[1:], columns=cols)
    out = rm.wml(score, ret, hold=(1,))
    assert out.loc[idx[1], "W1"] == pytest.approx(0.20)       # mean of 0.10, 0.20, 0.30
    assert out.loc[idx[1], "L1"] == pytest.approx(0.0)


def test_value_weighting_uses_supplied_weights():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    cols = [f"S{i:02d}" for i in range(40)]
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    ret.loc[idx[1], ["S36", "S37", "S38", "S39"]] = [0.10, 0.20, 0.30, 0.40]
    cap = pd.DataFrame(1.0, index=idx, columns=cols)
    cap.loc[idx[1], ["S36", "S37", "S38", "S39"]] = [1.0, 1.0, 2.0, 6.0]
    score = pd.DataFrame([np.arange(40.0)], index=idx[1:], columns=cols)
    out = rm.wml(score, ret, hold=(1,), weight=cap)
    assert out.loc[idx[1], "W1"] == pytest.approx((0.10 + 0.20 + 2 * 0.30 + 6 * 0.40) / 10)


def test_net_of_cost():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    p = pd.DataFrame({"W3": [0.02, 0.03], "L3": [0.01, 0.00], "WML3": [0.01, 0.03],
                      "TO_W3": [np.nan, 0.4], "TO_L3": [np.nan, 0.5]}, index=idx)
    n = rm.net_of_cost(p, 3, buy=0.001, sell=0.004)
    assert len(n) == 1                                         # month without turnover dropped
    assert n["W"].iloc[0] == pytest.approx(0.03 - 0.4 * 0.005)
    assert n["WML"].iloc[0] == pytest.approx(0.03 - 0.9 * 0.005)


# -------------------------------------------------------------------------- scores
def test_scores_match_direct_regression_and_window():
    ret, fac = simulate()
    rs, ts = rm.scores(ret, fac)
    assert rs.index[0] == ret.index[36]                        # first label = month 37
    t, sid = rs.index[5], "S010"
    win = pd.period_range(t - 36, t - 1, freq="M")
    y = (ret.loc[win, sid] - fac.loc[win, "RF"]).to_numpy()
    X = np.column_stack([np.ones(36), fac.loc[win, rm.FACTORS].to_numpy()])
    e = pd.Series(y - X @ np.linalg.lstsq(X, y, rcond=None)[0], index=win)
    part = e.loc[t - 12:t - 2]
    assert len(part) == 11
    assert rs.loc[t, sid] == pytest.approx(part.sum() / part.std(ddof=1))
    assert ts.loc[t, sid] == pytest.approx((1 + ret.loc[t - 12:t - 2, sid]).prod() - 1)


def test_stocks_without_36_months_are_skipped():
    ret, fac = simulate()
    ret.loc[ret.index[10], "S000"] = np.nan                    # one hole
    ret.loc[ret.index[:50], "S001"] = np.nan                   # listed late
    rs, _ = rm.scores(ret, fac)
    assert rs.loc[ret.index[36]:ret.index[46], "S000"].isna().all()
    assert rs.loc[ret.index[47]:, "S000"].notna().all()        # hole has left the window
    assert rs["S001"].isna().all()                             # only 22 months of history


def test_month_with_too_few_stocks_gives_no_scores():
    ret, fac = simulate(n_stocks=29)
    rs, ts = rm.scores(ret, fac)
    assert rs.empty and ts.empty


def test_planted_residual_winner_ranks_first():
    ret, fac = simulate()
    t = ret.index[60]
    ret.loc[t - 12:t - 2, "S005"] += 0.15                      # steady unexplained gain
    rs, _ = rm.scores(ret, fac)
    assert rs.loc[t].idxmax() == "S005"


# ------------------------------------------------------------------- no look-ahead
def test_scores_ignore_holding_month_and_later():
    ret, fac = simulate()
    t = ret.index[50]
    r2, f2 = ret.copy(), fac.copy()
    r2.loc[t:] = r2.loc[t:] * -3 + 0.5
    f2.loc[t:, rm.FACTORS] = 9.0
    a, b = rm.scores(ret, fac), rm.scores(r2, f2)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.loc[:t], y.loc[:t])
    assert not a[0].loc[t + 1].equals(b[0].loc[t + 1])         # later rows do change


def test_total_momentum_skips_most_recent_month():
    ret, fac = simulate()
    t = ret.index[50]
    r2 = ret.copy()
    r2.loc[t - 1] += 0.2
    _, a = rm.scores(ret, fac)
    _, b = rm.scores(r2, fac)
    pd.testing.assert_series_equal(a.loc[t], b.loc[t])


def test_strategy_returns_unchanged_when_future_data_is_removed():
    ret, fac = simulate(n_months=90)
    cut = ret.index[70]
    cap = pd.DataFrame(np.random.default_rng(5).uniform(1, 9, ret.shape), index=ret.index, columns=ret.columns)

    def run(r, f, c):
        rs, ts = rm.scores(r, f)
        return rm.wml(rs, r), rm.wml(ts, r), rm.wml(rs, r, pct=0.3, weight=c)

    full, part = run(ret, fac, cap), run(ret.loc[:cut], fac.loc[:cut], cap.loc[:cut])
    for x, y in zip(full, part):
        pd.testing.assert_frame_equal(x.loc[:cut], y)


def test_portfolio_uses_only_scores_formed_before_the_month():
    ret, fac = simulate(n_months=60)
    rs, _ = rm.scores(ret, fac)
    m = ret.index[50]
    rs2 = rs.copy()
    rs2.loc[m + 1:] = -rs2.loc[m + 1:]                         # flip later signals
    a, b = rm.wml(rs, ret), rm.wml(rs2, ret)
    pd.testing.assert_frame_equal(a.loc[:m], b.loc[:m])


# ---------------------------------------------------------------------- statistics
def test_newey_west_t():
    x = np.array([0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, 0.01])
    n, u = len(x), x - x.mean()
    assert rm.nw_t(x, 0) == pytest.approx(x.mean() / np.sqrt(u @ u / n / n))
    v = u @ u / n + 2 * 0.5 * (u[1:] @ u[:-1]) / n             # Bartlett weight 1 - 1/2
    assert rm.nw_t(x, 1) == pytest.approx(x.mean() / np.sqrt(v / n))


def test_newey_west_lowers_t_for_persistent_series():
    g = np.random.default_rng(0)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.8 * x[i - 1] + g.normal()
    x += 0.5
    assert abs(rm.nw_t(x, 12)) < abs(rm.nw_t(x, 0))


def test_max_drawdown_and_perf():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    x = pd.Series([0.10, -0.20, -0.10, 0.05, 0.30, 0.00], index=idx)
    dd, peak, trough = rm.max_drawdown(x)
    assert dd == pytest.approx(0.8 * 0.9 - 1)
    assert str(peak) == "2020-01" and str(trough) == "2020-03"
    first_month_loss = pd.Series([-0.10, 0.02], index=idx[:2])
    assert rm.max_drawdown(first_month_loss)[0] == pytest.approx(-0.10)
    p = rm.perf(x)
    assert p["ann_return_%"] == pytest.approx(((1 + x).prod() ** 2 - 1) * 100)
    assert p["ann_vol_%"] == pytest.approx(x.std() * np.sqrt(12) * 100)
    assert p["sharpe"] == pytest.approx(x.mean() / x.std() * np.sqrt(12))
    rf = pd.Series(0.01, index=idx)
    assert rm.perf(x, rf)["sharpe"] == pytest.approx((x.mean() - 0.01) / x.std() * np.sqrt(12))


# ---------------------------------------------------------------------- end to end
def test_long_input_to_table2():
    ret, fac = simulate(n_stocks=50, n_months=84, seed=2)
    long = ret.stack().rename("ret").reset_index()
    long.columns = ["month", "id", "ret"]
    long["month"] = long["month"].astype(str)
    f = fac.reset_index().rename(columns={"index": "month"})
    f["month"] = f["month"].astype(str)
    wide, fac2 = rm.to_wide(long), rm.prep_factors(f)
    rs, ts = rm.scores(wide, fac2)
    s = rm.summarize(rm.wml(rs, wide), rm.wml(ts, wide))
    assert s.index.tolist() == [3, 6, 9, 12]
    assert s.loc[3, "months"] == 84 - 36 - 2 and s.loc[12, "months"] == 84 - 36 - 11
    assert s[["residual_t", "total_t"]].notna().all().all()
    half = rm.summarize(rm.wml(rs, wide), rm.wml(ts, wide), start="2005-01")
    assert half.loc[3, "months"] == 24
    t = rm.table2({"Simland": s, "Otherland": s})
    assert t.index.tolist() == ["Simland", "Otherland"]
    assert ("Residual", "K=3") in t.columns and ("Total", "K=12") in t.columns


# ------------------------------------------------------------------ score weighting
def test_cap_weights():
    w = rm.cap_weights(np.array([6.0, 2.0, 1.0, 1.0, 0.0]), 0.4)
    assert w.sum() == pytest.approx(1.0) and w.max() == pytest.approx(0.4)
    assert w[1] / w[2] == pytest.approx(2.0) and w[4] == 0.0      # others keep proportions
    assert rm.cap_weights(np.array([3.0, 1.0]), None).tolist() == [0.75, 0.25]
    assert rm.cap_weights(np.array([3.0, 1.0]), 0.3).tolist() == [0.75, 0.25]   # cap infeasible


def test_score_weighting_tilts_to_highest_scores():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    cols = [f"S{i:02d}" for i in range(40)]
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    ret.loc[idx[1], ["S36", "S37", "S38", "S39"]] = [0.10, 0.20, 0.30, 0.40]
    s = pd.Series(np.arange(40.0) - 20, index=cols)             # winners score 16, 17, 18, 19
    s["S39"] = 49.0                                             # scores 16, 17, 18, 49
    score = pd.DataFrame([s], index=idx[1:])
    out = rm.wml(score, ret, hold=(1,), score_weight=True)
    assert out.loc[idx[1], "W1"] == pytest.approx((16 * .1 + 17 * .2 + 18 * .3 + 49 * .4) / 100)
    capped = rm.wml(score, ret, hold=(1,), score_weight=True, cap=0.4)
    rest = 0.6 / 51
    assert capped.loc[idx[1], "W1"] == pytest.approx(rest * (16 * .1 + 17 * .2 + 18 * .3) + 0.4 * 0.4)
