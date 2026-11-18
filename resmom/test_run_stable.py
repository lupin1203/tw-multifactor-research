"""Tests for run_stable.py on simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fake_data as fd
import run_step1 as rs
import run_stable as st

MATS = [1, 2, 3, 5, 7, 10, 20, 30]
DAYS = pd.bdate_range("2010-01-01", "2014-12-31")


def flat_curve(y: float, days=DAYS) -> pd.DataFrame:
    return pd.DataFrame(y, index=days, columns=MATS)


def test_par_bond_prices_at_one():
    assert st.bond_price(0.04, 10.0, 0.04) == pytest.approx(1.0, abs=1e-12)
    assert st.bond_price(0.05, 10.0, 0.04) < 1 < st.bond_price(0.03, 10.0, 0.04)
    assert st.bond_price(0.04, 0.0, 0.04) == 0.0


def test_ladder_earns_the_yield_when_rates_do_not_move():
    lad = st.ladder_usd(flat_curve(4.0), 10, 30, "2010-01-01", 0.0)
    yrs = (lad.index[-1] - lad.index[0]).days / 365.25
    a = (1 + lad["ret"]).prod() ** (1 / yrs) - 1
    assert a == pytest.approx(1.02 ** 2 - 1, abs=0.002)
    assert lad["ret"].std() < 1e-3                        # no jumps on coupon days


def test_ladder_loses_about_its_duration_on_a_rate_jump():
    Y = flat_curve(4.0)
    Y.loc["2012-06-01":] = 5.0
    day = Y.index[Y.index >= "2012-06-01"][0]
    long_ = st.ladder_usd(Y, 10, 30, "2010-01-01", 0.0).loc[day, "ret"]
    short = st.ladder_usd(Y, 0, 10, "2010-01-01", 0.0).loc[day, "ret"]
    assert -0.17 < long_ < -0.10                          # duration about 13 years
    assert -0.07 < short < -0.03                          # about 5 years


def test_ladder_pays_cost_only_at_the_roll():
    a = st.ladder_usd(flat_curve(4.0), 10, 30, "2010-01-01", 0.0)
    b = st.ladder_usd(flat_curve(4.0), 10, 30, "2010-01-01", 0.001)
    d = (b["ret"] - a["ret"]).abs()
    assert set(d[d > 1e-9].index.year) == set(d[d > 1e-9].index.year) and (d[d > 1e-9].index.month == 1).all()
    assert (b["traded"] > 0).sum() == 4                    # one roll a year after the first
    assert b["traded"].max() < 0.2


def test_to_twd_uses_the_previous_us_close_and_the_rate():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.0, index=days)
    r.loc["2020-02-03"] = 0.01
    fx = pd.Series(30.0, index=days)
    out = st.to_twd(r, fx, days)
    assert out.loc["2020-02-04"] == pytest.approx(0.01) and out.loc["2020-02-03"] == pytest.approx(0.0)
    fx.loc["2020-03-02":] = 30.3
    out = st.to_twd(pd.Series(0.0, index=days), fx, days)
    assert out.loc["2020-03-02"] == pytest.approx(0.01)


def test_ledoit_wolf_keeps_variances_and_is_positive_definite():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, (252, 1))
    X = common + rng.normal(0, 0.015, (252, 80))
    cov, d = st.lw_constant_corr(X)
    S = np.cov(X, rowvar=False, bias=True)
    assert 0 <= d <= 1 and d > 0.05
    assert np.allclose(np.diag(cov), np.diag(S)) and np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_min_var_known_answers_and_limits():
    w, ok = st.min_var(np.diag([1.0, 4.0]), np.array(["a", "b"]), 1.0, None)
    assert ok and np.allclose(w, [0.8, 0.2], atol=1e-4)
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (300, 40)) * np.linspace(0.5, 2, 40)
    g = np.array(["x"] * 20 + list("abcdefghijklmnopqrst"))
    w, ok = st.min_var(np.cov(X, rowvar=False), g, 0.05, 0.20)
    assert ok and w.sum() == pytest.approx(1.0) and w.max() <= 0.05 + 1e-9
    assert w[g == "x"].sum() <= 0.20 + 1e-6
    w, _ = st.min_var(np.eye(10), np.array(list("abcdefghij")), 0.05, 0.20)
    assert w.sum() == pytest.approx(0.5)                  # 10 x 5%: the rest stays in cash


@pytest.fixture(scope="module")
def result():
    raw = fd.fake_raw(seed=6, n=140, start="2005-01", end="2024-06", plant_mom=0.02, plant_yoy=0.02)
    for k in ("me",):                                    # make most stocks big enough
        raw[k] = raw[k] * 50
    days = raw["dret"].index
    rng = np.random.default_rng(7)
    y = 3 + np.cumsum(rng.normal(0, 0.05, len(days)))
    ust = pd.DataFrame({m: y + 0.05 * m for m in MATS}, index=days)
    fx = pd.DataFrame({"mid": 30 * np.exp(np.cumsum(rng.normal(0, 0.003, len(days)))), "half_spread": 0.0016}, index=days)
    agg = raw["dret"].mean(axis=1)
    return st.run(rs.Cfg(), st.CfgS(mc_draws=100), None, log=lambda *_: None, raw=raw, ust=ust, fx=fx, agg=agg)


def test_filters_nest_and_hold(result):
    Q, D = result["Q"], result["D"]
    assert not (Q["cand"] & ~Q["passed"]).any().any() and not (Q["passed"] & ~Q["pool"]).any().any()
    assert Q["cand"].sum(axis=1).sum() > 0


def test_minvar_weights_respect_the_limits(result):
    D, W = result["D"], result["W"]
    ind = D["groups_official"]
    assert len(W) > 100
    for f, w in W.items():
        assert w.max() <= 0.05 + 1e-6 and w.sum() <= 1 + 1e-9
        assert w.groupby(ind.reindex(w.index)).sum().max() <= 0.20 + 1e-4
        assert result["Q"]["cand"].loc[f, w.index].all()


def test_every_table_and_nothing_after_the_cut(result):
    T, S = result["T"], result["series"]
    assert {"bonds", "books", "stable", "monte_carlo", "combined_with_aggressive", "filter_counts",
            "minvar_industry_weights", "calendar_years_%", "drawdown_dates"} <= set(T)
    assert all(v.dropna().index.max() <= pd.Timestamp("2023-12-31") for v in S.values() if len(v.dropna()))
    assert T["stable"].loc[("A_book", "none"), "avg_exposure"] == pytest.approx(1.0)
    assert len(T["combined_with_aggressive"]) == 11


def test_holdout_stays_closed():
    raw = fd.fake_raw(seed=8, n=60, start="2015-01", end="2024-06")
    with pytest.raises(ValueError, match="holdout"):
        st.run(rs.Cfg(end="2024-03"), st.CfgS(mc_draws=10), None, log=lambda *_: None, raw=raw,
               ust=flat_curve(4.0), fx=pd.DataFrame({"mid": 30.0, "half_spread": 0.001}, index=DAYS))
