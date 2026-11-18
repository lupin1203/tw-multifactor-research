"""Tests for run_stable2.py on simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fake_data as fd
import run_step1 as rs
import run_stable as st
import run_stable2 as s2

DAYS = pd.bdate_range("2015-01-01", "2016-12-31")


def test_trend_target_acts_the_next_day():
    px = pd.Series(100.0, index=DAYS)
    px.loc["2016-03-01":] = 90.0                           # falls below its 200-day average that day
    tg = s2.trend_target(px, DAYS, 0.7, 0.4, 200)
    assert tg.loc["2016-03-01"] == 0.7 and tg.loc["2016-03-02"] == 0.4
    assert (tg.loc[:"2015-10-01"] == 0.7).all()            # no average yet: stay at 70%


def test_mix_dynamic_resets_when_the_target_changes():
    a, b = pd.Series(0.01, index=DAYS), pd.Series(0.0, index=DAYS)
    tg = pd.Series(0.7, index=DAYS)
    tg.loc["2015-06-10":] = 0.4
    m = s2.mix_dynamic(a, b, tg, cost=0.01)
    day = DAYS[DAYS >= "2015-06-10"][0]
    assert m.iloc[0] == pytest.approx(0.007)
    assert 0.004 - 0.01 * 0.5 < m.loc[day] < 0.004            # 40% of 1% minus the cost of the switch
    nxt = DAYS[DAYS.get_loc(day) + 1]
    assert m.loc[nxt] == pytest.approx(0.004 * 1.01 / 1.004, rel=1e-9)   # drifting from 40%
    assert np.allclose(s2.mix_dynamic(a, b, pd.Series(0.7, index=DAYS)), st.r2.two_assets(a, b, 0.7))


def test_downside_covariance_ignores_up_days():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, (300, 5))
    up = np.abs(rng.normal(0, 0.05, (300, 5)))
    X2 = np.where(X < 0, X, up)                            # same down days, wild up days
    a, _ = st.lw_constant_corr(np.minimum(X, 0), demean=False)
    b, _ = st.lw_constant_corr(np.minimum(X2, 0), demean=False)
    assert np.allclose(a, b)


@pytest.fixture(scope="module")
def result():
    raw = fd.fake_raw(seed=9, n=140, start="2005-01", end="2024-06", plant_mom=0.02, plant_yoy=0.02)
    raw["me"] = raw["me"] * 50
    days = raw["dret"].index
    rng = np.random.default_rng(10)
    y = 3 + np.cumsum(rng.normal(0, 0.05, len(days)))
    ust = pd.DataFrame({m: y + 0.05 * m for m in [1, 2, 3, 5, 7, 10, 20, 30]}, index=days)
    fx = pd.DataFrame({"mid": 30 * np.exp(np.cumsum(rng.normal(0, 0.003, len(days)))), "half_spread": 0.0016}, index=days)
    return s2.run(rs.Cfg(), st.CfgS(mc_draws=100), s2.Cfg2S(), None, log=lambda *_: None, raw=raw, ust=ust, fx=fx,
                  agg=raw["dret"].mean(axis=1))


def test_sixteen_combinations_and_a_pick(result):
    T, S = result["T"], result["series"]
    assert len(T["combinations"]) == 16 and T["combinations"]["p_goal_mc"].between(0, 1).all()
    assert len(T["books"]) == 4 and "pick" in T and len(T["combined_with_aggressive"]) == 11
    assert all(v.dropna().index.max() <= pd.Timestamp("2023-12-31") for v in S.values() if len(v.dropna()))


def test_quarterly_book_trades_only_in_quarter_months(result):
    T = result["T"]["books"]
    assert T.loc["quarterly|cov", "turnover_%/yr"] < T.loc["monthly|cov", "turnover_%/yr"]
    assert T.loc["quarterly|cov", "kept_by_buffer"] >= 0
