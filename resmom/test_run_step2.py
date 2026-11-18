"""Tests for run_step2.py on simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fake_data as fd
import run_step1 as rs
import run_step2 as r2

DAYS = pd.bdate_range("2015-01-01", "2018-12-31")


def const(x: float) -> pd.Series:
    return pd.Series(x, index=DAYS)


def test_overlay_without_rules_is_the_sleeve():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.0005, 0.01, len(DAYS)), index=DAYS)
    m = pd.Series(rng.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
    ov = r2.overlay(s, m, const(0.0001), None, 1.0, None)
    assert np.allclose(ov["ret"], s) and np.allclose(ov["exposure"], 1.0)


def test_vol_target_sets_exposure_from_the_vol_ratio_and_caps_it():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    s = 2 * m                                                   # twice as volatile as the market
    ov = r2.overlay(s, m, const(0.0), 0.7, 1.0, None)
    first_days = ov.index.to_series().groupby(ov.index.to_period("M")).transform("min") == ov.index
    e = ov.loc[first_days, "exposure"].iloc[1:]
    assert np.allclose(e, 0.35, atol=1e-9)                      # 0.7 x 1/2
    ov2 = r2.overlay(m / 2, m, const(0.0), 1.2, 1.5, None)      # half as volatile: 2.4, capped at 1.5
    assert np.allclose(ov2.loc[first_days, "exposure"].iloc[1:], 1.5, atol=1e-9)


def test_grossman_zhou_full_at_the_peak_and_zero_at_the_limit():
    up = const(0.001)
    ov = r2.overlay(up, up, const(0.0), None, 1.0, 0.30)
    assert np.allclose(ov["exposure"], 1.0)                     # always at a new peak
    crash = const(0.0)
    crash.loc["2016-01"] = -0.0175                              # about -30% over January 2016
    ov = r2.overlay(crash, const(0.0), const(0.0), None, 1.0, 0.30)
    lv = (1 + ov["ret"]).cumprod()
    assert (lv / lv.cummax() - 1).min() > -0.31
    feb = ov.loc["2016-02", "exposure"]
    assert feb.iloc[0] == pytest.approx(0.0, abs=0.05)         # at the limit nothing is held
    half = const(0.0)
    half.loc["2016-01"] = -0.15 / 20                            # about -14% in January
    e = r2.overlay(half, const(0.0), const(0.0), None, 1.0, 0.30).loc["2016-02", "exposure"].iloc[0]
    lv = (1 + half.loc[:"2016-01"]).prod()
    assert e == pytest.approx((1 - 0.7 / lv) / 0.3, abs=1e-6)


def test_overlay_decisions_use_only_earlier_days():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    m = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    a = r2.overlay(s, m, const(0.0), 0.7, 1.0, 0.2)
    s2 = s.copy()
    s2.loc["2017-06-15":] = rng.normal(0, 0.05, len(s2.loc["2017-06-15":]))
    b = r2.overlay(s2, m, const(0.0), 0.7, 1.0, 0.2)
    pd.testing.assert_series_equal(a["exposure"].loc[:"2017-06-15"], b["exposure"].loc[:"2017-06-15"])   # set before the change
    assert not a["exposure"].loc["2017-07"].equals(b["exposure"].loc["2017-07"])


def test_two_assets_resets_on_the_given_months():
    a, b = const(0.01), const(0.0)
    d = r2.two_assets(a, b, 0.5, months=(1, 4, 7, 10))
    assert d.iloc[0] == pytest.approx(0.005)
    assert d.loc["2015-03-31"] > d.iloc[1]                     # drifts towards a inside the quarter
    first_april = d.loc["2015-04"].index[0]
    assert d.loc[first_april] == pytest.approx(0.005)


def test_block_bootstrap_keeps_blocks():
    rng = np.random.default_rng(3)
    idx = r2.block_bootstrap(100, 6, 50, rng)
    assert idx.shape == (50, 100)
    step = np.diff(idx[:, :6], axis=1) % 100
    assert (step == 1).all()


def test_winner_weights_follow_resmom():
    i = pd.period_range("2020-01", periods=2, freq="M")
    cols = [f"s{k}" for k in range(200)]
    score = pd.DataFrame(np.tile(np.arange(200, dtype=float), (2, 1)), index=i, columns=cols)
    U = pd.DataFrame(True, index=i, columns=cols)
    w = r2.winner_weights(score, U, 0.10, 0.03)
    w0 = w[i[0]]
    assert len(w0) == 20 and set(w0.index) == set(cols[-20:])
    assert w0.sum() == pytest.approx(1.0) and w0.max() <= 0.06 + 1e-12     # 3% cap impossible with 20 names
    U.iloc[0, 150:] = False
    assert set(r2.winner_weights(score, U, 0.10, 0.03)[i[0]].index) == set(cols[135:150])


@pytest.fixture(scope="module")
def result():
    raw = fd.fake_raw(seed=4, n=140, start="2005-01", end="2024-06", plant_mom=0.02, plant_yoy=0.02)
    return r2.run(rs.Cfg(), r2.Cfg2(mc_draws=200, control_draws=3), None, log=lambda *_: None, raw=raw)


def test_book_daily_matches_the_monthly_book(result):
    rec = result["T"]["aggressive_reconcile"]
    assert (rec["corr_daily_vs_monthly_book"] > 0.98).all() and (rec["mean_diff_%/mo"].abs() < 0.2).all()


def test_every_table_and_nothing_after_the_cut(result):
    T, S = result["T"], result["series"]
    assert {"sleeves", "combined", "combined_monte_carlo", "sleeve_monte_carlo", "correlation",
            "correlation_summary", "control", "aggressive_reconcile", "stable_turnover"} <= set(T)
    assert all(v.dropna().index.max() <= pd.Timestamp("2023-12-31") for v in S.values() if len(v.dropna()))
    sl = T["sleeves"]
    assert sl.loc[("aggressive", "vol"), "avg_exposure"] <= 1.5 and sl.loc[("stable", "vol"), "avg_exposure"] <= 1.0
    assert sl.loc[("stable", "none"), "avg_exposure"] == pytest.approx(1.0)
    c = T["combined"]
    assert len(c) == 11 and T["combined_monte_carlo"]["p_goal_both_halves"].between(0, 1).all()
    assert set(T["control"]["kind"]) == {"size_matched", "random", "actual_winners"}


def test_holdout_stays_closed():
    raw = fd.fake_raw(seed=5, n=60, start="2015-01", end="2024-06")
    with pytest.raises(ValueError, match="holdout"):
        r2.run(rs.Cfg(end="2024-03"), r2.Cfg2(mc_draws=10, control_draws=1), None, log=lambda *_: None, raw=raw)
