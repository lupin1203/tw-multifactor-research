"""Simulated market in the same shape as run_step1.load_raw(), for testing the pipeline
without looking at real results. Nothing here resembles real Taiwan data in its numbers.

Planted effects (both off by default), written from the raw publication rule and NOT with
signals.py, so a wrong month label in signals.py cannot hide:
  revenue of month r is published by the 10th of r+1; the stock's return in calendar
  month r+2 gets   plant_mom x z(month-on-month change of r)  +  plant_yoy x z(3-month
  year-on-year growth ending in r),   z = cross-sectional normal score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

import data_tw as dt

INDUSTRIES = ["半導體業", "電子零組件業", "光電業", "生技醫療業", "電機機械", "金融保險", "航運業", None]


def _z(x: pd.DataFrame) -> pd.DataFrame:
    n = x.notna().sum(axis=1)
    r = (x.rank(axis=1) - 0.5).div(n.where(n > 0), axis=0)
    return pd.DataFrame(ndtri(r.to_numpy(dtype=float)), index=x.index, columns=x.columns)


def fake_raw(seed: int = 0, n: int = 200, start: str = "2008-01", end: str = "2025-06",
             plant_mom: float = 0.0, plant_yoy: float = 0.0, season: float = 0.15) -> dict:
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    days = pd.bdate_range(months[0].start_time, months[-1].end_time.normalize())
    per = days.to_period("M")
    ids = [str(1101 + i) for i in range(n)]
    T = len(days)

    # listing and delisting
    first = np.where(rng.random(n) < 0.85, 0, rng.integers(200, T // 2, n))
    last = np.where(rng.random(n) < 0.08, rng.integers(T // 2, T - 300, n), T)
    alive = (np.arange(T)[:, None] >= first) & (np.arange(T)[:, None] < last)

    # monthly revenue, from three years before the first price
    rmonths = pd.period_range(months[0] - 36, months[-1], freq="M")
    growth = rng.normal(0.004, 0.05, (len(rmonths), n)).cumsum(0)
    season = 1 + season * np.sin(2 * np.pi * (rmonths.month.to_numpy()[:, None] / 12 + rng.random(n)))
    size = np.exp(rng.normal(0, 1, n))
    rev = pd.DataFrame(np.exp(growth) * season * 2e8 * size, index=rmonths, columns=ids)

    # planted drift for calendar month r+2
    drift = pd.DataFrame(0.0, index=months, columns=ids)
    if plant_mom or plant_yoy:
        mom = rev / rev.shift(1) - 1
        r3 = rev.rolling(3).sum()
        yoy = r3 / r3.shift(12) - 1
        eff = plant_mom * _z(mom).fillna(0) + plant_yoy * _z(yoy).fillna(0)
        eff.index = eff.index + 2
        drift = eff.reindex(months).fillna(0.0)

    # daily returns: market + idiosyncratic + planted drift spread over the month's days
    mkt = rng.normal(0.0004, 0.010, T)
    beta = rng.uniform(0.6, 1.4, n)
    n_days = pd.Series(1, index=days).groupby(per).transform("sum").to_numpy()
    r = beta * mkt[:, None] + rng.normal(0, 0.02, (T, n)) + drift.reindex(per).to_numpy() / n_days[:, None]
    r = np.clip(r, -0.095, 0.095)
    # three stocks stop trading for about two months and come back with a jump: a real loss, a
    # real gain, and a capital reduction (price x 4, shares / 4) that the adjustment data misses
    whole = np.flatnonzero((first == 0) & (last == T))[:3]
    halts, back = {}, []
    for k, (j, jump) in enumerate(zip(whole, (-0.40, 0.25, 3.0))):
        h0 = T // 2 + 37 * k
        alive[h0:h0 + 45, j] = False
        r[h0 + 45, j] = jump
        back.append((h0 + 45, j))
        halts[ids[j]] = {"stops": days[h0], "resumes": days[h0 + 45], "jump": jump}
    dret_raw = pd.DataFrame(np.where(alive, r, np.nan), index=days, columns=ids)
    for j in range(n):                                    # the first day of a listing has a price but no return
        dret_raw.iloc[first[j], j] = np.nan
    dret = dret_raw.copy()
    for i, j in back:                                     # the cleaned returns drop the jump (beyond the price limit)
        dret.iloc[i, j] = np.nan
    p0 = np.exp(rng.normal(np.log(60), 0.7, n))
    close = pd.DataFrame(np.where(alive, p0 * np.cumprod(1 + np.where(alive, r, 0.0), axis=0), np.nan), index=days, columns=ids)
    volume = pd.DataFrame(np.where(alive, np.exp(rng.normal(np.log(6e5 * size), 0.6, (T, n))), np.nan), index=days, columns=ids)

    # monthly panel
    close_m = close.groupby(per).last()                   # last traded price of the month, as in the real panel
    shares = pd.DataFrame(np.tile(2e8 * size, (len(months), 1)), index=months, columns=ids)
    cr = [s for s, h in halts.items() if h["jump"] > 1]
    shares.loc[shares.index >= halts[cr[0]]["resumes"].to_period("M"), cr[0]] /= 4      # the capital reduction
    shares = shares.where(close_m.notna())
    me = close_m * shares
    ret_raw = dt.monthly_from_daily(dret).reindex(index=months, columns=ids).where(me.notna())
    pbr = pd.DataFrame(np.exp(rng.normal(0.3, 0.5, n) + rng.normal(0, 0.05, (len(months), n))), index=months, columns=ids).where(me.notna())

    # quarterly statements (single quarters)
    q = rev.groupby(rev.index.asfreq("Q")).sum()
    q = q[rev.groupby(rev.index.asfreq("Q")).count().iloc[:, 0] == 3]
    gm = rng.uniform(0.15, 0.5, n)
    e = lambda sd: rng.normal(0, sd, q.shape)
    ni = q * (gm - 0.15 + e(0.03))
    assets = q.rolling(4).sum() * 4
    stm = {"rev": q, "gross": q * (gm + e(0.02)), "oper": q * (gm - 0.10 + e(0.02)), "ni": ni,
           "cfo": ni * (1 + e(0.3)), "assets": assets, "liab": assets * rng.uniform(0.2, 0.7, n)}

    # flows (shares); institutional figures start later than margin figures
    so = shares.reindex(per).set_axis(days).to_numpy()
    inst_on = (days >= pd.Timestamp(months[0].start_time) + pd.DateOffset(years=4))[:, None]
    walk = lambda scale: np.abs(rng.normal(0, scale, (T, n)).cumsum(0)) * np.nan_to_num(so)
    net = lambda: pd.DataFrame(np.where(alive & inst_on, rng.normal(0, 2e-4, (T, n)) * np.nan_to_num(so), np.nan), index=days, columns=ids)
    bal = lambda scale: pd.DataFrame(np.where(alive, walk(scale), np.nan), index=days, columns=ids)
    flows = {"foreign_net": net(), "trust_net": net(), "margin_bal": bal(2e-4), "short_bal": bal(5e-5), "sbl_bal": bal(1e-4)}

    # factors, attributes, market-wide series, benchmark
    ew = ret_raw.mean(axis=1)
    fac = pd.DataFrame({"RMRF": ew - 0.001, "SMB": rng.normal(0, 0.02, len(months)), "HML": rng.normal(0, 0.02, len(months)),
                        "RF": 0.001}, index=months).dropna()
    ind = pd.Series(rng.choice(np.array(INDUSTRIES, dtype=object), n), index=ids)
    fin = (ind == "金融保險").fillna(False).astype(bool)
    half = pd.Series(np.arange(n) % 2 == 0, index=ids)            # the first industry is split in two, like semiconductors
    fine = (ind + "-A").where((ind == INDUSTRIES[0]) & half, ind)
    info = pd.DataFrame({"industry": fine, "industry_official": ind, "financial": fin, "late": fin | (rng.random(n) < 0.05)})
    fut_on = days >= pd.Timestamp(months[0].start_time) + pd.DateOffset(years=6)
    market = {"inst_net": pd.Series(rng.normal(0, 5e9, T), index=days),
              "turnover": pd.Series(np.exp(rng.normal(np.log(1e11), 0.3, T)), index=days),
              "taiex": pd.Series(8000 * np.cumprod(1 + mkt), index=days),
              "usdtwd": pd.Series(30 + rng.normal(0, 0.05, T).cumsum(), index=days),
              "fut_foreign": pd.Series(rng.normal(0, 2000, T).cumsum(), index=days)[fut_on]}
    bench = {"0050": (ew + pd.Series(rng.normal(0, 0.01, len(months)), index=months)).dropna()}
    # the "official" index: the common daily move plus a little noise, as a level
    mkt_level = pd.Series(10000 * np.cumprod(1 + mkt + np.random.default_rng(seed + 1000).normal(0, 0.0005, T)), index=days)
    # of the stocks that vanish for good, every second one was a merger (holder paid out) with a
    # delisting date a week after the last trade; the others are unexplained
    gone_for_good = np.flatnonzero(last < T)
    mergers = pd.DataFrame({"stock": [ids[j] for j in gone_for_good[::2]],
                            "delisted": [days[last[j] - 1] + pd.Timedelta(days=7) for j in gone_for_good[::2]]})
    return {"dret": dret, "dret_raw": dret_raw, "close": close, "volume": volume, "ret_raw": ret_raw, "me": me, "pbr": pbr,
            "halts": halts, "mergers": mergers, "mkt_level": mkt_level,
            "shares": shares, "close_m": close_m, "fac": fac, "info": info, "market": market, "bench": bench,
            "rev": rev, "stm": stm, "div_yield": pd.DataFrame(rng.uniform(0, 0.06, (len(months), n)), index=months, columns=ids),
            "flows": flows}


def scramble_after(raw: dict, cut: str = "2023-12", seed: int = 99) -> dict:
    """A copy of `raw` in which every number dated after `cut` is replaced by garbage
    (fundamentals: every figure not yet published by the end of `cut`)."""
    import signals as sg
    rng = np.random.default_rng(seed)
    c = pd.Period(cut, "M")

    def junk(x, later):
        x = x.copy()
        k = int(np.sum(later))
        if k:
            shape = (k,) + x.shape[1:]
            x.loc[later] = (rng.normal(0, 1, shape) * np.nanmean(np.abs(x.to_numpy(dtype=float)))).astype(float)
        return x

    def one(x, kind="auto"):
        if isinstance(x, dict):
            return {k: one(v) for k, v in x.items()}
        if not isinstance(x, (pd.DataFrame, pd.Series)):
            return x
        i = x.index
        if isinstance(i, pd.DatetimeIndex):
            return junk(x, i > c.end_time)
        if isinstance(i, pd.PeriodIndex) and i.freqstr.startswith("Q"):
            return junk(x, np.array([sg.usable_month(p) > c for p in i]))
        if isinstance(i, pd.PeriodIndex):
            return junk(x, i > c)
        return x

    out = {k: one(v) for k, v in raw.items() if k not in ("rev", "halts", "mergers")}
    out["rev"] = junk(raw["rev"], raw["rev"].index + 1 > c)
    out["halts"] = raw["halts"]
    # garbage about the future: every stock is "merged" shortly after the cut
    later = pd.DataFrame({"stock": list(raw["me"].columns), "delisted": c.end_time.normalize() + pd.Timedelta(days=10)})
    out["mergers"] = pd.concat([raw["mergers"], later], ignore_index=True)
    return out
