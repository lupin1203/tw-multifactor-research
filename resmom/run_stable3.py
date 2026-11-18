"""Stable sleeve, round three (FACTOR_SPEC.md, 穩定那塊第三輪, 2026-11-18).

The stock part is the multi-factor book (step 2's aggressive|none daily returns), mixed with
US Treasuries: stock weight 70% or 60%, 1-3 or 10-30 year ladder, 200-day switch off or on
(40%). Bond returns come from round two's output; both files end at 2023-12.

    python run_stable3.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import run_step1 as rs
import run_step2 as r2
import run_stable as st
import run_stable2 as s2

HERE = Path(__file__).parent
OUT = HERE / "output" / "stable3" / "to_2023-12"
BONDS = HERE / "output" / "stable2" / "to_2023-12" / "daily_returns.csv"


def main() -> None:
    cfg, c, c2 = rs.Cfg(), st.CfgS(), s2.Cfg2S()
    D = rs.prepare(rs.load_raw(), cfg)
    last, first = D["end"].end_time, pd.Period(c.start, "M").start_time
    cut = lambda d: d.loc[(d.index >= first) & (d.index <= last)]
    mkt = cut(D["mkt_d"].dropna())
    rf_d = pd.Series(D["rf"].reindex(mkt.index.to_period("M")).to_numpy() / 21, index=mkt.index)
    buy, sell, _ = D["cost"]["tradable"]
    fx_half = float(st.dt.usdtwd()["half_spread"].loc[first:last].median())
    cost = (buy + sell) / 2 + c.bond_cost + fx_half
    agg = cut(pd.read_csv(st.AGG_FILE, index_col=0, parse_dates=True)["aggressive|none"].dropna())
    bd = pd.read_csv(BONDS, index_col=0, parse_dates=True)
    bonds = {"short": cut(bd["bond|short"].dropna()), "long": cut(bd["bond|long"].dropna())}
    mk_m = r2.monthly(mkt)
    block = max(1, round(len(mk_m) ** (1 / 3)))
    rng = np.random.default_rng(c.seed)
    rows, series = {}, {"aggressive": agg}
    for w, bond, trend in itertools.product((0.7, 0.6), ("short", "long"), ("off", "on")):
        tgt = (s2.trend_target(D["market"]["taiex"], mkt.index, w, c2.stock_lo, c2.ma_days) if trend == "on"
               else pd.Series(w, index=mkt.index))
        s = s2.mix_dynamic(agg, bonds[bond], tgt, cost=cost)
        key = (w, bond, trend)
        series["|".join(map(str, key))] = s
        mc = r2.mc_report(r2.monthly(s), mk_m, c.ret_gap, c.dd_mult, None, block, c.mc_draws, rng)
        rows[key] = {**r2.evaluate(s, mkt, rf_d, c.ret_gap, c.dd_mult), "p_goal_mc": mc["p_goal_both_halves"],
                     "changes": (w != 0.7) + (bond == "long") + (trend == "on")}
    rows[("aggressive", "-", "-")] = r2.evaluate(agg, mkt, rf_d, c.ret_gap, c.dd_mult)
    tab = pd.DataFrame(rows).T.rename_axis(["stock_weight", "bonds", "trend"])
    passed = tab[tab["goal_met"].astype(bool) & tab["p_goal_mc"].notna()]
    pick = None
    if len(passed):
        near = passed[passed["p_goal_mc"] >= passed["p_goal_mc"].max() - c2.pick_margin]
        pick = near.sort_values(["changes", "p_goal_mc"], ascending=[True, False]).index[0]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "combinations.csv", encoding="utf-8-sig")
    st.calendar_years({"market": mkt, **series}).to_csv(OUT / "calendar_years_%.csv", encoding="utf-8-sig")
    pd.DataFrame(series).to_csv(OUT / "daily_returns.csv")
    pd.DataFrame({"picked": ["|".join(map(str, pick)) if pick else "none passed"], "n_passed": [len(passed)]}).to_csv(
        OUT / "pick.csv", encoding="utf-8-sig")
    print(tab[["ret_%_full", "dd_x_mkt_full", "ret_%_h1", "dd_x_mkt_h1", "ret_%_h2", "mkt_ret_%_h2", "dd_x_mkt_h2",
               "goal_met", "p_goal_mc"]].round(2).to_string())
    print("pick:", pick, "passed:", len(passed))


if __name__ == "__main__":
    main()
