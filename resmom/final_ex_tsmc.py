"""Post-hoc comparison (FACTOR_SPEC.md, 去除台積電的比較, 2026-11-18): the frozen 80/20 mix and its
sleeves against a value-weighted market WITHOUT TSMC (2330), next to the official total-return
index and the self-built value-weighted market with TSMC. The benchmark was chosen after the
holdout was seen, so this is not a clean test.

    python final_ex_tsmc.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_step1 as rs
import run_step2 as r2

HERE = Path(__file__).parent
END = "2026-10"
SRC = HERE / "output" / "final" / f"to_{END}" / "daily_returns.csv"
OUT = HERE / "output" / "final" / f"to_{END}" / "ex_tsmc"
HOLDOUT = pd.Timestamp("2024-01-01")
GOALS = {"mix_80_20": (1.3, None, 1.1), "aggressive": (None, 0.05, 1.2), "stable": (None, None, 0.7)}  # ret_mult, ret_gap, dd_mult


def one(p: pd.Series, m: pd.Series) -> dict:
    x = pd.concat([p.rename("p"), m.rename("m")], axis=1, sort=True).dropna()
    rp, rm_, dp, dm = r2.ann(x["p"]), r2.ann(x["m"]), r2.dd_daily(x["p"]), r2.dd_daily(x["m"])
    return {"ret_%": rp * 100, "mkt_ret_%": rm_ * 100, "dd_x_mkt": dp / dm if dm < 0 else np.nan}


def main() -> None:
    D = rs.prepare(rs.load_raw(), rs.Cfg(end=END, open_holdout=True))
    f = pd.read_csv(SRC, index_col=0, parse_dates=True)
    days = f.index
    dret, me = D["dret"], D["me"]
    keep = [c for c in dret.columns if c != "2330"]
    bench = {"official_TR": f["market"],
             "vw_all": rs.vw_daily(dret, me).reindex(days),
             "vw_ex_2330": rs.vw_daily(dret[keep], me[[c for c in me.columns if c != "2330"]]).reindex(days)}
    share = (me["2330"] / me.sum(axis=1)).dropna()
    pre, post = days[days < HOLDOUT], days[days >= HOLDOUT]
    k = len(pre) // 2
    periods = {"h1_2009-2016": pre[:k], "h2_2016-2023": pre[k:], "holdout_2024-2026": post, "full": days}
    rows = []
    for b, m in bench.items():
        for s, (ret_mult, ret_gap, dd_mult) in GOALS.items():
            for per, idx in periods.items():
                r = one(f[s].loc[idx], m.loc[idx])
                need = (ret_mult * r["mkt_ret_%"] if ret_mult else r["mkt_ret_%"] + ret_gap * 100 if ret_gap is not None
                        else -np.inf)
                rows.append({"benchmark": b, "series": s, "period": per, **r, "ret_needed_%": need,
                             "goal_met": bool(r["ret_%"] >= need and r["dd_x_mkt"] <= dd_mult)})
    tab = pd.DataFrame(rows).set_index(["benchmark", "series", "period"])
    yrs = pd.DataFrame({k: (1 + v.dropna()).groupby(v.dropna().index.year).prod() - 1 for k, v in bench.items()}) * 100
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "goals.csv", encoding="utf-8-sig")
    yrs.to_csv(OUT / "benchmark_years_%.csv", encoding="utf-8-sig")
    share.groupby(share.index.year).last().mul(100).to_csv(OUT / "tsmc_weight_%.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string())
    print(yrs.round(1).to_string())
    print((share.groupby(share.index.year).last() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
