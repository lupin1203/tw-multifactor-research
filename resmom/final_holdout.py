"""The single final run (FACTOR_SPEC.md, 封存期檢驗, 2026-11-18): the frozen mix, aggressive|none
80% + stable 20% (quality book quarterly, downside covariance, 70% + UST 10-30y 30%, no switch),
reset to 80/20 at the start of every month, exactly as in the 2023 table. Reads step 2's and
stable round two's runs with --end 2026-10 --open-holdout.

    python final_holdout.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_step1 as rs
import run_step2 as r2

HERE = Path(__file__).parent
END = "2026-10"
OUT = HERE / "output" / "final" / f"to_{END}"
STABLE_KEY = "long|off|quarterly|down"
W_STABLE = 0.2
START = "2009-07"
HOLDOUT = pd.Timestamp("2024-01-01")


def stats(d: pd.Series, m: pd.Series) -> dict:
    x = pd.concat([d.rename("p"), m.rename("m")], axis=1).dropna()
    rp, rm_, dp, dm = r2.ann(x["p"]), r2.ann(x["m"]), r2.dd_daily(x["p"]), r2.dd_daily(x["m"])
    return {"start": str(x.index[0].date()), "end": str(x.index[-1].date()), "days": len(x),
            "ret_%": rp * 100, "mkt_ret_%": rm_ * 100, "ret_x_mkt": rp / rm_ if rm_ > 0 else np.nan,
            "dd_%": dp * 100, "mkt_dd_%": dm * 100, "dd_x_mkt": dp / dm if dm < 0 else np.nan,
            "cum_%": ((1 + x["p"]).prod() - 1) * 100, "mkt_cum_%": ((1 + x["m"]).prod() - 1) * 100}


def main() -> None:
    cfg = rs.Cfg(end=END, open_holdout=True)
    D = rs.prepare(rs.load_raw(), cfg)
    first, last = pd.Period(START, "M").start_time, D["end"].end_time
    cut = lambda d: d.loc[(d.index >= first) & (d.index <= last)].dropna()
    mkt = cut(D["mkt_d"])
    agg = cut(pd.read_csv(HERE / "output" / "step2" / f"to_{END}" / "daily_returns.csv", index_col=0,
                          parse_dates=True)["aggressive|none"])
    stb = cut(pd.read_csv(HERE / "output" / "stable2" / f"to_{END}" / "daily_returns.csv", index_col=0,
                          parse_dates=True)[STABLE_KEY])
    mix = r2.two_assets(stb, agg, W_STABLE, months=tuple(range(1, 13)))
    series = {"mix_80_20": mix, "aggressive": agg, "stable": stb}
    rows = {}
    for name, s in series.items():
        rows[(name, "to_2023-12")] = stats(s.loc[:HOLDOUT - pd.Timedelta(days=1)], mkt)
        rows[(name, "holdout_2024-01_on")] = stats(s.loc[HOLDOUT:], mkt)
        rows[(name, "full")] = stats(s, mkt)
    tab = pd.DataFrame(rows).T.rename_axis(["series", "period"])
    yrs = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in {"market": mkt, **series}.items()}) * 100
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    yrs.to_csv(OUT / "calendar_years_%.csv", encoding="utf-8-sig")
    pd.DataFrame({"market": mkt, **series}).to_csv(OUT / "daily_returns.csv")
    pd.set_option("display.width", 250)
    print(tab[["start", "end", "ret_%", "mkt_ret_%", "ret_x_mkt", "dd_%", "mkt_dd_%", "dd_x_mkt", "cum_%",
               "mkt_cum_%"]].to_string())
    print(yrs.round(1).to_string())


if __name__ == "__main__":
    main()
