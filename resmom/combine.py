"""Half residual-momentum winners + half MSCI Momentum index, Taiwan or US.

Rebalanced to 50/50 every month. The residual half is the equal-weighted top decile
after trading costs (Taiwan: 0.1425% fee each way + 0.3% tax on sells; US: assumed
0.10% each way); the MSCI half is the gross total-return index in local currency
(no fund fee or tracking cost); the cost of the monthly 50/50 rebalance is ignored.

Needs output/<country>/monthly_returns.csv (run_countries.py) and the MSCI index
file under data/.   Run: python combine.py [tw] [us]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import resmom as rm

HERE = Path(__file__).parent
CFG = {   # folder, MSCI file, MSCI label, benchmark column, round-trip cost, chart title
    "tw": ("taiwan", "msci_taiwan_momentum_703828.csv", "MSCI Taiwan Momentum", "0050",
           0.001425 + 0.004425, "Taiwan", "TWD"),
    "us": ("us", "msci_usa_momentum_703025.csv", "MSCI USA Momentum", "SPY", 0.002, "US", "USD"),
}
EVENTS = {"dd_2020_%": ("2020-01", "2020-12"), "dd_2022_%": ("2022-01", "2022-12")}


def run(key: str) -> None:
    folder, msci_file, MOM, BENCH, RT, title, ccy = CFG[key]
    OUT = HERE / "output" / folder
    m = pd.read_csv(OUT / "monthly_returns.csv", index_col=0)
    m.index = pd.PeriodIndex(m.index, freq="M")
    msci = pd.read_csv(HERE / "data" / msci_file, index_col=0)["ret"]
    msci.index = pd.PeriodIndex(msci.index, freq="M")
    m[MOM] = msci
    rf, bench = m["rf"].fillna(m["rf"].ffill()), m[BENCH]

    series = {}
    for K in rm.HOLD:
        net = (m[f"res_ew10_W{K}"] - m[f"res_ew10_TO_W{K}"].fillna(m[f"res_ew10_TO_W{K}"].mean()) * RT)
        series[f"Residual winners K={K} (net)"] = net
        series[f"50/50 K={K}"] = 0.5 * net + 0.5 * m[MOM]
    series[MOM] = m[MOM]
    series[BENCH] = bench
    S = pd.DataFrame(series).dropna()

    rows = {}
    for name, x in S.items():
        p = rm.perf(x, rf)
        ex = x - bench.reindex(x.index)
        rows[name] = {
            **{k: p[k] for k in ("ann_return_%", "ann_vol_%", "sharpe", "max_dd_%", "dd_peak", "dd_trough")},
            **{k: rm.max_drawdown(rm.cut(x, a, b))[0] * 100 for k, (a, b) in EVENTS.items()},
            f"excess_vs_{BENCH}_%/mo": ex.mean() * 100, "excess_t": rm.nw_t(ex, 6) if name != BENCH else np.nan,
            "tracking_error_%": ex.std() * np.sqrt(12) * 100,
            f"corr_{BENCH}": x.corr(bench.reindex(x.index)),
        }
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(OUT / "combine_5050.csv", encoding="utf-8-sig")
    S.to_csv(OUT / "combine_5050_monthly.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    print(f"\n##### {title}: {S.index[0]} .. {S.index[-1]} ({len(S)} months)\n{tbl.to_string()}")
    print("\ncorr(residual winners K=6 net, MSCI momentum) =",
          round(S["Residual winners K=6 (net)"].corr(S[MOM]), 3))
    half = len(S) // 2
    for lab, part in (("first half", S.iloc[:half]), ("second half", S.iloc[half:])):
        a = ((1 + part).prod() ** (12 / len(part)) - 1) * 100
        print(f"{lab} {part.index[0]}..{part.index[-1]} ann return %:",
              a[["Residual winners K=3 (net)", "50/50 K=3", "50/50 K=6", MOM, BENCH]].round(1).to_dict())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    style = {"Residual winners K=3 (net)": ("C0", 1.3, "-"), MOM: ("C1", 1.3, "-"),
             "50/50 K=3": ("C2", 2.4, "-"), BENCH: ("C3", 1.5, ":")}
    t = S.index.to_timestamp(how="end")
    for name, (c, lw, ls) in style.items():
        cum = (1 + S[name]).cumprod()
        axes[0].plot(t, cum, label=name, color=c, lw=lw, ls=ls)
        axes[1].plot(t, (cum / cum.cummax() - 1) * 100, label=name, color=c, lw=lw, ls=ls)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title}: growth of 1 ({ccy}, log scale)")
    axes[1].set_title("Drawdown from previous peak, %")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "combine_5050.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    for k in ([a.lower() for a in sys.argv[1:]] or ["tw", "us"]):
        run(k)
