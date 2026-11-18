"""Taiwan: 100% 0050 plus a residual-momentum long-short overlay.

portfolio return = 0050 + X * (winners - losers, after costs)

Overlay        top 10% minus bottom 10% by residual-momentum score, equal weight,
               K = 3 overlapping cohorts. X = size of each leg as a share of capital.
Universes      all stocks; market cap >= NT$3bn; >= NT$10bn (small caps are hard to
               borrow, so the floors show what survives among shortable names).
Costs          both legs: fee 0.1425% each way + 0.3% tax on sells, on turnover;
               short leg: stock-borrowing fee, 3% a year assumed (0% and 6% shown).
               Returns are total returns, so the short leg pays the dividends.
Not modelled   margin / collateral for the short leg and financing of the long leg
               (the overlay is treated as self-financing), short-sale bans, recalls.
Pass line      (same as before) alpha t > 2 after costs and alpha > 0 in both halves.

Run: python overlay_ls_tw.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import resmom as rm
from overlay_tw import alpha_stats, bench, EVENTS, FEE, TAX_STOCK

HERE = Path(__file__).parent
OUT = HERE / "output/taiwan"
START, END = "2016-01", "2026-10"
K = 3
BORROW = 0.03                       # per year, on the short notional
SIZES = (0.3, 0.5, 1.0)
FLOORS = {"all stocks": 0.0, "cap >= NT$3bn": 3e9, "cap >= NT$10bn": 1e10}


def main() -> None:
    p = pd.read_parquet(HERE.parent / "resmom_tw/output/tw_panel.parquet").rename(columns={"stock_id": "id"})
    fac = rm.prep_factors(pd.read_csv(HERE.parent / "resmom_tw/output/tw_factors.csv"))
    ret, cap = rm.to_wide(p), rm.to_wide(p, "me_lag")
    rs, _ = rm.scores(ret, fac)
    b0 = rm.cut(bench("0050_TW"), START, END)
    rf = fac["RF"].reindex(pd.period_range("2009-01", END, freq="M")).ffill()
    rt = 2 * FEE + TAX_STOCK

    ls, info = {}, {}
    for lab, v in FLOORS.items():
        sc = rs.where(cap.reindex_like(rs) >= v) if v else rs
        w = rm.wml(sc, ret, hold=(K,))
        to = (w[f"TO_W{K}"] + w[f"TO_L{K}"]).fillna((w[f"TO_W{K}"] + w[f"TO_L{K}"]).mean())
        gross = rm.cut(w[f"WML{K}"], START, END)
        trade = rm.cut(to * rt, START, END)
        ls[lab] = {"gross": gross, "trade": trade}
        n = rm.cut(sc, START, END).notna().sum(1).mean()
        info[lab] = {"stocks": round(n), "per_leg": round(n * 0.1),
                     "gross_%/mo": gross.mean() * 100, "gross_t": rm.nw_t(gross, K),
                     "turnover_%/mo_per_leg": rm.cut(to, START, END).mean() * 50,
                     "trading_cost_%/mo": trade.mean() * 100,
                     **{f"net_%/mo_borrow_{int(b * 100)}%": (gross - trade - b / 12).mean() * 100 for b in (0, 0.03, 0.06)},
                     **{f"net_t_borrow_{int(b * 100)}%": rm.nw_t(gross - trade - b / 12, K) for b in (0, 0.03, 0.06)},
                     "ann_vol_%": gross.std() * np.sqrt(12) * 100,
                     "corr_0050": gross.corr(b0)}
    info = pd.DataFrame(info).T

    S = {}
    for lab in FLOORS:
        net = ls[lab]["gross"] - ls[lab]["trade"] - BORROW / 12
        for x in SIZES:
            S[f"0050 + {int(x * 100)}% long-short ({lab})"] = b0 + x * net
        S[f"long-short alone ({lab}, net)"] = net
    S["0050"] = b0
    S = pd.DataFrame(S)
    mid = S.index[len(S) // 2]

    rows = {}
    for name, x in S.items():
        alone = name.startswith("long-short")
        pf = rm.perf(x, None if alone else rf)
        is_b = name == "0050"
        y = x + rf.reindex(x.index) if alone else x            # long-short is already an excess return
        a, at, bt = alpha_stats(y, S["0050"], rf)
        a1, t1, _ = alpha_stats(y.loc[:mid - 1], S["0050"], rf)
        a2, t2, _ = alpha_stats(y.loc[mid:], S["0050"], rf)
        ex = x - S["0050"]
        rows[name] = {
            "ann_return_%": pf["ann_return_%"], "ann_vol_%": pf["ann_vol_%"], "sharpe": pf["sharpe"],
            "max_dd_%": pf["max_dd_%"], "dd_peak": pf["dd_peak"], "dd_trough": pf["dd_trough"],
            **{k: rm.max_drawdown(rm.cut(x, s, e))[0] * 100 for k, (s, e) in EVENTS.items()},
            "beta_0050": bt,
            "excess_%/mo": np.nan if is_b or alone else ex.mean() * 100,
            "excess_t": np.nan if is_b or alone else rm.nw_t(ex, 6),
            "alpha_%/mo": np.nan if is_b else a * 100, "alpha_t": np.nan if is_b else at,
            "alpha_1st_half_%/mo": np.nan if is_b else a1 * 100, "alpha_1st_t": np.nan if is_b else t1,
            "alpha_2nd_half_%/mo": np.nan if is_b else a2 * 100, "alpha_2nd_t": np.nan if is_b else t2,
        }
        if not is_b and not alone:
            rows[name]["pass"] = bool(at > 2 and a1 > 0 and a2 > 0)
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(OUT / "overlay_ls.csv", encoding="utf-8-sig")
    info.to_csv(OUT / "overlay_ls_legs.csv", encoding="utf-8-sig")
    S.to_csv(OUT / "overlay_ls_monthly.csv")

    pd.set_option("display.width", 330, "display.max_columns", 40, "display.float_format", "{:.2f}".format)
    print(f"Taiwan {START}..{END} ({len(S)} months); halves split at {mid}; borrow fee {BORROW:.0%}/yr")
    print(f"\n--- long-short leg ---\n{info.T.to_string()}")
    print(f"\n--- combined ---\n{tbl.to_string()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    style = {"0050 + 50% long-short (cap >= NT$3bn)": ("C2", 2.4, "-"),
             "0050 + 100% long-short (cap >= NT$3bn)": ("C4", 1.3, "-"),
             "0050 + 50% long-short (all stocks)": ("C0", 1.3, "-"),
             "long-short alone (cap >= NT$3bn, net)": ("C1", 1.0, "--"),
             "0050": ("C3", 1.6, ":")}
    t = S.index.to_timestamp(how="end")
    for name, (c, lw, ls_) in style.items():
        cum = (1 + S[name]).cumprod()
        axes[0].plot(t, cum, label=name, color=c, lw=lw, ls=ls_)
        axes[1].plot(t, (cum / cum.cummax() - 1) * 100, label=name, color=c, lw=lw, ls=ls_)
    axes[0].set_yscale("log")
    axes[0].set_title("Taiwan: growth of 1 after costs (TWD, log scale)")
    axes[1].set_title("Drawdown from previous peak, %")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_ls.png", dpi=130)


if __name__ == "__main__":
    main()
