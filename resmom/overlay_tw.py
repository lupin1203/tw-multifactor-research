"""Taiwan: score-weighted residual-momentum winners combined with 00631L (2x 0050)
or with plain 0050. Goal: beat 0050 at about the same market exposure.

Stock sleeve   top 10% by residual-momentum score, weight proportional to the score,
               3% cap per stock, K = 3 overlapping cohorts, after fee 0.1425% each way
               and 0.3% tax on sells.
ETF sleeve     00631L or 0050, actual monthly total returns (Yahoo, dividend-adjusted).
Mixes          'beta-matched': ETF weight chosen each quarter so that the estimated
               exposure to 0050 is 1.0, using only the previous 36 months;
               fixed 30/70 and 50/50. Rebalanced at the start of each quarter, with
               fee 0.1425% each way plus tax on sells (ETF 0.1%, stocks 0.3%).
Pass line      (set before running) beta-matched mix: alpha t > 2 after costs and
               alpha positive in both halves of the period.

Run: python overlay_tw.py     (needs ../resmom_tw/output and data/bench_*.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import resmom as rm

HERE = Path(__file__).parent
OUT = HERE / "output/taiwan"
START, END = "2016-01", "2026-10"
K, CAP = 3, 0.03
FEE, TAX_STOCK, TAX_ETF = 0.001425, 0.003, 0.001
ETF_FROM = "2015-08"                # Yahoo's 00631L history is unreliable before this
EVENTS = {"dd_2020_%": ("2020-01", "2020-12"), "dd_2022_%": ("2022-01", "2022-12")}


def bench(name: str) -> pd.Series:
    s = pd.read_csv(HERE / f"data/bench_{name}.csv", index_col=0)["ret"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def beta(y: pd.Series, x: pd.Series) -> float:
    d = pd.concat([y, x], axis=1).dropna()
    return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())


def alpha_stats(y: pd.Series, mkt: pd.Series, rf: pd.Series, lags: int = 6) -> tuple[float, float, float]:
    """Regress excess return on the market's excess return: (alpha per month, its
    Newey-West t, beta)."""
    d = pd.concat([y - rf, mkt - rf], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy()])
    yv = d.iloc[:, 0].to_numpy()
    b = np.linalg.solve(X.T @ X, X.T @ yv)
    u = yv - X @ b
    g = X * u[:, None]
    S = g.T @ g
    for j in range(1, lags + 1):
        G = g[j:].T @ g[:-j]
        S += (1 - j / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0])), float(b[1])


def mix(etf: pd.Series, stocks: pd.Series, target, tax_etf: float = TAX_ETF) -> tuple[pd.Series, pd.Series]:
    """Two-sleeve portfolio, weights drift within a quarter and are reset to
    target(month) at the start of Jan/Apr/Jul/Oct. Returns (net return, ETF weight)."""
    months = pd.period_range(START, END, freq="M")
    out, wts, w = {}, {}, None
    for m in months:
        cost = 0.0
        if w is None or m.month in (1, 4, 7, 10):
            tgt = float(target(m))
            if w is None:
                w = tgt                                   # initial purchase not charged
            else:
                tr = abs(tgt - w)
                buy_etf = tgt > w
                cost = tr * (FEE + (0 if buy_etf else tax_etf)) + tr * (FEE + (TAX_STOCK if buy_etf else 0))
                w = tgt
        a, b = etf[m], stocks[m]
        r = w * a + (1 - w) * b
        out[m], wts[m] = r - cost, w
        w = w * (1 + a) / (1 + r)
    return pd.Series(out), pd.Series(wts)


def main() -> None:
    p = pd.read_parquet(HERE.parent / "resmom_tw/output/tw_panel.parquet").rename(columns={"stock_id": "id"})
    fac = rm.prep_factors(pd.read_csv(HERE.parent / "resmom_tw/output/tw_factors.csv"))
    ret = rm.to_wide(p)
    rs, _ = rm.scores(ret, fac)
    rt = 2 * FEE + TAX_STOCK
    sleeves = {}
    for lab, kw in (("score-weighted", dict(score_weight=True, cap=CAP)), ("equal-weighted", {})):
        w = rm.wml(rs, ret, hold=(K,), **kw)
        sleeves[lab] = (w[f"W{K}"] - w[f"TO_W{K}"].fillna(w[f"TO_W{K}"].mean()) * rt,
                        w[f"W{K}"], w[f"TO_W{K}"])
    sw, ew = sleeves["score-weighted"][0], sleeves["equal-weighted"][0]
    b0, l2 = bench("0050_TW"), bench("00631L_TW").loc[ETF_FROM:]
    rf = fac["RF"].reindex(pd.period_range("2009-01", END, freq="M")).ffill()

    def matched(stocks: pd.Series):
        def f(m):                                         # uses months before m only
            h = slice(m - 36, m - 1)
            bs = beta(stocks.loc[h], b0.loc[h])
            hl = l2.loc[h].dropna()
            bl = beta(hl, b0.loc[h]) if len(hl) >= 12 else 2.0
            return float(np.clip((1 - bs) / (bl - bs), 0, 1))
        return f

    S, W = {}, {}
    for name, etf, stocks, tgt, tax in [
        ("beta-matched: 00631L + score-wtd winners", l2, sw, matched(sw), TAX_ETF),
        ("beta-matched: 00631L + equal-wtd winners", l2, ew, matched(ew), TAX_ETF),
        ("30/70: 00631L + score-wtd winners", l2, sw, lambda m: 0.30, TAX_ETF),
        ("50/50: 00631L + score-wtd winners", l2, sw, lambda m: 0.50, TAX_ETF),
        ("30/70: 0050 + score-wtd winners", b0, sw, lambda m: 0.30, TAX_ETF),
        ("50/50: 0050 + score-wtd winners", b0, sw, lambda m: 0.50, TAX_ETF),
    ]:
        S[name], W[name] = mix(etf, stocks, tgt, tax)
    S["score-wtd winners alone (net)"] = rm.cut(sw, START, END)
    S["equal-wtd winners alone (net)"] = rm.cut(ew, START, END)
    S["00631L"] = rm.cut(l2, START, END)
    S["0050"] = rm.cut(b0, START, END)
    S = pd.DataFrame(S)
    mid = S.index[len(S) // 2]

    rows = {}
    for name, x in S.items():
        pf = rm.perf(x, rf)
        ex = x - S["0050"]
        a, at, bt = alpha_stats(x, S["0050"], rf)
        a1, t1, _ = alpha_stats(x.loc[:mid - 1], S["0050"], rf)
        a2, t2, _ = alpha_stats(x.loc[mid:], S["0050"], rf)
        is_b = name == "0050"
        rows[name] = {
            "ann_return_%": pf["ann_return_%"], "ann_vol_%": pf["ann_vol_%"], "sharpe": pf["sharpe"],
            "max_dd_%": pf["max_dd_%"], "dd_peak": pf["dd_peak"], "dd_trough": pf["dd_trough"],
            **{k: rm.max_drawdown(rm.cut(x, s, e))[0] * 100 for k, (s, e) in EVENTS.items()},
            "beta_0050": bt,
            "excess_%/mo": np.nan if is_b else ex.mean() * 100, "excess_t": np.nan if is_b else rm.nw_t(ex, 6),
            "alpha_%/mo": np.nan if is_b else a * 100, "alpha_t": np.nan if is_b else at,
            "alpha_1st_half_%/mo": np.nan if is_b else a1 * 100, "alpha_1st_t": np.nan if is_b else t1,
            "alpha_2nd_half_%/mo": np.nan if is_b else a2 * 100, "alpha_2nd_t": np.nan if is_b else t2,
            "avg_etf_weight_%": W[name].mean() * 100 if name in W else np.nan,
        }
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(OUT / "overlay.csv", encoding="utf-8-sig")
    S.to_csv(OUT / "overlay_monthly.csv")
    pd.DataFrame(W).to_csv(OUT / "overlay_etf_weights.csv")

    pd.set_option("display.width", 320, "display.max_columns", 40, "display.float_format", "{:.2f}".format)
    print(f"Taiwan {START}..{END} ({len(S)} months); halves split at {mid}\n{tbl.to_string()}")
    for lab, (net, gross, to) in sleeves.items():
        g, n = rm.cut(gross, START, END), rm.cut(net, START, END)
        print(f"{lab} winners K={K}: gross {g.mean() * 100:.3f} %/mo, net {n.mean() * 100:.3f} %/mo, "
              f"turnover {rm.cut(to, START, END).mean() * 100:.1f} %/mo")
    wm = W["beta-matched: 00631L + score-wtd winners"]
    print(f"beta-matched 00631L weight: min {wm.min():.2f}, mean {wm.mean():.2f}, max {wm.max():.2f}")
    main_row = tbl.loc["beta-matched: 00631L + score-wtd winners"]
    ok = main_row["alpha_t"] > 2 and main_row["alpha_1st_half_%/mo"] > 0 and main_row["alpha_2nd_half_%/mo"] > 0
    print(f"PASS LINE (alpha t > 2 and alpha > 0 in both halves): {'PASSED' if ok else 'NOT PASSED'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    style = {"beta-matched: 00631L + score-wtd winners": ("C2", 2.4, "-"),
             "50/50: 00631L + score-wtd winners": ("C4", 1.3, "-"),
             "30/70: 0050 + score-wtd winners": ("C0", 1.3, "-"),
             "score-wtd winners alone (net)": ("C1", 1.0, "--"),
             "0050": ("C3", 1.6, ":")}
    t = S.index.to_timestamp(how="end")
    for name, (c, lw, ls) in style.items():
        cum = (1 + S[name]).cumprod()
        axes[0].plot(t, cum, label=name, color=c, lw=lw, ls=ls)
        axes[1].plot(t, (cum / cum.cummax() - 1) * 100, label=name, color=c, lw=lw, ls=ls)
    axes[0].set_yscale("log")
    axes[0].set_title("Taiwan: growth of 1 after costs (TWD, log scale)")
    axes[1].set_title("Drawdown from previous peak, %")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overlay.png", dpi=130)


if __name__ == "__main__":
    main()
