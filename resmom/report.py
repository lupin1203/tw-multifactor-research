"""Full residual-momentum report for one country: tables and a cumulative-return chart.

Academic (matching the paper)
  table2_decile.csv      WML mean %/month and Newey-West t, K = 3/6/9/12, residual vs total
  table4_30pct.csv       same with 30% winners / losers
  table6_subperiods.csv  the reported period cut into two halves
Trading (not in the paper)
  trading.csv            annualised return / volatility / Sharpe, max drawdown and dates,
                         drawdown inside 2020 and 2022, turnover, net-of-cost returns
  long_only.csv          winner portfolio alone against the market and a benchmark ETF
  value_weighted.csv     value-weighted portfolios (is the profit a small-cap effect?)
  size_floor.csv         equal-weighted deciles using only stocks above a market-cap floor
  monthly_returns.csv    every monthly series
  cumulative.png         residual momentum, total momentum, market

Command line (one country):
  python report.py --name Taiwan --returns ret.parquet --factors fac.csv --out out_dir
         [--bench bench.csv --bench-name 0050] [--cost-buy 0.001425 --cost-sell 0.004425]
         [--start 2016-01 --end 2026-10]
  returns: id, month, ret [, me_lag]     factors: month, RMRF, SMB, HML [, RF]
  bench:   month, ret   (total return of an index fund, decimals)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import resmom as rm

EVENTS = {"dd_2020_%": ("2020-01", "2020-12"), "dd_2022_%": ("2022-01", "2022-12")}


def _events(x: pd.Series) -> dict:
    """Worst peak-to-trough loss inside each event window."""
    return {k: (rm.max_drawdown(rm.cut(x, a, b))[0] * 100 if len(rm.cut(x, a, b).dropna()) else np.nan)
            for k, (a, b) in EVENTS.items()}


def run_report(name: str, returns: pd.DataFrame, factors: pd.DataFrame, out_dir,
               bench: pd.Series | None = None, bench_name: str = "benchmark",
               cost_buy: float = 0.0, cost_sell: float = 0.0,
               start: str | None = None, end: str | None = None,
               hold=rm.HOLD, chart_k: int = 6, verbose: bool = True,
               cap_floors: dict[str, float] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ret, fac = rm.to_wide(returns), rm.prep_factors(factors)
    cap = rm.to_wide(returns, "me_lag") if "me_lag" in returns.columns else None
    rs, ts = rm.scores(ret, fac)

    P = {("res", "ew10"): rm.wml(rs, ret, hold), ("tot", "ew10"): rm.wml(ts, ret, hold),
         ("res", "ew30"): rm.wml(rs, ret, hold, pct=0.30), ("tot", "ew30"): rm.wml(ts, ret, hold, pct=0.30)}
    if cap is not None:
        for kind, sc in (("res", rs), ("tot", ts)):
            P[(kind, "vw10")] = rm.wml(sc, ret, hold, weight=cap)
            P[(kind, "vw30")] = rm.wml(sc, ret, hold, pct=0.30, weight=cap)

    floors = {}
    if cap is not None and cap_floors:
        for lab, v in cap_floors.items():
            ok = cap.reindex_like(rs) >= v              # cap known before the holding month
            floors[lab] = (rm.wml(rs.where(ok), ret, hold), rm.wml(ts.where(ok), ret, hold),
                           float(ok.loc[ok.index >= pd.Period(start or "1900-01", "M")].sum(1).mean()))

    # reported period: by default every month in which all K are available
    full = P[("res", "ew10")].dropna(subset=[f"WML{max(hold)}"]).index
    start, end = start or str(full.min()), end or str(full.max())
    P = {k: rm.cut(v, start, end) for k, v in P.items()}
    months = P[("res", "ew10")].index
    mid = months[len(months) // 2]
    mkt = rm.cut(fac["RMRF"] + fac["RF"], start, end)
    rf = rm.cut(fac["RF"], start, end)
    if bench is not None:
        bench = bench.copy()
        bench.index = pd.PeriodIndex(bench.index.astype(str), freq="M")
        bench = rm.cut(bench, start, end)

    T = {}
    T["table2_decile"] = rm.summarize(P[("res", "ew10")], P[("tot", "ew10")], hold)
    T["table4_30pct"] = rm.summarize(P[("res", "ew30")], P[("tot", "ew30")], hold)
    T["table6_subperiods"] = pd.concat({
        f"{start}..{mid - 1}": rm.summarize(P[("res", "ew10")], P[("tot", "ew10")], hold, end=str(mid - 1)),
        f"{mid}..{end}": rm.summarize(P[("res", "ew10")], P[("tot", "ew10")], hold, start=str(mid)),
    }, names=["period"])
    if cap is not None:
        T["value_weighted"] = pd.concat({
            "10% VW": rm.summarize(P[("res", "vw10")], P[("tot", "vw10")], hold),
            "30% VW": rm.summarize(P[("res", "vw30")], P[("tot", "vw30")], hold),
        }, names=["portfolio"])

    if floors:
        d = {}
        for lab, (a, b, cnt) in floors.items():
            x = rm.summarize(a, b, hold, start, end)
            x.insert(1, "avg_stocks", round(cnt))
            d[lab] = x
        T["size_floor"] = pd.concat(d, names=["universe"])

    # ------------------------------------------------------------- trading view
    rows, lo_rows = [], []
    label = {"res": "residual", "tot": "total"}
    for kind in ("res", "tot"):
        p = P[(kind, "ew10")]
        for K in hold:
            g = p[f"WML{K}"].dropna()
            n = rm.net_of_cost(p, K, cost_buy, cost_sell)
            pg, pn = rm.perf(g), rm.perf(n["WML"])
            to = (p[f"TO_W{K}"].mean() + p[f"TO_L{K}"].mean()) / 2
            rows.append({"strategy": label[kind], "K": K,
                         "mean_%/mo": g.mean() * 100, "t": rm.nw_t(g, K),
                         "ann_return_%": pg["ann_return_%"], "ann_vol_%": pg["ann_vol_%"],
                         "sharpe": pg["sharpe"], "max_dd_%": pg["max_dd_%"],
                         "dd_peak": pg["dd_peak"], "dd_trough": pg["dd_trough"], **_events(g),
                         "turnover_%/mo_per_leg": to * 100,
                         "cost_%/mo": (g.reindex(n.index) - n["WML"]).mean() * 100,
                         "net_mean_%/mo": n["WML"].mean() * 100, "net_t": rm.nw_t(n["WML"], K),
                         "net_ann_return_%": pn["ann_return_%"], "net_sharpe": pn["sharpe"],
                         "net_max_dd_%": pn["max_dd_%"]})
            w = p[f"W{K}"].dropna()
            ref = bench if bench is not None else mkt
            for tag, x in (("gross", w), ("net", n["W"])):
                px = rm.perf(x, rf)
                ex = (x - ref.reindex(x.index)).dropna()
                lo_rows.append({"portfolio": f"{label[kind]} winners K={K} ({tag})", **px, **_events(x),
                                f"excess_vs_{bench_name if bench is not None else 'market'}_%/mo": ex.mean() * 100,
                                "excess_t": rm.nw_t(ex, K)})
    T["trading"] = pd.DataFrame(rows).set_index(["strategy", "K"])
    base = [("market (value-weighted, all stocks)", mkt)]
    if bench is not None:
        base.append((bench_name, bench))
    for nm, x in base:
        lo_rows.append({"portfolio": nm, **rm.perf(x, rf), **_events(x)})
    T["long_only"] = pd.DataFrame(lo_rows).set_index("portfolio")

    for k, t in T.items():
        t.to_csv(out_dir / f"{k}.csv", encoding="utf-8-sig")
    monthly = pd.concat([v.add_prefix(f"{a}_{b}_") for (a, b), v in P.items()], axis=1)
    monthly["market"], monthly["rf"] = mkt, rf
    if bench is not None:
        monthly[bench_name] = bench
    monthly.to_csv(out_dir / "monthly_returns.csv")
    _plot(name, P, mkt, bench, bench_name, cost_buy + cost_sell, chart_k, out_dir / "cumulative.png")

    if verbose:
        pd.set_option("display.width", 260, "display.max_columns", 40,
                      "display.float_format", "{:.3f}".format)
        print(f"\n################ {name}: {start} .. {end} ({len(months)} months) ################")
        for k, t in T.items():
            print(f"\n--- {k} ---\n{t.to_string()}")
    return {"tables": T, "monthly": monthly, "start": start, "end": end}


def _plot(name, P, mkt, bench, bench_name, roundtrip, K, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, tot = P[("res", "ew10")], P[("tot", "ew10")]

    def line(ax, x, lab, **kw):
        x = x.dropna()
        ax.plot(x.index.to_timestamp(how="end"), (1 + x).cumprod(), label=lab, **kw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    line(ax, res[f"WML{K}"], "Residual momentum (long-short)", color="C0", lw=2)
    line(ax, tot[f"WML{K}"], "Total-return momentum (long-short)", color="C1", lw=2)
    line(ax, rm.net_of_cost(res, K, roundtrip, 0)["WML"], "Residual momentum, after costs", color="C0", lw=1, ls="--")
    line(ax, mkt, "Market (value-weighted)", color="0.35", lw=1.5)
    ax.set_title(f"{name}: winner minus loser vs market, K={K}, growth of 1")
    ax = axes[1]
    line(ax, res[f"W{K}"], "Residual-momentum winners", color="C0", lw=2)
    line(ax, tot[f"W{K}"], "Total-momentum winners", color="C1", lw=2)
    line(ax, mkt, "Market (value-weighted)", color="0.35", lw=1.5)
    if bench is not None:
        line(ax, bench, bench_name, color="C3", lw=1.5, ls=":")
    ax.set_title(f"{name}: long-only winners vs market, K={K}, growth of 1")
    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(1, color="grey", lw=0.6)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Residual momentum report for one country")
    ap.add_argument("--name", required=True)
    ap.add_argument("--returns", required=True)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--bench-name", default="benchmark")
    ap.add_argument("--cost-buy", type=float, default=0.0)
    ap.add_argument("--cost-sell", type=float, default=0.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--chart-k", type=int, default=6)
    a = ap.parse_args()
    bench = None
    if a.bench:
        b = _read(a.bench)
        bench = b.set_index("month")["ret"]
    run_report(a.name, _read(a.returns).rename(columns={a.id_col: "id"}), _read(a.factors), a.out,
               bench, a.bench_name, a.cost_buy, a.cost_sell, a.start, a.end, chart_k=a.chart_k)


if __name__ == "__main__":
    main()
