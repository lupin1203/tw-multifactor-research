"""Taiwan Fama-French factors + residual momentum (Blitz, Huij, Martens 2011),
following section 2.2 of 林信助 "國際股市之殘差動能" (NSC 101-2410-H-004-057).

Input : data/ from fetch_tw.py (FinMind; includes delisted stocks)
Output: output/  tw_panel.parquet, tw_factors.csv, strategy_returns.csv, summary.csv, chart

Steps
  1. Monthly total returns from daily closes, corrected on ex-dividend / capital-reduction /
     split days with the exchange's reference prices. Clipped to [-95%, +100%].
  2. Factors: June-end size (top 90% of cumulative market cap = Big), B/M = 1 / official
     P/B at Dec t-1 with 30/70 breakpoints, 2x3 value-weighted portfolios, July t..June t+1.
  3. Each month: 36-month rolling 3-factor regression per stock; score = sum of residuals
     t-12..t-2 divided by their standard deviation (same window, as in Blitz et al.).
  4. Deciles, equal-weighted; long top 10%, short bottom 10%; hold K = 3, 6, 9, 12 months
     with overlapping cohorts. Mean monthly return and Newey-West t.
  5. Same thing for plain 12-1 total-return momentum on the same stocks.

Run: python build_tw.py [--start 2016-01] [--end 2026-10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "output"

BIG_CAP_SHARE = 0.90          # FF (2012): stocks making up the top 90% of market cap are Big
BM_BREAKS_FROM_BIG = True     # FF (2012): B/M breakpoints taken from Big stocks
RET_CLIP = (-0.95, 1.00)      # paper, following Ince & Porter (2006)
WINDOW, FORM_LAG = 36, 11     # 36-month regression; residuals t-12..t-2 = 11 months
HOLD = (3, 6, 9, 12)
PAPER_TW = {3: (0.805, 1.39), 6: (0.605, 1.19), 9: (0.420, 0.91), 12: (0.273, 0.65)}


# ------------------------------------------------------------------------ raw data
def read_all(dataset: str, cols: list[str]) -> pd.DataFrame:
    out = []
    for f in sorted((DATA / dataset).glob("*.parquet")):
        d = pd.read_parquet(f)
        if len(d):
            out.append(d[cols])
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def bulk_events() -> pd.DataFrame:
    """Split / par-value-change reference prices (small tables, fetched once)."""
    f = DATA / "split_events.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import os
    tok = os.environ.get("FINMIND_TOKEN", "")
    url = "https://api.finmindtrade.com/api/v4/data"
    rows = []
    for ds, b, a in [("TaiwanStockSplitPrice", "before_price", "after_price"),
                     ("TaiwanStockParValueChange", "before_close", "after_ref_close")]:
        j = requests.get(url, params={"dataset": ds, "start_date": "2005-01-01", "token": tok},
                         timeout=60).json()
        d = pd.DataFrame(j.get("data", []))
        if len(d):
            rows.append(d[["date", "stock_id", b, a]].set_axis(
                ["date", "stock_id", "before", "after"], axis=1))
    ev = pd.concat(rows, ignore_index=True).drop_duplicates(["date", "stock_id"])
    ev.to_parquet(f, index=False)
    return ev


def risk_free() -> pd.Series:
    """Monthly rf = five major banks' average 1-year time-deposit rate / 12 (CBC table EG2BM01)."""
    f = DATA / "rf_cbc_EG2BM01.json"
    if not f.exists():
        r = requests.get("https://cpx.cbc.gov.tw/API/DataAPI/Get",
                         params={"FileName": "EG2BM01"}, timeout=60)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
    rows = json.loads(f.read_text(encoding="utf-8"))["data"]["dataSets"]
    s = pd.Series({pd.Period(r[0].replace("M", "-"), "M"): pd.to_numeric(r[2], errors="coerce")
                   for r in rows}).ffill()
    return (s / 100 / 12).rename("rf")


def build_panel() -> pd.DataFrame:
    """stock-month panel: ret (total return, clipped), me (month-end market cap), pbr."""
    px = read_all("TaiwanStockPrice", ["date", "stock_id", "close"])
    px = px[px["close"] > 0].sort_values(["stock_id", "date"])

    # Price-adjustment factor on event days: prev close / reference price.
    div = read_all("TaiwanStockDividendResult", ["date", "stock_id", "before_price", "after_price"])
    div = div.set_axis(["date", "stock_id", "before", "after"], axis=1)
    cr = read_all("TaiwanStockCapitalReductionReferencePrice",
                  ["date", "stock_id", "ClosingPriceonTheLastTradingDay",
                   "PostReductionReferencePrice"])
    cr = cr.set_axis(["date", "stock_id", "before", "after"], axis=1)
    sp = bulk_events()
    sp["date"] = pd.to_datetime(sp["date"])
    ev = pd.concat([div, cr, sp], ignore_index=True)
    ev = ev[(ev["before"] > 0) & (ev["after"] > 0)]
    ev["adj"] = ev["before"] / ev["after"]
    ev = ev.groupby(["stock_id", "date"], as_index=False)["adj"].prod()

    px = px.merge(ev, on=["stock_id", "date"], how="left")
    px["adj"] = px["adj"].fillna(1.0)
    prev = px.groupby("stock_id")["close"].shift()
    px["r"] = px["close"] / prev * px["adj"] - 1
    px["month"] = px["date"].dt.to_period("M")
    px["lr"] = np.log1p(px["r"].clip(lower=-0.999))
    g = px.groupby(["stock_id", "month"])
    m = pd.DataFrame({"ret": np.expm1(g["lr"].sum(min_count=1)), "close": g["close"].last()})
    m = m.reset_index()
    # return only valid if the stock also traded in the previous calendar month
    mnum = m["month"].dt.year * 12 + m["month"].dt.month
    gap = mnum - mnum.groupby(m["stock_id"]).shift()
    m["ret"] = m["ret"].where(gap == 1).clip(*RET_CLIP)

    sh = read_all("TaiwanStockShareholding", ["date", "stock_id", "NumberOfSharesIssued"])
    sh = sh[sh["NumberOfSharesIssued"] > 0]
    sh["month"] = sh["date"].dt.to_period("M")
    sh = sh.sort_values("date").groupby(["stock_id", "month"])["NumberOfSharesIssued"].last()
    m = m.merge(sh.rename("shares").reset_index(), on=["stock_id", "month"], how="left")
    # Companies FinMind has no share count for at all (delisted TPEx companies): shares from the
    # exchange's monthly volume / turnover rate (fetch_tpex_shares.py). Never used for a stock
    # that has FinMind figures.
    fill_file = DATA / "shares_fill" / "tpex_turnover_shares.parquet"
    if fill_file.exists():
        fl = pd.read_parquet(fill_file)
        fl = fl[~fl["stale"]]          # before 2009-11 the page's rate rests on a later share count: not used
        fl["month"] = pd.PeriodIndex(fl["month"], freq="M")
        none = ~m["stock_id"].isin(sh.index.get_level_values("stock_id").unique())
        add = m.loc[none, ["stock_id", "month"]].merge(fl[["stock_id", "month", "shares"]], on=["stock_id", "month"], how="left")
        m.loc[none, "shares"] = add["shares"].to_numpy()
        print(f"  shares from turnover rates: {int(add['shares'].notna().sum())} stock-months, "
              f"{add.loc[add['shares'].notna(), 'stock_id'].nunique()} stocks")
    m = m.sort_values(["stock_id", "month"])
    m["shares"] = m.groupby("stock_id")["shares"].ffill(limit=6)
    m["me"] = m["shares"] * m["close"]

    pb = read_all("TaiwanStockPER", ["date", "stock_id", "PBR"])
    pb["month"] = pb["date"].dt.to_period("M")
    pb = pb.sort_values("date").groupby(["stock_id", "month"])["PBR"].last()
    m = m.merge(pb.rename("pbr").reset_index(), on=["stock_id", "month"], how="left")
    m["me_lag"] = m.groupby("stock_id")["me"].shift().where(gap == 1)
    return m[["stock_id", "month", "ret", "me", "me_lag", "pbr"]]


# -------------------------------------------------------------------------- factors
def build_factors(panel: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    wide_me = panel.pivot(index="month", columns="stock_id", values="me")
    wide_pb = panel.pivot(index="month", columns="stock_id", values="pbr")
    ports = []
    for t in range(wide_me.index.min().year + 1, wide_me.index.max().year + 1):
        jun, dec = pd.Period(f"{t}-06", "M"), pd.Period(f"{t - 1}-12", "M")
        if jun not in wide_me.index or dec not in wide_pb.index:
            continue
        d = pd.DataFrame({"me": wide_me.loc[jun], "pbr": wide_pb.loc[dec]}).dropna()
        d = d[(d["me"] > 0) & (d["pbr"] > 0)]
        if len(d) < 60:
            continue
        d["bm"] = 1 / d["pbr"]
        d = d.sort_values("me", ascending=False)
        d["size"] = np.where(d["me"].cumsum() / d["me"].sum() <= BIG_CAP_SHARE, "B", "S")
        ref = d[d["size"] == "B"] if BM_BREAKS_FROM_BIG else d
        lo, hi = ref["bm"].quantile([0.3, 0.7])
        d["bmg"] = np.select([d["bm"] <= lo, d["bm"] > hi], ["L", "H"], "M")
        d["year"] = t
        ports.append(d[["year", "size", "bmg"]].reset_index())
        print(f"  June {t}: {len(d)} stocks, {int((d['size'] == 'B').sum())} big, "
              f"B/M breaks {lo:.2f}/{hi:.2f}")
    ports = pd.concat(ports, ignore_index=True)

    p = panel.dropna(subset=["ret", "me_lag"])
    p = p[p["me_lag"] > 0].copy()
    p["year"] = np.where(p["month"].dt.month >= 7, p["month"].dt.year, p["month"].dt.year - 1)
    vw = lambda x: np.average(x["ret"], weights=x["me_lag"])
    mkt = p.groupby("month").apply(vw)
    q = p.merge(ports, on=["stock_id", "year"])
    q["port"] = q["size"] + "/" + q["bmg"]
    six = q.groupby(["month", "port"]).apply(vw).unstack()
    f = pd.DataFrame(index=six.index)
    f["RMRF"] = mkt.reindex(f.index) - rf.reindex(f.index)
    f["SMB"] = six[["S/L", "S/M", "S/H"]].mean(1) - six[["B/L", "B/M", "B/H"]].mean(1)
    f["HML"] = six[["S/H", "B/H"]].mean(1) - six[["S/L", "B/L"]].mean(1)
    f["RF"] = rf.reindex(f.index)
    return f.dropna()


# ------------------------------------------------------------------------- strategy
def signals(panel: pd.DataFrame, fac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual-momentum and total-momentum scores at each formation month-end."""
    ret = panel.pivot(index="month", columns="stock_id", values="ret")
    ret = ret.reindex(pd.period_range(ret.index.min(), ret.index.max(), freq="M"))
    months = [m for m in ret.index if m in fac.index]
    res, tot = {}, {}
    for i in range(WINDOW - 1, len(months)):
        win = months[i - WINDOW + 1:i + 1]
        if (win[-1] - win[0]).n != WINDOW - 1:
            continue
        Y = ret.loc[win].sub(fac.loc[win, "RF"], axis=0)
        Y = Y.loc[:, Y.notna().all()]                       # full 36 months required
        if Y.shape[1] < 30:                                 # paper: at least 30 stocks
            continue
        X = np.column_stack([np.ones(WINDOW), fac.loc[win, ["RMRF", "SMB", "HML"]].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, Y.to_numpy(), rcond=None)
        e = Y.to_numpy() - X @ beta
        e = e[-(FORM_LAG + 1):-1]                           # t-12 .. t-2 (skip latest month)
        res[win[-1]] = pd.Series(e.sum(0) / e.std(0, ddof=1), index=Y.columns)
        r = ret.loc[win, Y.columns].to_numpy()[-(FORM_LAG + 1):-1]
        tot[win[-1]] = pd.Series(np.expm1(np.log1p(r).sum(0)), index=Y.columns)
    return pd.DataFrame(res).T, pd.DataFrame(tot).T


def strategy(score: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Winner-minus-loser monthly returns for each K, overlapping cohorts."""
    ret = panel.pivot(index="month", columns="stock_id", values="ret")
    cohort = {}                                  # formation month -> (winners, losers)
    for t, s in score.iterrows():
        s = s.dropna()
        n = max(int(round(len(s) * 0.10)), 1)
        r = s.sort_values()
        cohort[t] = (r.index[-n:], r.index[:n])
    out = {}
    for m in ret.index:
        row = {}
        for K in HOLD:
            legs = []
            for k in range(1, K + 1):
                c = cohort.get(m - k)
                if c is None:
                    break
                w, l = ret.loc[m, c[0]].mean(), ret.loc[m, c[1]].mean()
                legs.append((w, l))
            if len(legs) == K:                   # only months with all K cohorts live
                w, l = np.nanmean([x[0] for x in legs]), np.nanmean([x[1] for x in legs])
                row[f"W{K}"], row[f"L{K}"], row[f"WML{K}"] = w, l, w - l
        if row:
            out[m] = row
    return pd.DataFrame(out).T


def nw_t(x: pd.Series, lags: int) -> float:
    x = x.dropna().to_numpy()
    n, u = len(x), x - x.mean()
    v = (u @ u) / n
    for j in range(1, lags + 1):
        v += 2 * (1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n
    return x.mean() / np.sqrt(v / n)


def summarize(res: pd.DataFrame, tot: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sl = lambda d: d[(d.index >= pd.Period(start, "M")) & (d.index <= pd.Period(end, "M"))]
    res, tot = sl(res), sl(tot)
    rows = []
    for K in HOLD:
        a, b = res[f"WML{K}"], tot[f"WML{K}"]
        rows.append({
            "K": K, "months": a.notna().sum(),
            "residual_%/mo": a.mean() * 100, "residual_t": nw_t(a, K),
            "residual_winner_%": res[f"W{K}"].mean() * 100,
            "residual_loser_%": res[f"L{K}"].mean() * 100,
            "residual_std_%": a.std() * 100,
            "total_mom_%/mo": b.mean() * 100, "total_mom_t": nw_t(b, K),
            "total_mom_std_%": b.std() * 100,
            "paper_%/mo": PAPER_TW[K][0], "paper_t": PAPER_TW[K][1],
        })
    return pd.DataFrame(rows).set_index("K")


def plot(res: pd.DataFrame, tot: pd.DataFrame, start: str, end: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = lambda d: d[(d.index >= pd.Period(start, "M")) & (d.index <= pd.Period(end, "M"))]
    res, tot = sl(res), sl(tot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, d, name in [(axes[0], res, "Residual momentum"), (axes[1], tot, "Total-return momentum")]:
        for K in HOLD:
            s = d[f"WML{K}"].dropna()
            ax.plot(s.index.to_timestamp(), (1 + s).cumprod(), label=f"K={K}")
        ax.set_title(f"Taiwan {name}: winner minus loser, growth of 1")
        ax.axhline(1, color="grey", lw=0.6)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "tw_resmom.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)

    rf = risk_free()
    panel = build_panel()
    last = panel["month"].max()
    if panel.loc[panel["month"] == last, "stock_id"].nunique() < \
            0.9 * panel.loc[panel["month"] == last - 1, "stock_id"].nunique():
        last -= 1
    end = a.end or str(last - 1 if pd.Timestamp.today().to_period("M") == last else last)
    panel.to_parquet(OUT / "tw_panel.parquet", index=False)
    print(f"panel: {panel['stock_id'].nunique()} stocks, {panel['month'].min()}..{panel['month'].max()}")

    fac = build_factors(panel, rf)
    fac.to_csv(OUT / "tw_factors.csv")
    print(f"factors {fac.index.min()}..{fac.index.max()}  mean %/mo:\n{(fac.mean() * 100).round(3)}")

    res_score, tot_score = signals(panel, fac)
    print(f"signals {res_score.index.min()}..{res_score.index.max()}, "
          f"avg {res_score.notna().sum(1).mean():.0f} stocks per month")
    res, tot = strategy(res_score, panel), strategy(tot_score, panel)
    res.add_prefix("res_").join(tot.add_prefix("tot_")).to_csv(OUT / "strategy_returns.csv")
    tbl = summarize(res, tot, a.start, end)
    tbl.to_csv(OUT / "summary.csv")
    print(f"\n=== Taiwan, {a.start} .. {end} (TWD; paper = 1987/09-2009/12, USD) ===\n{tbl.T}")
    plot(res, tot, a.start, end)


if __name__ == "__main__":
    main()
