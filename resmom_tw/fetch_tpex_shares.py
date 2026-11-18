"""Shares outstanding for the companies FinMind has no share count for (delisted TPEx
companies: no TaiwanStockShareholding rows, hardly any statements), from the exchange's
own page 個股月成交資訊 (https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-month.html).

The page gives, per month, 成交仟股 (volume, thousand shares) and 成交週轉率(%), so
shares = volume / turnover. The printed monthly rate is the sum of the DAILY rates, each
cut to two decimals: measured on 30 stocks with a known share count it is short by
0.005 points per trading day (0.0049-0.0054 at every turnover level). So
    shares = volume / (printed rate + 0.005 x trading days)
and what is left is noise of about 0.0029 x sqrt(days) points. A month is used when that
noise is below MAX_ROUNDING of the rate; other months take the nearest usable month.
Checked against balance-sheet share capital: median ratio 1.000, 92% within 2%.

LIMIT: the exchange's rates for months before about 2009-11 were computed with a LATER
share count (roughly the one of late 2009; for companies gone before that, presumably
their last one), not the count of the time: on stocks with a known history the implied
count is flat from 2005 to 2009 and tracks the true one only from 2009-11. So filled
months before STALE_BEFORE carry a share count that is too recent (too high for a
company that kept issuing shares); they are marked stale=True. From 2009-11 the
implied counts follow the real changes (e.g. 5346: -38% in 2010 and -60% in 2012,
5387: -65% in 2010, as in the companies' capital reductions).

One request per stock and year, cached. Writes
    data/shares_fill/tpex_turnover_shares.parquet   stock_id, month, shares, exact (bool)
    data/shares_fill/validation.csv                 the same method on stocks FinMind does have

    python fetch_tpex_shares.py            # about 25 minutes the first time
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
PRICE, SHARES = DATA / "TaiwanStockPrice", DATA / "TaiwanStockShareholding"
CACHE = DATA / "tpex_month_cache"
OUT_DIR = DATA / "shares_fill"
URL = "https://www.tpex.org.tw/www/zh-tw/statistics/monthlyStock"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)",
           "Referer": "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-month.html"}
MAX_ROUNDING = 0.02          # a month counts as exact when the rounding noise is below this share of the rate
STALE_BEFORE = "2009-11"     # earlier months: the page's rate is based on a later share count
DAY_LOSS, DAY_NOISE = 0.005, 0.0029      # points of turnover lost per trading day, and the sd of that loss
N_VALIDATE, VALIDATE_YEARS = 30, (2007, 2012, 2017, 2022)


def fetch(stock: str, year: int) -> list:
    """Rows of one stock-year ([] when the exchange has nothing)."""
    f = CACHE / f"{stock}_{year}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = requests.post(URL, data={"date": str(year), "code": stock, "id": "", "response": "json"},
                              headers=HEADERS, timeout=30)
            if r.status_code == 200:
                j = r.json()
                tables = j.get("tables") or []
                rows = []
                if tables and tables[0].get("data"):
                    t = tables[0]
                    assert t["fields"][0] == "年" and "成交仟股" in t["fields"][7] and "週轉率" in t["fields"][8], t["fields"]
                    rows = t["data"]
                if tables or "查無" in str(j.get("stat", "")):       # an answer, possibly "nothing for this code and year"
                    f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                    time.sleep(1.5)
                    return rows
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{stock} {year}: no answer from the exchange")


def num(x: str) -> float:
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def trading_days(stock: str) -> pd.Series:
    """Days with a trade in each month, from the daily prices already on disk."""
    d = pd.read_parquet(PRICE / f"{stock}.parquet")
    if "close" not in d.columns:                 # an empty file: FinMind has no price for this code
        return pd.Series(dtype=float)
    d = d[(d["close"] > 0) & (d["Trading_Volume"] > 0)]
    return d.groupby(pd.to_datetime(d["date"]).dt.to_period("M")).size().astype(float)


def estimate(stock: str, years) -> pd.DataFrame:
    """month, shares (missing where the month is not exact), volume, turnover (as printed)."""
    days, out = trading_days(stock), []
    for y in years:
        for r in fetch(stock, y):
            m = pd.Period(f"{int(r[0]) + 1911}-{int(r[1]):02d}", "M")
            vol, turn, n = num(r[7]) * 1000, num(r[8]), days.get(m, np.nan)
            true = turn + DAY_LOSS * n
            ok = n > 0 and true > 0 and DAY_NOISE * np.sqrt(n) / true < MAX_ROUNDING
            out.append((m, vol / (true / 100) if ok else np.nan, vol, turn))
    return pd.DataFrame(out, columns=["month", "shares", "volume", "turnover"]).set_index("month").sort_index()


def price_years(stock: str) -> list[int]:
    return sorted({m.year for m in trading_days(stock).index})


def finmind_shares(stock: str) -> pd.Series:
    f = SHARES / f"{stock}.parquet"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_parquet(f)
    if not len(d):
        return pd.Series(dtype=float)
    d = d[d["NumberOfSharesIssued"] > 0].assign(month=pd.to_datetime(d["date"]).dt.to_period("M")).sort_values("date")
    return d.groupby("month")["NumberOfSharesIssued"].last().astype(float)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    info = pd.read_parquet(DATA / "universe.parquet").drop_duplicates("stock_id").set_index("stock_id")
    stocks = sorted(f.stem for f in PRICE.glob("[0-9]*.parquet") if f.stem.isdigit() and len(f.stem) == 4)
    have = {s: finmind_shares(s) for s in stocks}
    targets = [s for s in stocks if have[s].empty and price_years(s)]
    print(f"{len(targets)} stocks with prices and no share count", flush=True)

    fill, summary = [], []
    for i, s in enumerate(targets, 1):
        e = estimate(s, price_years(s))
        exact = e["shares"].notna()
        sh = e["shares"].ffill().bfill()
        summary.append({"stock": s, "name": info["stock_name"].get(s), "type": info["type"].get(s), "months": len(e),
                        "exact_months": int(exact.sum()), "first": str(e.index.min()) if len(e) else "", "last": str(e.index.max()) if len(e) else "",
                        "shares_min": sh.min(), "shares_max": sh.max(),
                        "biggest_jump": float((sh / sh.shift() - 1).abs().max()) if len(sh) > 1 else np.nan})
        if exact.any():
            fill.append(pd.DataFrame({"stock_id": s, "month": e.index.astype(str), "shares": sh.to_numpy(), "exact": exact.to_numpy(),
                                      "stale": e.index.astype(str) < STALE_BEFORE}))
        if i % 10 == 0:
            print(f"  {i}/{len(targets)}", flush=True)
    summary = pd.DataFrame(summary).set_index("stock")
    summary.to_csv(OUT_DIR / "fill_summary.csv", encoding="utf-8-sig")
    fill = pd.concat(fill, ignore_index=True)
    fill.to_parquet(OUT_DIR / "tpex_turnover_shares.parquet", index=False)
    print(f"wrote {len(fill)} stock-months for {fill['stock_id'].nunique()} stocks; "
          f"no usable month: {int((summary['exact_months'] == 0).sum())} stocks", flush=True)

    # the same method on TPEx stocks whose share count FinMind does have
    known = [s for s in stocks if len(have[s]) >= 60 and info["type"].get(s) == "tpex"]
    sample = known[:: max(1, len(known) // N_VALIDATE)][:N_VALIDATE]
    rows = []
    for i, s in enumerate(sample, 1):
        yrs = [y for y in VALIDATE_YEARS if y in price_years(s)]
        e = estimate(s, yrs)
        for m, r in e.iterrows():
            if m in have[s].index:
                rows.append({"stock": s, "name": info["stock_name"].get(s), "month": str(m), "finmind": have[s][m],
                             "estimate": r["shares"], "turnover_%": r["turnover"]})
        if i % 10 == 0:
            print(f"  validation {i}/{len(sample)}", flush=True)
    v = pd.DataFrame(rows)
    v["ratio"] = v["estimate"] / v["finmind"]
    v.to_csv(OUT_DIR / "validation.csv", index=False, encoding="utf-8-sig")
    x = v["ratio"].dropna()
    print(f"validation: {len(sample)} stocks, {len(v)} months, {len(x)} exact; estimate / FinMind median {x.median():.4f}; "
          f"within 2%: {((x - 1).abs() < 0.02).mean():.3f}, within 5%: {((x - 1).abs() < 0.05).mean():.3f}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
