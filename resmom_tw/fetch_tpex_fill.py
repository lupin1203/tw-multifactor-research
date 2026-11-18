"""Fill the hole in the FinMind daily prices of TPEx stocks, 2007-01-02 .. 2007-03-02.

FinMind has no row in that period for the TPEx stocks with codes 1565-3306 (57 of the 59
stocks in that code block; every TPEx stock outside the block is complete). The stocks
traded normally: the exchange's own page 個股日成交資訊 has every day. This script reads
those months from the exchange (one request per stock and month, cached) and writes the
missing days to data/TaiwanStockPrice/_tpex_fill_2007.parquet, in the same columns as the
per-stock FinMind files, so build_tw.py and resmom/data_tw.py pick them up unchanged.
Days FinMind already has are left alone and used to check that the two sources agree.

    python fetch_tpex_fill.py          # about 5 minutes the first time, then cached
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
PRICE = DATA / "TaiwanStockPrice"
CACHE = DATA / "tpex_fill_cache"
OUT = PRICE / "_tpex_fill_2007.parquet"
URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
LO, HI = 1565, 3306                       # the code block with the hole
GAP = ("2007-01-02", "2007-03-02")
MONTHS = ["2007/01/01", "2007/02/01", "2007/03/01", "2007/04/01"]
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)",
           "Referer": "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html"}


def hole_stocks() -> dict[str, pd.DataFrame]:
    """Stocks in the code block that trade before and after the gap but have no row inside it."""
    out = {}
    for f in sorted(PRICE.glob("[0-9]*.parquet")):
        if not (f.stem.isdigit() and LO <= int(f.stem) <= HI):
            continue
        d = pd.read_parquet(f)
        if not len(d):
            continue
        date = d["date"].astype(str)
        before = ((date >= "2006-10-01") & (date < GAP[0])).any()
        inside = ((date >= GAP[0]) & (date <= GAP[1])).any()
        after = ((date > GAP[1]) & (date <= "2007-06-30")).any()
        if before and after and not inside:
            out[f.stem] = d
    return out


def fetch(stock: str, month: str) -> dict:
    f = CACHE / f"{stock}_{month[:7].replace('/', '')}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(4):
        r = requests.post(URL, data={"code": stock, "date": month, "id": "", "response": "json"},
                          headers=HEADERS, timeout=30)
        if r.status_code == 200:
            j = r.json()
            if j.get("stat") == "ok":
                f.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
                time.sleep(1.5)
                return j
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{stock} {month}: no answer from the exchange")


def rows(j: dict, stock: str) -> pd.DataFrame:
    t = j["tables"][0]
    assert t["fields"][:7] == ["日 期", "成交仟股", "成交仟元", "開盤", "最高", "最低", "收盤"], t["fields"]
    out = []
    for r in t["data"]:
        y, m, d = r[0].strip().split("/")
        try:
            close, lots = float(r[6].replace(",", "")), float(r[1].replace(",", ""))
        except ValueError:                      # "--": no trade that day
            continue
        if close > 0:
            out.append((f"{int(y) + 1911}-{m}-{d}", stock, close, int(round(lots * 1000))))
    return pd.DataFrame(out, columns=["date", "stock_id", "close", "Trading_Volume"])


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    stocks = hole_stocks()
    print(f"{len(stocks)} stocks with the hole", flush=True)
    fill, same, differ, vol_ratio = [], 0, [], []
    for i, (s, have) in enumerate(stocks.items(), 1):
        have = have.assign(date=have["date"].astype(str)).set_index("date")
        first_back = have.index[have.index > GAP[1]].min()
        got = []
        for m in MONTHS:
            if m[:7].replace("/", "-") > first_back[:7]:
                break
            got.append(rows(fetch(s, m), s))
        got = pd.concat(got, ignore_index=True)
        new = got[~got["date"].isin(have.index) & (got["date"] > "2006-12-31") & (got["date"] < first_back)]
        both = got[got["date"].isin(have.index)]
        for r in both.itertuples():
            if abs(have.at[r.date, "close"] - r.close) < 1e-6:
                same += 1
            else:
                differ.append((s, r.date, have.at[r.date, "close"], r.close))
            if have.at[r.date, "Trading_Volume"] > 0:
                vol_ratio.append(r.Trading_Volume / have.at[r.date, "Trading_Volume"])
        fill.append(new)
        if i % 10 == 0:
            print(f"  {i}/{len(stocks)}", flush=True)
    fill = pd.concat(fill, ignore_index=True).sort_values(["stock_id", "date"])
    fill["Trading_Volume"] = fill["Trading_Volume"].astype("int64")
    fill.to_parquet(OUT, index=False)
    print(f"wrote {len(fill)} rows for {fill['stock_id'].nunique()} stocks, {fill['date'].min()} .. {fill['date'].max()} -> {OUT.name}")
    print(f"days both sources have: close equal on {same}, different on {len(differ)}")
    for d in differ[:10]:
        print("   differs:", d)
    if vol_ratio:
        v = pd.Series(vol_ratio)
        print("volume, exchange / FinMind on those days: median %.3f, share within 1%%: %.2f, share near 1000x: %.2f" % (
            v.median(), ((v - 1).abs() < 0.01).mean(), ((v / 1000 - 1).abs() < 0.01).mean()))
    print("ALL DONE")


if __name__ == "__main__":
    main()
