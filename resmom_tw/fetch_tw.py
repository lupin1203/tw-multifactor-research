"""Download Taiwan stock data from FinMind for the residual-momentum replication.

Universe: every 4-digit common stock ever listed on TWSE or TPEx since 2005, including
delisted ones (no survivorship bias). Depositary receipts (TDRs) are excluded.

Per stock, five datasets -> data/<dataset>/<stock_id>.parquet
  TaiwanStockPrice                            daily close (unadjusted)
  TaiwanStockShareholding                     NumberOfSharesIssued -> market cap
  TaiwanStockPER                              official P/B -> book-to-market
  TaiwanStockDividendResult                   ex-dividend reference prices -> total return
  TaiwanStockCapitalReductionReferencePrice   capital-reduction reference prices

Free FinMind tier = 600 requests/hour, so the full run takes roughly 18 hours.
Resumable: finished files are skipped. Needs env var FINMIND_TOKEN.
Run: python fetch_tw.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "")
START = "2005-01-01"
PACE = 6.2                       # seconds per request incl. latency (~580 / hour)
DATASETS = [
    "TaiwanStockPrice",
    "TaiwanStockShareholding",
    "TaiwanStockPER",
    "TaiwanStockDividendResult",
    "TaiwanStockCapitalReductionReferencePrice",
]
KEEP = {
    "TaiwanStockPrice": ["date", "stock_id", "close", "Trading_Volume"],
    "TaiwanStockShareholding": ["date", "stock_id", "NumberOfSharesIssued"],
    "TaiwanStockPER": ["date", "stock_id", "PBR"],
}


def get(dataset: str, **params) -> pd.DataFrame:
    """One API call with rate-limit and network retry."""
    while True:
        t0 = time.time()
        try:
            r = requests.get(URL, params={"dataset": dataset, "token": TOKEN, **params},
                             timeout=120)
            j = r.json()
        except Exception as e:                      # network hiccup
            print(f"  network error ({type(e).__name__}), retry in 60s", flush=True)
            time.sleep(60)
            continue
        time.sleep(max(0.0, PACE - (time.time() - t0)))
        msg = str(j.get("msg", ""))
        if j.get("status") == 402 or "limit" in msg.lower():
            print("  rate limit reached, sleeping 10 min", flush=True)
            time.sleep(600)
            continue
        if msg != "success":
            raise RuntimeError(f"{dataset} {params}: {msg}")
        return pd.DataFrame(j.get("data", []))


def build_universe() -> pd.DataFrame:
    f = DATA / "universe.parquet"
    if f.exists():
        return pd.read_parquet(f)
    info = get("TaiwanStockInfo")
    info = info[info["type"].isin(["twse", "tpex"])].drop_duplicates("stock_id")
    info = info[info["stock_id"].str.fullmatch(r"[1-9]\d{3}")]
    info = info[info["industry_category"] != "存託憑證"]
    de = get("TaiwanStockDelisting")
    de = de[de["stock_id"].str.fullmatch(r"[1-9]\d{3}") & (de["date"] >= START)]
    de = de[~de["stock_id"].str.startswith("91")]            # TDRs
    de = de.sort_values("date").drop_duplicates("stock_id", keep="last")
    uni = info[["stock_id", "stock_name", "type", "industry_category"]].merge(
        de[["stock_id", "date"]].rename(columns={"date": "delist_date"}),
        on="stock_id", how="outer")
    uni["type"] = uni["type"].fillna("delisted")
    uni = uni.sort_values("stock_id").reset_index(drop=True)
    uni.to_parquet(f, index=False)
    return uni


def main() -> None:
    if not TOKEN:
        raise SystemExit("FINMIND_TOKEN is not set")
    DATA.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        (DATA / ds).mkdir(exist_ok=True)
    uni = build_universe()
    ids = uni["stock_id"].tolist()
    print(f"universe {len(ids)}: {uni['type'].value_counts().to_dict()}", flush=True)
    for n, sid in enumerate(ids):
        for ds in DATASETS:
            f = DATA / ds / f"{sid}.parquet"
            if f.exists():
                continue
            df = get(ds, data_id=sid, start_date=START)
            if len(df) and ds in KEEP:
                df = df[KEEP[ds]]
            df.to_parquet(f, index=False)
        if n % 25 == 0:
            print(f"{n + 1}/{len(ids)} {sid}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
