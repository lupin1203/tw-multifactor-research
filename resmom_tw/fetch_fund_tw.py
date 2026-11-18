"""Download fundamentals from FinMind for the multi-factor work.

Same universe as fetch_tw.py (data/universe.parquet: TWSE + TPEx incl. delisted).
Datasets are fetched one after another, each for every stock, so the first ones are
complete early:
  1 TaiwanStockMonthRevenue            monthly revenue, from 2004 (create_time is empty before
                                       2026 and is a load date after, NOT the publication date;
                                       date = first day of the month after the revenue month)
  2 TaiwanStockFinancialStatements     income statement, quarterly, from 2005
  3 TaiwanStockCashFlowsStatement      cash flows, from 2008
  4 TaiwanStockBalanceSheet            balance sheet, from 2012
  5 TaiwanStockPER -> TaiwanStockPER_full   daily PER, PBR, dividend yield
Output data/<dataset>/<stock_id>.parquet. About 580 requests an hour on the free
tier, roughly 4 hours per dataset. Resumable. Needs env var FINMIND_TOKEN.
Run: python fetch_fund_tw.py
"""
from __future__ import annotations

import pandas as pd

from fetch_tw import DATA, TOKEN, get

JOBS = [  # dataset, output folder, start date, columns to keep (None = all)
    ("TaiwanStockMonthRevenue", "TaiwanStockMonthRevenue", "2004-01-01",
     ["date", "stock_id", "revenue", "revenue_month", "revenue_year", "create_time"]),
    ("TaiwanStockFinancialStatements", "TaiwanStockFinancialStatements", "2005-01-01",
     ["date", "stock_id", "type", "value"]),
    ("TaiwanStockCashFlowsStatement", "TaiwanStockCashFlowsStatement", "2005-01-01",
     ["date", "stock_id", "type", "value"]),
    ("TaiwanStockBalanceSheet", "TaiwanStockBalanceSheet", "2005-01-01",
     ["date", "stock_id", "type", "value"]),
    ("TaiwanStockPER", "TaiwanStockPER_full", "2005-01-01",
     ["date", "stock_id", "PER", "PBR", "dividend_yield"]),
]


def main() -> None:
    if not TOKEN:
        raise SystemExit("FINMIND_TOKEN is not set")
    ids = pd.read_parquet(DATA / "universe.parquet")["stock_id"].tolist()
    for ds, folder, start, keep in JOBS:
        out = DATA / folder
        out.mkdir(exist_ok=True)
        print(f"== {ds} -> {folder}", flush=True)
        for n, sid in enumerate(ids):
            f = out / f"{sid}.parquet"
            if f.exists():
                continue
            df = get(ds, data_id=sid, start_date=start)
            if len(df) and keep:
                df = df[[c for c in keep if c in df.columns]]
            df.to_parquet(f, index=False)
            if n % 50 == 0:
                print(f"{ds} {n + 1}/{len(ids)} {sid}", flush=True)
        print(f"DONE {ds}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
