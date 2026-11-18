"""Download selling-pressure data from FinMind (run after fetch_fund_tw.py; both share
the 600 requests/hour limit, so do not run them at the same time).

  1 TaiwanStockMarginPurchaseShortSale        daily margin / short balances, from 2005
  2 TaiwanStockInstitutionalInvestorsBuySell  daily buy and sell by investor type, from 2012-05
  3 TaiwanDailyShortSaleBalances              daily borrowed-share short balance, from 2005-07
Output data/<dataset>/<stock_id>.parquet, about 4 hours per dataset. Resumable.
Run: python fetch_flow_tw.py
"""
from __future__ import annotations

import pandas as pd

from fetch_tw import DATA, TOKEN, get

JOBS = [
    ("TaiwanStockMarginPurchaseShortSale", "2005-01-01",
     ["date", "stock_id", "MarginPurchaseTodayBalance", "MarginPurchaseLimit", "MarginPurchaseBuy",
      "MarginPurchaseSell", "ShortSaleTodayBalance", "ShortSaleBuy", "ShortSaleSell"]),
    ("TaiwanStockInstitutionalInvestorsBuySell", "2012-01-01",
     ["date", "stock_id", "name", "buy", "sell"]),
    # borrowed-share (SBL) short balance, from 2005-07: needed to tell foreign buying that is
    # short covering from new long positions (added 2026-11-16)
    ("TaiwanDailyShortSaleBalances", "2005-07-01",
     ["date", "stock_id", "SBLShortSalesShortSales", "SBLShortSalesReturns",
      "SBLShortSalesCurrentDayBalance", "MarginShortSalesCurrentDayBalance"]),
]


def main() -> None:
    if not TOKEN:
        raise SystemExit("FINMIND_TOKEN is not set")
    ids = pd.read_parquet(DATA / "universe.parquet")["stock_id"].tolist()
    for ds, start, keep in JOBS:
        out = DATA / ds
        out.mkdir(exist_ok=True)
        print(f"== {ds}", flush=True)
        for n, sid in enumerate(ids):
            f = out / f"{sid}.parquet"
            if f.exists():
                continue
            df = get(ds, data_id=sid, start_date=start)
            if len(df):
                df = df[[c for c in keep if c in df.columns]]
            df.to_parquet(f, index=False)
            if n % 50 == 0:
                print(f"{ds} {n + 1}/{len(ids)} {sid}", flush=True)
        print(f"DONE {ds}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
