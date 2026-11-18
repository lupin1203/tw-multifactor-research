"""US Treasury yields by maturity, daily, from FinMind's GovernmentBondsYield dataset
(FRED's CSV download refuses scripts; FinMind is the source the rest of this project uses).

Output data/market/ust_yields.parquet: daily index, columns y1 y2 y3 y5 y7 y10 y20 y30 (% a year).
The TWD per USD rate is already in data/market/usdtwd.parquet (Bank of Taiwan, from FinMind).
Needs env var FINMIND_TOKEN.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
OUT = HERE / "data" / "market" / "ust_yields.parquet"
URL = "https://api.finmindtrade.com/api/v4/data"
START = "2007-01-01"
YEARS = [1, 2, 3, 5, 7, 10, 20, 30]


def one(y: int) -> pd.Series:
    r = requests.get(URL, params={"dataset": "GovernmentBondsYield", "data_id": f"United States {y}-Year",
                                  "start_date": START},
                     headers={"Authorization": "Bearer " + os.environ.get("FINMIND_TOKEN", "")}, timeout=90)
    r.raise_for_status()
    d = pd.DataFrame(r.json().get("data", []))
    if d.empty:
        raise SystemExit(f"no data for the {y}-year")
    s = pd.to_numeric(d.set_index(pd.to_datetime(d["date"]))["value"], errors="coerce")
    return s[~s.index.duplicated()].rename(f"y{y}")


def main() -> None:
    cols = []
    for y in YEARS:
        cols.append(one(y))
        s = cols[-1]
        print(s.name, s.index.min().date(), s.index.max().date(), len(s), "missing", int(s.isna().sum()), flush=True)
        time.sleep(2)
    df = pd.concat(cols, axis=1).sort_index()
    df = df.mask(df == -1.0).dropna(how="all")     # -1 marks a closed day (one row, 2010-10-11 Columbus Day)
    df.to_parquet(OUT)
    print("saved", OUT, df.shape)


if __name__ == "__main__":
    main()
