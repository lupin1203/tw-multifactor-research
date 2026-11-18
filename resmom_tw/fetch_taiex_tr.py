"""Official total-return TAIEX (發行量加權股價報酬指數, dividends reinvested), daily, from the
exchange's own page (https://www.twse.com.tw/zh/indices/taiex/mfi94u.html, data from 2003-01).
One request per month, cached; writes data/market/taiex_tr.parquet (date, close).

    python fetch_taiex_tr.py            # about 15 minutes the first time
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
CACHE = DATA / "taiex_tr_cache"
OUT = DATA / "market" / "taiex_tr.parquet"
URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)",
           "Referer": "https://www.twse.com.tw/zh/indices/taiex/mfi94u.html"}


def fetch(month: pd.Period, refresh: bool = False) -> dict:
    f = CACHE / f"{month.strftime('%Y%m')}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"date": month.strftime("%Y%m01"), "response": "json"}, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                j = r.json()
                if j.get("stat") == "OK":
                    f.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
                    time.sleep(3.0)
                    return j
        except (requests.RequestException, ValueError):
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{month}: no answer from the exchange")


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    now = pd.Timestamp.today().to_period("M")
    rows = []
    months = pd.period_range("2003-01", now, freq="M")
    for i, m in enumerate(months, 1):
        j = fetch(m, refresh=(m >= now - 1))          # the last two months may still be growing
        assert j["fields"][1].startswith("發行量加權股價報酬指數"), j["fields"]
        for d, v in j["data"]:
            y, mm, dd = d.strip().split("/")
            rows.append((pd.Timestamp(int(y) + 1911, int(mm), int(dd)), float(v.replace(",", ""))))
        if i % 24 == 0:
            print(f"  {i}/{len(months)}", flush=True)
    out = pd.DataFrame(rows, columns=["date", "close"]).drop_duplicates("date").sort_values("date")
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} days, {out['date'].min().date()} .. {out['date'].max().date()} -> {OUT.name}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
