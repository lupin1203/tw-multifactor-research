"""Run the residual-momentum report for Taiwan and the US from the data already built
under research/resmom_tw and research/ff3_us, then print one cross-country Table 2.

  Taiwan  FinMind panel incl. delisted stocks, own FF3 factors, benchmark 0050
          costs: 0.1425% fee each way + 0.3% securities transaction tax on sells
  US      yfinance prices of current SEC filers (survivorship bias: no delisted
          stocks), Ken French's factors, benchmark SPY
          costs: assumed 0.10% each way (spread + commission)

Run: python run_countries.py [tw] [us]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import resmom as rm
from report import run_report

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = HERE / "data"


# Yahoo leaves 0050.TW prices before 2014-01 unadjusted for its 1:4 split (June 2025), which
# shows up as a false -75.5% return in 2014-01 (006203, same market, was -2.2% that month).
# Multiplying the price ratio by 4 gives -1.9%. Found 2026-11-15; the cached csv was patched too.
SPLIT_FIX = {"0050.TW": ("2014-01", 4.0)}


def benchmark(ticker: str) -> pd.Series:
    """Monthly total return of an ETF from Yahoo (dividend-adjusted close), cached."""
    f = DATA / f"bench_{ticker.replace('.', '_')}.csv"
    if not f.exists():
        import yfinance as yf
        DATA.mkdir(exist_ok=True)
        px = yf.download(ticker, start="2005-01-01", interval="1mo", auto_adjust=True,
                         progress=False)["Close"].squeeze().dropna()
        r = px.pct_change().dropna()
        r.index = r.index.to_period("M").astype(str)
        if ticker in SPLIT_FIX:                              # see SPLIT_FIX
            m, k = SPLIT_FIX[ticker]
            if m in r.index and r[m] < -0.5:
                r[m] = (1 + r[m]) * k - 1
        r.rename("ret").rename_axis("month").to_csv(f)
    return pd.read_csv(f).set_index("month")["ret"]


def taiwan() -> dict:
    p = pd.read_parquet(ROOT / "resmom_tw/output/tw_panel.parquet").rename(columns={"stock_id": "id"})
    fac = pd.read_csv(ROOT / "resmom_tw/output/tw_factors.csv")
    return run_report("Taiwan", p, fac, HERE / "output/taiwan", benchmark("0050.TW"), "0050",
                      cost_buy=0.001425, cost_sell=0.004425, start="2016-01", end="2026-10",
                      cap_floors={"cap >= NT$3bn": 3e9, "cap >= NT$10bn": 1e10})


def us() -> dict:
    sys.path.insert(0, str(ROOT / "ff3_us"))
    import build_ff3
    p = build_ff3.build_panel().rename(columns={"ticker": "id"})
    fac = build_ff3.load_french().reset_index()
    last = min(p["month"].max(), fac["month"].max())
    return run_report("US", p[["id", "month", "ret", "me_lag"]], fac, HERE / "output/us",
                      benchmark("SPY"), "SPY", cost_buy=0.001, cost_sell=0.001, end=str(last),
                      cap_floors={"cap >= $300m": 3e8, "cap >= $2bn": 2e9})


def main() -> None:
    which = [a.lower() for a in sys.argv[1:]] or ["tw", "us"]
    done = {}
    if "tw" in which:
        done["Taiwan"] = taiwan()
    if "us" in which:
        done["US"] = us()
    for key, title in (("table2_decile", "Table 2: 10% winners minus losers"),
                       ("table4_30pct", "Table 4: 30% winners minus losers")):
        t = rm.table2({f"{k} {v['start']}..{v['end']}": v["tables"][key] for k, v in done.items()})
        t.to_csv(HERE / f"output/{key}_all.csv", encoding="utf-8-sig")
        print(f"\n===== {title}, mean % per month (Newey-West t) =====\n{t.to_string()}")


if __name__ == "__main__":
    main()
