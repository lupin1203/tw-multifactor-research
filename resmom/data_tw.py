"""Taiwan data layer: raw FinMind files (../resmom_tw/data) -> the matrices that
signals.py expects. Nothing here computes a factor return.

  daily()         adjusted daily returns, unadjusted closes, share volume (cached)
  monthly()       monthly returns, market cap, PBR, shares, month-end close
  revenue()       revenue-month x stock
  statements()    quarterly single-quarter figures (income statement, operating cash
                  flow, total assets, liabilities)
  yields()        month-end dividend yield
  flows()         daily foreign / investment-trust net buying, margin, short and
                  borrowed-share short balances, in shares
  industries()    cleaned official industry, financial and late-filer flags
  market()        market-wide daily series for the market-state indicators
  factors()       monthly three-factor returns and the risk-free rate

Units: prices NT$, volume and all flow figures in shares, money in NT$.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE.parent / "resmom_tw" / "data"
TW_OUT = HERE.parent / "resmom_tw" / "output"
CACHE = TW_OUT / "factors"

INCOME = {"Revenue": "rev", "GrossProfit": "gross", "OperatingIncome": "oper"}
# Net income is stored under a different name in each period (checked 2026-11-16):
# IncomeAfterTax 2005-07, NetIncome 2008-12 (-2018), IncomeAfterTaxes and
# EquityAttributableToOwnersOfParent from 2013. First available wins.
NET_INCOME = ["EquityAttributableToOwnersOfParent", "IncomeAfterTaxes", "NetIncome", "IncomeAfterTax"]
# Monthly revenue and quarterly statements are parent-only until 2012 and consolidated
# from 2013: the median year-on-year revenue growth jumps to +20-50% in every month of 2013.
BASIS_BREAK_M, BASIS_BREAK_Q = "2013-01", "2013Q1"
BALANCE = {"TotalAssets": "assets", "Liabilities": "liab"}
CFO = "CashFlowsFromOperatingActivities"


def _files(dataset: str):
    return sorted((RAW / dataset).glob("*.parquet"))


def read_all(dataset: str, cols: list[str] | None = None, types: list[str] | None = None) -> pd.DataFrame:
    """Concatenate one dataset's per-stock files (empty files are skipped)."""
    out = []
    for f in _files(dataset):
        d = pd.read_parquet(f)
        if not len(d):
            continue
        if types is not None:
            d = d[d["type"].isin(types)]
        out.append(d if cols is None else d[[c for c in cols if c in d.columns]])
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _wide(df: pd.DataFrame, value: str, index: str = "date") -> pd.DataFrame:
    return df.pivot_table(index=index, columns="stock_id", values=value, aggfunc="last").sort_index()


# ---------------------------------------------------------------------------- daily
def adjustment_events() -> pd.DataFrame:
    """stock_id, date, adj = previous close / reference price on ex-dividend, capital
    reduction, split and par-value-change days (same sources as build_tw.build_panel)."""
    div = read_all("TaiwanStockDividendResult", ["date", "stock_id", "before_price", "after_price"])
    div = div.set_axis(["date", "stock_id", "before", "after"], axis=1)
    cr = read_all("TaiwanStockCapitalReductionReferencePrice",
                  ["date", "stock_id", "ClosingPriceonTheLastTradingDay", "PostReductionReferencePrice"])
    cr = cr.set_axis(["date", "stock_id", "before", "after"], axis=1)
    sp = pd.read_parquet(RAW / "split_events.parquet")
    sp["date"] = pd.to_datetime(sp["date"])
    ev = pd.concat([div, cr, sp[["date", "stock_id", "before", "after"]]], ignore_index=True)
    ev = ev[(ev["before"] > 0) & (ev["after"] > 0)]
    ev["adj"] = ev["before"] / ev["after"]
    return ev.groupby(["stock_id", "date"], as_index=False)["adj"].prod()


def daily(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{'ret': adjusted daily return, 'close': unadjusted close, 'volume': shares traded},
    each date x stock, RAW: it still contains emerging-board (pre-listing) prices, days
    whose dividend / split adjustment is missing from the source, and jumps where a
    stock code was reused by another company years later. Use panel() for analysis."""
    names = ("ret", "close", "volume")
    fs = {k: CACHE / f"daily_{k}.parquet" for k in names}
    if not refresh and all(f.exists() for f in fs.values()):
        return {k: pd.read_parquet(f) for k, f in fs.items()}
    px = read_all("TaiwanStockPrice", ["date", "stock_id", "close", "Trading_Volume"])
    px = px[px["close"] > 0].sort_values(["stock_id", "date"])
    px = px.merge(adjustment_events(), on=["stock_id", "date"], how="left")
    px["adj"] = px["adj"].fillna(1.0)
    px["ret"] = px["close"] / px.groupby("stock_id")["close"].shift() * px["adj"] - 1
    out = {"ret": _wide(px, "ret"), "close": _wide(px, "close"),
           "volume": _wide(px.rename(columns={"Trading_Volume": "volume"}), "volume")}
    for k in ("ret", "volume"):                       # pivot drops all-missing rows (the first day has no return)
        out[k] = out[k].reindex(index=out["close"].index, columns=out["close"].columns)
    CACHE.mkdir(parents=True, exist_ok=True)
    for k, f in fs.items():
        out[k].to_parquet(f)
    return out


LIMIT_CHANGE = pd.Timestamp("2015-06-01")          # daily price limit 7% before, 10% from this day


def clean_daily_returns(ret: pd.DataFrame, close: pd.DataFrame, listed: pd.DataFrame,
                        free_days: int = 5) -> tuple[pd.DataFrame, int]:
    """Daily returns for analysis: (cleaned returns, number of impossible returns removed).

    listed   month x stock, True in months with a known market cap. Other months are
             dropped: they are emerging-board trading before the listing (no price
             limit, no share count) or stocks the source has no share data for.
    A return beyond the daily price limit (7% until May 2015, 10% after, plus half a
    point for rounding) cannot be real once the stock is listed; it comes from a missing
    dividend / split / capital-reduction adjustment and is set to missing. The first
    `free_days` trading days after listing have no limit and are left alone.
    """
    per = ret.index.to_period("M")
    L = listed.reindex(index=per.unique(), columns=ret.columns).fillna(False).astype(bool)
    Ld = L.reindex(per).to_numpy()
    r = ret.where(Ld)
    lim = np.where(ret.index < LIMIT_CHANGE, 0.075, 0.105)[:, None]
    nth = np.cumsum(Ld & close.reindex(index=ret.index, columns=ret.columns).notna().to_numpy(), axis=0)
    bad = (np.abs(r.to_numpy()) > lim) & (nth > free_days + 1)
    return r.mask(bad), int(bad.sum())


def monthly_from_daily(dret: pd.DataFrame, clip: tuple[float, float] = (-0.95, 1.00)) -> pd.DataFrame:
    """Compounded monthly return; missing unless the stock also had a daily return in the
    previous calendar month (same rule as build_tw.build_panel). Clipped."""
    per = dret.index.to_period("M")
    m = np.expm1(np.log1p(dret.clip(lower=-0.999)).groupby(per).sum(min_count=1))
    n = dret.notna().groupby(per).sum()
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
    n = n.reindex(m.index).fillna(0)
    return m.where(n.shift(1) > 0).clip(*clip)


def exit_returns(ret: pd.DataFrame, close: pd.DataFrame, dret_adj: pd.DataFrame, shares: pd.DataFrame,
                 incomplete_last: bool = True, max_gap: int = 12, tol: float = 0.05,
                 clip: tuple[float, float] = (-0.95, 1.00)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What a position is worth after the stock stops trading. Returns (gone, actual, events).

    gone     month x stock, True in an "exit month": the stock had a monthly return in the
             previous month, has none now and did not trade at all this month. The position
             cannot be traded; its whole outcome is booked in this month.
    actual   the return over the gap, where it can be computed: the stock trades again
             within `max_gap` months (inside the data given, so a cut sample only uses what
             was known at its end) AND two independent estimates agree within `tol`:
               price based  the adjusted return on the first day back (previous close =
                            last close before the gap); wrong when a capital reduction is
                            missing from the adjustment data (price x 4 is not +300%)
               cap based    market value after / before the gap (price x shares); wrong
                            when new shares were sold during the gap
             The price-based figure is then used for long and short positions alike.
             Missing everywhere else (never trades again, estimates disagree, or only one
             exists): the outcome is unknown, and the caller fills in a conservative value
             for each side.
    events   one row per exit month: last and next trading day and the two estimates.

    ret: monthly returns (cleaned); close: daily unadjusted closes; dret_adj: daily adjusted
    returns BEFORE the price-limit cleaning (a return across a gap may exceed the limit);
    shares: monthly shares outstanding. incomplete_last: the last month of the data may
    not be complete yet, so it is never an exit month.
    """
    per = close.index.to_period("M")
    traded = close.notna().groupby(per).any().reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)
    gone = ret.shift(1).notna() & ret.isna() & ~traded
    if incomplete_last and len(gone):
        gone.iloc[-1] = False
    actual = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    sh_before, sh_after = shares.ffill(), shares.bfill(limit=3)
    at = lambda df, m, s: float(df.at[m, s]) if m in df.index and s in df.columns else np.nan
    rows = []
    for s in gone.columns[gone.any().to_numpy()]:
        c = close[s].dropna()
        for m in gone.index[gone[s].to_numpy()]:
            pre, post = c.loc[: (m - 1).end_time], c.loc[m.end_time:]
            row = {"month": m, "stock": s, "last_trade": pre.index[-1] if len(pre) else pd.NaT, "next_trade": pd.NaT,
                   "price_based": np.nan, "cap_based": np.nan, "used": np.nan}
            if len(pre) and len(post) and (post.index[0].to_period("M") - m).n < max_gap:
                d0, d1 = pre.index[-1], post.index[0]
                gp = at(dret_adj, d1, s)
                s0, s1 = at(sh_before, d0.to_period("M"), s), at(sh_after, d1.to_period("M"), s)
                gc = post.iloc[0] * s1 / (pre.iloc[-1] * s0) - 1 if s0 > 0 and s1 > 0 else np.nan
                row.update(next_trade=d1, price_based=gp, cap_based=gc)
                if np.isfinite(gp) and np.isfinite(gc) and abs(gp - gc) <= tol:
                    row["used"] = float(np.clip(gp, *clip))
                    actual.at[m, s] = row["used"]
            rows.append(row)
    events = pd.DataFrame(rows, columns=["month", "stock", "last_trade", "next_trade", "price_based", "cap_based", "used"])
    return gone, actual, events


MERGERS = HERE / "data" / "exit_mergers_tw.csv"


def merger_exits() -> pd.DataFrame:
    """Delistings in which the holder was paid out: the company was merged, bought out or
    turned into a holding company, so each share became cash or shares of the successor
    worth about the last traded price. Columns: stock, delisted (official date), evidence
    (tpex_official = the exchange's own list with that reason; web = a source found for
    that deal; known = widely reported deal, not sourced one by one), note, source.
    Delistings NOT in this file are treated as outcome unknown (financial trouble, or a
    reason that could not be verified)."""
    m = pd.read_csv(MERGERS, dtype={"stock": str})
    m["delisted"] = pd.to_datetime(m["delisted"])
    return m


def merger_flags(events: pd.DataFrame, mergers: pd.DataFrame | None, known_by: pd.Timestamp,
                 window: int = 120, slack: int = 5) -> pd.Series:
    """For each exit month (a row of exit_returns' events): True if it is a merger exit.
    The stock never trades again in the data given, and `mergers` has a delisting of that
    stock dated from `slack` days before to `window` days after its last trade and not
    later than `known_by` (the end of the sample: nothing later may be used). The date
    window keeps a reused stock code, or an earlier suspension of the same stock, apart."""
    out = pd.Series(False, index=events.index)
    if mergers is None or not len(mergers) or not len(events):
        return out
    by = mergers[mergers["delisted"] <= known_by].groupby("stock")["delisted"].apply(list).to_dict()
    for i, r in events.iterrows():
        if pd.notna(r["next_trade"]) or pd.isna(r["last_trade"]):
            continue
        out.at[i] = any(-slack <= (d - r["last_trade"]).days <= window for d in by.get(r["stock"], ()))
    return out


def panel() -> dict:
    """Everything price-based, cleaned and on the listed universe only:
    dret, close, volume (daily); ret_raw, me, pbr, shares, close_m, listed (monthly);
    dret_raw = the daily returns before cleaning (only for exit_returns);
    plus removed = number of impossible daily returns dropped.
    ret_raw is missing in the month after a stock stops trading; exit_returns() says what
    a position was worth then.
    Monthly returns are rebuilt from the cleaned daily returns, so they differ from
    tw_panel.parquet in the months touched by a missing adjustment."""
    d, m = daily(), monthly()
    listed = m["me"].notna()
    dret, removed = clean_daily_returns(d["ret"], d["close"], listed)
    ret = monthly_from_daily(dret).reindex(index=m["me"].index, columns=m["me"].columns).where(listed)
    return {"dret": dret, "dret_raw": d["ret"], "close": d["close"], "volume": d["volume"], "ret_raw": ret,
            "me": m["me"], "pbr": m["pbr"], "shares": m["shares"],
            "close_m": m["close"], "listed": listed, "removed": removed}


# -------------------------------------------------------------------------- monthly
def _per(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    return df


def monthly() -> dict[str, pd.DataFrame]:
    """From the existing panel (build_tw.py): ret (clipped total return), me (month-end
    market cap), pbr; plus month-end close and shares = me / close."""
    p = pd.read_parquet(TW_OUT / "tw_panel.parquet")
    p["month"] = p["month"].astype(str)
    w = lambda v: _per(p.pivot(index="month", columns="stock_id", values=v).sort_index())
    out = {"ret": w("ret"), "me": w("me"), "pbr": w("pbr")}
    d = daily()
    close = d["close"].copy()
    me_idx = pd.Series(close.index, index=close.index.to_period("M")).groupby(level=0).max()
    close_m = close.ffill(limit=5).loc[me_idx.to_numpy()]
    close_m.index = me_idx.index
    out["close"] = close_m.reindex(index=out["me"].index, columns=out["me"].columns)
    out["shares"] = (out["me"] / out["close"]).where(out["close"] > 0)
    return out


# -------------------------------------------------------------------------- revenue
def revenue() -> pd.DataFrame:
    """Revenue-month x stock (NT$). FinMind's `date` is the first day of the month after
    the revenue month; revenue_year / revenue_month give the revenue month itself."""
    r = read_all("TaiwanStockMonthRevenue", ["date", "stock_id", "revenue", "revenue_month", "revenue_year"])
    r["month"] = pd.PeriodIndex.from_fields(year=r["revenue_year"].astype(int), month=r["revenue_month"].astype(int), freq="M")
    w = r.pivot_table(index="month", columns="stock_id", values="revenue", aggfunc="last").sort_index()
    return w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="M"))


# ----------------------------------------------------------------------- statements
def _quarterly(df: pd.DataFrame, value: str = "value") -> pd.DataFrame:
    df = df.copy()
    df["q"] = df["date"].dt.to_period("Q")
    w = df.pivot_table(index="q", columns="stock_id", values=value, aggfunc="last").sort_index()
    return w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="Q"))


def ytd_to_quarter(ytd: pd.DataFrame) -> pd.DataFrame:
    """Year-to-date figures (cash-flow statement) -> single quarters: Q1 as reported,
    later quarters minus the previous quarter of the same year. A quarter whose
    predecessor is missing gets no value."""
    prev = ytd.shift(1)
    q1 = np.asarray(ytd.index.quarter == 1)
    out = ytd - prev
    out.loc[q1] = ytd.loc[q1]
    return out


def statements() -> dict[str, pd.DataFrame]:
    """Quarter x stock, single-quarter figures: rev, gross, oper, ni (attributable to the
    parent where reported, otherwise total), cfo; and quarter-end assets, liab."""
    inc = read_all("TaiwanStockFinancialStatements", ["date", "stock_id", "type", "value"],
                   types=list(INCOME) + NET_INCOME)
    out = {v: _quarterly(inc[inc["type"] == k]) for k, v in INCOME.items()}
    ni = None
    for k in NET_INCOME:
        x = _quarterly(inc[inc["type"] == k])
        ni = x if ni is None else ni.combine_first(x)
    out["ni"] = ni
    if (RAW / "TaiwanStockCashFlowsStatement").exists():
        cf = read_all("TaiwanStockCashFlowsStatement", ["date", "stock_id", "type", "value"], types=[CFO])
        out["cfo"] = ytd_to_quarter(_quarterly(cf))
    if (RAW / "TaiwanStockBalanceSheet").exists():
        bs = read_all("TaiwanStockBalanceSheet", ["date", "stock_id", "type", "value"], types=list(BALANCE))
        for k, v in BALANCE.items():
            out[v] = _quarterly(bs[bs["type"] == k])
    return out


def yields() -> pd.DataFrame:
    """Month-end dividend yield (decimal) published by the exchange."""
    y = read_all("TaiwanStockPER_full", ["date", "stock_id", "dividend_yield"])
    y["month"] = y["date"].dt.to_period("M")
    w = y.sort_values("date").pivot_table(index="month", columns="stock_id", values="dividend_yield", aggfunc="last")
    return w.sort_index() / 100


# ---------------------------------------------------------------------------- flows
def flows() -> dict[str, pd.DataFrame]:
    """Daily, in shares: foreign_net, trust_net (buy - sell), margin_bal, short_bal
    (margin short), sbl_bal (borrowed-share short balance). Margin figures come in lots
    of 1,000 shares and are converted."""
    out = {}
    cal = daily()["close"]
    ins = read_all("TaiwanStockInstitutionalInvestorsBuySell", ["date", "stock_id", "name", "buy", "sell"])
    ins["net"] = ins["buy"] - ins["sell"]
    out["foreign_net"] = _wide(ins[ins["name"] == "Foreign_Investor"], "net")
    out["trust_net"] = _wide(ins[ins["name"] == "Investment_Trust"], "net")
    mg = read_all("TaiwanStockMarginPurchaseShortSale",
                  ["date", "stock_id", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"])
    out["margin_bal"] = _wide(mg, "MarginPurchaseTodayBalance") * 1000
    out["short_bal"] = _wide(mg, "ShortSaleTodayBalance") * 1000
    if (RAW / "TaiwanDailyShortSaleBalances").exists():
        sb = read_all("TaiwanDailyShortSaleBalances", ["date", "stock_id", "SBLShortSalesCurrentDayBalance"])
        out["sbl_bal"] = _wide(sb, "SBLShortSalesCurrentDayBalance")
    traded = cal.notna()
    for k, v in out.items():                          # put everything on the price calendar
        v = v.reindex(index=cal.index, columns=cal.columns)
        if k.endswith("_net"):                        # no record on a trading day = no institutional trade
            started = v.notna().cumsum() > 0
            out[k] = v.fillna(0).where(traded & started)
        else:                                         # balances carry over
            out[k] = v.ffill(limit=5).where(traded)
    return out


def universe() -> pd.DataFrame:
    return pd.read_parquet(RAW / "universe.parquet")


# ------------------------------------------------------------------------- industry
# The official labels differ between the two exchanges and over time; same industry,
# one name. "電子工業" is the old umbrella label for all electronics: build_universe
# kept the first of several labels a stock has, which for 228 TWSE stocks is this one.
# stock_info_raw.parquet (every label of every stock) replaces it when it is available.
SAME_INDUSTRY = {"其他電子類": "其他電子業", "金融業": "金融保險", "電子通路類": "電子通路業",
                 "數位雲端類": "數位雲端", "綠能環保類": "綠能環保", "居家生活類": "居家生活",
                 "運動休閒類": "運動休閒", "觀光事業": "觀光餐旅"}
NOT_AN_INDUSTRY = {"創新板股票", "創新版股票", "其他", "存託憑證", "ETF", "大盤", "Index", "所有證券"}
UMBRELLA = "電子工業"
FINANCIAL = "金融保險"


def clean_industry(labels: pd.Series) -> pd.Series:
    """Official label -> cleaned label; boards and the catch-all '其他' become missing
    (those stocks are ranked market-wide)."""
    x = labels.replace(SAME_INDUSTRY)
    return x.where(~x.isin(NOT_AN_INDUSTRY))


SEMI, SEMI_CHAIN, MIN_SEGMENT = "半導體業", HERE / "data" / "semi_chain_tw.csv", 5


def semi_segments() -> pd.Series:
    """Stock id -> segment of the semiconductor chain (IC設計, 晶圓製造, 記憶體, 封測, 設備與材料),
    from the exchange's industry value-chain platform (https://ic.tpex.org.tw/introduce.php?ic=D000,
    read 2026-11-17: today's companies only). A company in several boxes takes the first of
    記憶體 > 晶圓製造 > 封測 > IC設計 > 設備與材料."""
    if not SEMI_CHAIN.exists():
        return pd.Series(dtype=object)
    d = pd.read_csv(SEMI_CHAIN, dtype=str)
    return d.set_index("stock")["segment"]


def industries() -> pd.DataFrame:
    """Index stock id; columns industry (cleaned, semiconductors split by segment, may be
    missing), industry_official (the same without the split), financial (bool),
    late (bool: financial or foreign primary listing '-KY', later filing deadlines).
    The classification is the current one, not the one at each past date.
    Only stocks the exchange labels 半導體業 are split; one without a segment (delisted, or
    not on the platform) and segments with fewer than MIN_SEGMENT stocks keep 半導體業."""
    u = universe().set_index("stock_id")
    ind = clean_industry(u["industry_category"])
    f = RAW / "stock_info_raw.parquet"
    if f.exists():                                   # prefer a specific label over the umbrella one
        raw = pd.read_parquet(f)
        raw = raw[raw["type"].isin(["twse", "tpex"])].assign(industry=lambda d: clean_industry(d["industry_category"]))
        raw = raw[raw["industry"].notna() & (raw["industry"] != UMBRELLA)].drop_duplicates("stock_id")
        fine = raw.set_index("stock_id")["industry"].reindex(ind.index)
        ind = fine.where((ind == UMBRELLA) & fine.notna(), ind)
    fin = (ind == FINANCIAL).fillna(False)
    seg = semi_segments().reindex(ind.index).where(ind == SEMI)
    seg = seg.where(seg.map(seg.value_counts()) >= MIN_SEGMENT)
    fine = (SEMI + "-" + seg).where(seg.notna(), ind)
    return pd.DataFrame({"industry": fine, "industry_official": ind, "financial": fin,
                         "late": fin | u["stock_name"].str.contains("KY", na=False)})


# --------------------------------------------------------------------------- market
def factors() -> pd.DataFrame:
    """Month x (RMRF, SMB, HML, RF) from build_tw.py."""
    import resmom as rm
    return rm.prep_factors(pd.read_csv(TW_OUT / "tw_factors.csv"))


def market() -> dict[str, pd.Series]:
    """Daily market-wide series (data/market): inst_net = net buying of the three
    institutional groups together (NT$), turnover = trading value of the whole market
    (NT$), taiex = price index (no dividends), usdtwd = spot mid rate (TWD per USD),
    fut_foreign = foreign investors' net open interest in TAIEX futures (contracts)."""
    M = RAW / "market"
    day = lambda d: d.assign(date=pd.to_datetime(d["date"])).set_index("date").sort_index()
    out = {}
    ins = day(pd.read_parquet(M / "stock_inst_total.parquet"))
    tot = ins[ins["name"] == "total"]
    out["inst_net"] = (tot["buy"] - tot["sell"]).astype(float)
    tx = day(pd.read_parquet(M / "taiex.parquet"))
    out["turnover"], out["taiex"] = tx["Trading_money"].astype(float), tx["close"].astype(float)
    fx = day(pd.read_parquet(M / "usdtwd.parquet"))
    mid = (fx["spot_buy"] + fx["spot_sell"]) / 2
    out["usdtwd"] = mid.where(mid > 0)
    fut = day(pd.read_parquet(M / "fut_inst_TX.parquet"))
    f = fut[fut["institutional_investors"] == "外資"]
    out["fut_foreign"] = (f["long_open_interest_balance_volume"] - f["short_open_interest_balance_volume"]).astype(float)
    return {k: v[~v.index.duplicated(keep="last")] for k, v in out.items()}


def market_total_return() -> pd.Series | None:
    """Daily level of the exchange's total-return index (TWSE-listed stocks, dividends
    reinvested), from resmom_tw/fetch_taiex_tr.py; None if not downloaded."""
    f = RAW / "market" / "taiex_tr.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    s = d.assign(date=pd.to_datetime(d["date"])).set_index("date")["close"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def etf_0050_daily() -> pd.Series | None:
    """Daily total return of 0050 (dividends reinvested) from FinMind prices and dividend
    events (data/market/etf_0050_*.parquet). Splits are not in the dividend table: a day
    whose close is about 1/k of the previous close (k = 2..10, |log error| < 2%) is taken
    as a 1:k split. None if not downloaded."""
    M = RAW / "market"
    if not (M / "etf_0050_price.parquet").exists():
        return None
    px = pd.read_parquet(M / "etf_0050_price.parquet")
    px = px[px["close"] > 0].assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["close"].astype(float).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    adj = pd.Series(1.0, index=px.index)
    dv = pd.read_parquet(M / "etf_0050_div.parquet")
    for r in dv.itertuples():
        d = pd.Timestamp(r.date)
        if d in adj.index and r.before_price > 0 and r.after_price > 0:
            adj[d] *= r.before_price / r.after_price
    ratio = px / px.shift()
    for d, x in ratio.items():
        if x < 0.6:
            k = round(1 / x)
            if 2 <= k <= 10 and abs(np.log(x * k)) < 0.02 * 5:          # within a normal day's move
                adj[d] *= k
    return (ratio * adj - 1).iloc[1:]


def ust_yields() -> pd.DataFrame | None:
    """Daily US Treasury yields in % a year, columns = maturity in years (1 2 3 5 7 10 20 30),
    from resmom_tw/fetch_ust.py (FinMind GovernmentBondsYield); None if not downloaded."""
    f = RAW / "market" / "ust_yields.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f).sort_index()
    d.columns = [int(c[1:]) for c in d.columns]
    return d[sorted(d.columns)]


def usdtwd() -> pd.DataFrame | None:
    """Daily Bank of Taiwan spot rate, TWD per USD: columns mid and half_spread (as a fraction
    of mid). Days without a spot quote (-1) are left out. None if not downloaded."""
    f = RAW / "market" / "usdtwd.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d = d[(d["spot_buy"] > 0) & (d["spot_sell"] > 0)]
    d = d.assign(date=pd.to_datetime(d["date"])).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    mid = (d["spot_buy"] + d["spot_sell"]) / 2
    return pd.DataFrame({"mid": mid, "half_spread": (d["spot_sell"] - d["spot_buy"]) / 2 / mid})


def bench_0050() -> pd.Series:
    """Monthly total return of 0050 (Yahoo, from 2009-02; split glitch already fixed)."""
    r = pd.read_csv(HERE / "data" / "bench_0050_TW.csv").set_index("month")["ret"]
    r.index = pd.PeriodIndex(r.index.astype(str), freq="M")
    return r
