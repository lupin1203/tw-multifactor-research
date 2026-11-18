"""Top-N portfolio with a soft turnover penalty and a cost model that depends on the
size of each order. Rules are in FACTOR_SPEC.md (成本與換股規則; 組合、驗證與資金情境).

Soft penalty
  A stock outside the portfolio replaces a held one only if
      expected return(new) - expected return(old) > gamma * (cost to sell old + cost to buy new).
  This is done by ranking on an adjusted figure: held stocks get  alpha + gamma * sell cost,
  the others get  alpha - gamma * buy cost,  and the portfolio is filled from the top.
  Strong newcomers still get in; only marginal swaps are blocked, and they are
  reconsidered every month.

Costs per order (fractions of the order value)
  fee       0.1425% x discount, at least NT$1 an order, both ways
  tax       0.3% on sells
  spread    half a tick each way (tick depends on the price level)
  impact    k x daily volatility x sqrt(order value / average daily trading value)

Position size
  capital / N, but never more than 20% of the stock's average daily trading value;
  what does not fit goes to the next names, so large capital holds more stocks.
  Positions are equal at purchase and then drift; they are not re-equalised monthly
  (that would trade every holding a little each month and pay tax for it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIMIT_CHANGE = pd.Timestamp("2015-06-01")      # daily price limit 7% before, 10% from this day


def tick_size(price):
    """Taiwan stock tick size for a price (ETFs use a different table)."""
    p = np.asarray(price, dtype=float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def limit_flags(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closed at limit-up, closed at limit-down) from daily unadjusted closes.

    Limit-up = previous close x (1 + limit) rounded down to the tick; limit-down =
    previous close x (1 - limit) rounded up. Ex-dividend days use a different
    reference price and are not handled, so a few flags are missed or wrong.
    """
    prev = close.shift(1)
    lim = pd.Series(np.where(close.index < LIMIT_CHANGE, 0.07, 0.10), index=close.index)
    up_raw, dn_raw = prev.mul(1 + lim, axis=0), prev.mul(1 - lim, axis=0)
    tu, td = tick_size(up_raw), tick_size(dn_raw)
    up = np.floor(up_raw / tu + 1e-9) * tu
    dn = np.ceil(dn_raw / td - 1e-9) * td
    return (close >= up - 1e-9) & prev.notna(), (close <= dn + 1e-9) & prev.notna()


@dataclass
class Costs:
    fee: float = 0.001425
    fee_discount: float = 0.2          # 2 折
    min_fee: float = 1.0               # NT$ per order
    tax: float = 0.003                 # on sells
    spread_ticks: float = 0.5          # half a tick each way
    impact_k: float = 1.0
    borrow: float = 0.03               # per year, short positions

    def _common(self, value, price, adv, dvol):
        value = np.maximum(np.asarray(value, dtype=float), 1e-9)
        price, adv, dvol = (np.asarray(x, dtype=float) for x in (price, adv, dvol))
        fee = np.maximum(value * self.fee * self.fee_discount, self.min_fee) / value
        spread = np.nan_to_num(self.spread_ticks * tick_size(price) / price, nan=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            impact = self.impact_k * np.nan_to_num(dvol, nan=0.03) * np.sqrt(value / np.where(adv > 0, adv, np.nan))
        return fee + spread + np.nan_to_num(impact, nan=0.05)

    def buy(self, value, price, adv, dvol):
        return self._common(value, price, adv, dvol)

    def sell(self, value, price, adv, dvol):
        return self._common(value, price, adv, dvol) + self.tax


def _select(order: list, held: dict, slot, V: float, ind: list | None, max_industry: float, forced: set) -> list:
    """Walk down the ranking and fill the capital. Stocks are column positions; `slot`
    gives the size of a new position, `ind` the industry code (-1 = none). Held names
    take their current value. New names are skipped when they would push an industry
    above its limit; held names are never sold for that reason."""
    chosen, ind_val, left = [], {}, V
    for s in forced:
        chosen.append(s)
        left -= held[s]
        if ind is not None:
            ind_val[ind[s]] = ind_val.get(ind[s], 0.0) + held[s]
    for s in order:
        if s in forced:
            continue
        size = held[s] if s in held else float(slot[s])
        if not size > 0:
            continue
        if left < 0.5 * size:
            if left <= 0:
                break
            continue
        g = -1 if ind is None else ind[s]
        if s not in held and g >= 0 and (ind_val.get(g, 0.0) + size) / V > max_industry + 1e-9:
            continue
        chosen.append(s)
        left -= size
        if g >= 0:
            ind_val[g] = ind_val.get(g, 0.0) + size
    return chosen


def simulate(alpha: pd.DataFrame, ret: pd.DataFrame, price: pd.DataFrame, adv: pd.DataFrame,
             dvol: pd.DataFrame, eligible: pd.DataFrame | None = None, no_buy: pd.DataFrame | None = None,
             no_sell: pd.DataFrame | None = None, industry: pd.Series | None = None,
             capital: float = 1e6, n: int = 30, gamma: float = 1.0, costs: Costs | None = None,
             max_adv_share: float = 0.20, max_industry: float = 0.30, short: bool = False,
             start: str | None = None, end: str | None = None, size_mult: pd.DataFrame | None = None) -> dict:
    """Month-by-month simulation.

    alpha     formation month x stock: expected return over the signal's life (decimal).
              short=True builds the basket of the LOWEST alpha and reports the return
              of being short it (borrowing fee included).
    ret       month x stock monthly returns; a missing return of a held stock counts as
              0% and the position is closed at the next rebalance.
    price     month-end unadjusted close; adv = average daily trading value (NT$);
              dvol = daily return standard deviation. All formation month x stock.
    eligible  may be bought (liquidity floor, size floor ...); no_buy / no_sell: closed
              at limit-up / limit-down on the formation day. A month missing from one of
              these three tables means no restriction of that kind in that month.
    size_mult formation month x stock: a new position is size_mult x capital / n instead of
              capital / n (still limited by max_adv_share); missing = 1. Used for weights
              in inverse proportion to volatility.
    Equal adjusted alphas keep the order: held names first, then the column order of alpha.
    Returns {'monthly': DataFrame indexed by holding month (gross, cost, borrow, net,
    names, turnover, max_weight), 'log': per formation month the names bought, sold,
    blocked by the penalty and kept because of it}.
    """
    costs = costs or Costs()
    idx, cols = alpha.index, alpha.columns
    A = -alpha.to_numpy(dtype=float) if short else alpha.to_numpy(dtype=float)
    arr = lambda df: df.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    PX, AV, DV = arr(price), arr(adv), arr(dvol)
    SM = None if size_mult is None else np.nan_to_num(arr(size_mult), nan=1.0)

    def flags(df):                                   # (table as booleans, months the table covers)
        if df is None:
            return None, None
        return df.reindex(index=idx, columns=cols).fillna(False).to_numpy(dtype=bool), idx.isin(df.index)

    (EL, el_on), (NB, nb_on), (NS, ns_on) = flags(eligible), flags(no_buy), flags(no_sell)
    R = ret.reindex(columns=cols).to_numpy(dtype=float)
    ret_row = {m: i for i, m in enumerate(ret.index)}
    ind = None
    if industry is not None:
        lab = industry.reindex(cols)
        ind = pd.factorize(lab.where(lab.map(lambda g: isinstance(g, str))))[0].tolist()   # -1 = no industry
    first = pd.Period(start, "M") if start else None
    last = pd.Period(end, "M") if end else None
    label = lambda ps: [cols[p] for p in ps]

    held: dict = {}                                  # column position -> current value
    cash, rows, log = float(capital), {}, {}
    for i, m in enumerate(idx):
        if (m + 1) not in ret_row or (first is not None and m + 1 < first) or (last is not None and m + 1 > last):
            continue
        a = A[i]
        has = ~np.isnan(a)
        V = sum(held.values()) + cash
        if V <= 0:
            break
        hp = list(held)
        ok = has.copy()
        if hp:
            ok[hp] = False
        if EL is not None and el_on[i]:
            ok &= EL[i]
        if NB is not None and nb_on[i]:
            ok &= ~NB[i]
        with np.errstate(invalid="ignore"):
            slot = np.minimum(V / n * (1.0 if SM is None else SM[i]), max_adv_share * AV[i])
            ok &= (AV[i] > 0) & (PX[i] > 0) & (slot > 0)
        cand = np.flatnonzero(ok)
        bc = np.zeros(len(cols))
        bc[cand] = costs.buy(slot[cand], PX[i, cand], AV[i, cand], DV[i, cand])
        hv = np.array([held[p] for p in hp], dtype=float)
        sc = dict(zip(hp, costs.sell(hv, PX[i, hp], AV[i, hp], DV[i, hp]).tolist())) if hp else {}
        forced = {p for p in hp if NS is not None and ns_on[i] and NS[i, p]}       # cannot be sold today
        stay = [p for p in hp if has[p]]                                           # held and still has a signal

        def ranking(g: float) -> list:
            keys = np.array(stay + cand.tolist(), dtype=int)
            adj = np.concatenate([a[stay] + g * np.array([sc[p] for p in stay]), a[cand] - g * bc[cand]])
            return keys[np.argsort(-adj, kind="stable")].tolist()

        chosen = _select(ranking(gamma), held, slot, V, ind, max_industry, forced)
        free = _select(ranking(0.0), held, slot, V, ind, max_industry, forced) if gamma else chosen
        in_chosen, in_free = set(chosen), set(free)
        blocked = [s for s in free if s not in in_chosen and s not in held]
        kept = [s for s in chosen if s in held and s not in in_free]

        cost_paid, traded, sold, bought = 0.0, 0.0, [], []
        for s in hp:
            if s not in in_chosen:
                v = held.pop(s)
                c = v * sc[s]
                cash += v - c
                cost_paid += c
                traded += v
                sold.append(s)
        for s in chosen:
            if s in held:
                continue
            c, size = float(bc[s]), float(slot[s])
            v = min(size, cash / (1 + c))
            if v < 0.25 * size:
                continue
            held[s] = v
            cash -= v * (1 + c)
            cost_paid += v * c
            traded += v
            bought.append(s)

        start_values = {cols[p]: v for p, v in held.items()}       # after trading, before the month's return
        start_cash = cash
        r = R[ret_row[m + 1]]
        invested = sum(held.values())
        gross_pnl = 0.0
        for s in list(held):
            x = float(r[s])
            x = 0.0 if not np.isfinite(x) else x
            gross_pnl += held[s] * x
            held[s] *= 1 + x
        borrow = costs.borrow / 12 * invested / V if short else 0.0
        gross = gross_pnl / V
        # short=True: the basket is carried as if it were long; the reported return is that of being short it
        net = -gross - cost_paid / V - borrow if short else gross - cost_paid / V
        rows[m + 1] = {"gross": -gross if short else gross, "cost": cost_paid / V, "borrow": borrow, "net": net,
                       "names": len(held), "turnover": traded / (2 * V),
                       "max_weight": max(held.values()) / max(sum(held.values()) + cash, 1e-9) if held else 0.0,
                       "cash_share": cash / max(sum(held.values()) + cash, 1e-9)}
        log[m] = {"bought": label(bought), "sold": label(sold), "blocked": label(blocked), "kept": label(kept),
                  "held": start_values, "cash": start_cash, "value_before": V, "cost": cost_paid}
    return {"monthly": pd.DataFrame(rows).T, "log": log}


def blocked_report(log: dict, ret: pd.DataFrame, months: int = 3, rocket: float = 0.50) -> dict:
    """Were good stocks kept out by the penalty? Compares the next `months` months of
    the stocks it blocked with the held stocks it kept instead, and counts 'rockets'
    (up more than `rocket`) on both sides."""
    lr = np.log1p(ret.clip(lower=-0.999))
    fwd = np.expm1(sum(lr.shift(-k) for k in range(1, months + 1)))
    b, k = [], []
    for m, d in log.items():
        if m not in fwd.index:
            continue
        b += fwd.loc[m].reindex(d["blocked"]).dropna().tolist()
        k += fwd.loc[m].reindex(d["kept"]).dropna().tolist()
    b, k = np.array(b), np.array(k)
    return {"blocked_n": len(b), "kept_n": len(k),
            f"blocked_next{months}m_%": b.mean() * 100 if len(b) else np.nan,
            f"kept_next{months}m_%": k.mean() * 100 if len(k) else np.nan,
            "blocked_rockets": int((b > rocket).sum()), "kept_rockets": int((k > rocket).sum())}


def penalty_report(with_pen: dict, without: dict, ret: pd.DataFrame) -> dict:
    """Missed return, cost saved and what happened to the blocked stocks."""
    a, b = with_pen["monthly"], without["monthly"]
    idx = a.index.intersection(b.index)
    return {"missed_gross_%/mo": (b.loc[idx, "gross"] - a.loc[idx, "gross"]).mean() * 100,
            "cost_saved_%/mo": (b.loc[idx, "cost"] - a.loc[idx, "cost"]).mean() * 100,
            "net_gain_%/mo": (a.loc[idx, "net"] - b.loc[idx, "net"]).mean() * 100,
            "turnover_with_%/mo": a.loc[idx, "turnover"].mean() * 100,
            "turnover_without_%/mo": b.loc[idx, "turnover"].mean() * 100,
            **blocked_report(with_pen["log"], ret)}
