"""Trade proposals: corridor (breachers only) vs. full rebalance.

Pure functions only -- no Streamlit imports. All functions take the DataFrame
returned by drift.compute_drift() and return $ trades indexed by ticker
(positive = buy, negative = sell).
"""

import pandas as pd

# Leftover cash below this many dollars is floating-point noise, not cash.
CASH_TOL = 1e-6


def _gap_dollars(drift: pd.DataFrame) -> pd.Series:
    """$ each holding must trade to land exactly on target."""
    total = drift["value"].sum()
    return (drift["target_weight"] - drift["current_weight"]) * total


def full_trades(drift: pd.DataFrame) -> pd.Series:
    """Trade every holding back to its target weight."""
    return _gap_dollars(drift).rename("trade")


def corridor_trades(drift: pd.DataFrame) -> pd.Series:
    """Trade only breaching holdings back to target, then balance the cash.

    Breachers land exactly on target. Their net trade leaves leftover cash L.
        L > 0: spread across UNDERWEIGHT non-breachers, pro-rata to their $ gap
        L < 0: raise from OVERWEIGHT non-breachers, pro-rata to their $ excess
    Because L equals (total underweight gap - total overweight excess) of the
    non-breachers, pro-rata-by-gap never pushes anyone past their target.

    Invariants (tested): trades sum to $0; no cash flows into an overweight
    position; no position is sold below its target.
    """
    gap = _gap_dollars(drift)
    breach = drift["breach"]
    trades = gap.where(breach, 0.0)

    leftover = -trades.sum()
    if leftover > CASH_TOL:
        eligible = ~breach & (gap > 0)    # underweight non-breachers buy
    elif leftover < -CASH_TOL:
        eligible = ~breach & (gap < 0)    # overweight non-breachers sell
    else:
        return trades.rename("trade")

    share = gap[eligible] / gap[eligible].sum()
    trades[eligible] = leftover * share
    return trades.rename("trade")


def turnover(trades: pd.Series, min_trade: float = 0.01) -> dict:
    """{"dollars": gross $ traded (sum of |trade|), "n_trades": count of trades
    with |trade| >= min_trade}."""
    size = trades.abs()
    return {"dollars": float(size.sum()), "n_trades": int((size >= min_trade).sum())}
