"""Portfolio risk: annualized covariance, volatility, and risk contributions.

Pure functions only -- no Streamlit imports.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read the cached adjusted-close CSV (Date index, one column per ticker)."""
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def annualized_cov(prices: pd.DataFrame) -> pd.DataFrame:
    """Covariance of daily simple returns x 252."""
    return prices.pct_change().dropna().cov() * TRADING_DAYS


def _align(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Reorder weights to cov's tickers; refuse weight on a ticker with no prices."""
    unpriced = [t for t in w.index if t not in cov.index and w[t] != 0]
    if unpriced:
        raise ValueError(f"No price history for: {', '.join(unpriced)}")
    return w.reindex(cov.index, fill_value=0.0).astype(float)


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    """sigma = sqrt(w' Sigma w). Weights are aligned to cov's tickers."""
    w = _align(w, cov)
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each holding's contribution w_i (Sigma w)_i / sigma.

    Contributions sum to sigma; divide by sigma for % of risk.
    """
    w = _align(w, cov)
    sigma = portfolio_vol(w, cov)
    if sigma == 0:
        return pd.Series(0.0, index=w.index)
    return w * (cov @ w) / sigma


def tracking_error(w: pd.Series, benchmark: pd.Series, cov: pd.DataFrame) -> float:
    """Volatility of (w - benchmark): how differently w behaves from the benchmark.

    0 when w equals the benchmark, and never negative, so "share of the gap
    closed" built on it stays between 0% and 100% (unlike a difference in vols).
    """
    return portfolio_vol(w.sub(benchmark, fill_value=0.0), cov)


def past_year(w: pd.Series, prices: pd.DataFrame) -> dict:
    """How a fixed mix would have done over the price history.

    Daily return = sum of w_i x r_i (the mix is held constant). Returns
    {"total_return": compounded return, "max_drawdown": worst peak-to-trough
    fall (<= 0)}.
    """
    rets = prices.pct_change().dropna()
    daily = rets @ _align(w, rets.cov())
    wealth = (1 + daily).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return {
        "total_return": float(wealth.iloc[-1] - 1),
        "max_drawdown": float(min((wealth / peak - 1).min(), 0.0)),
    }
