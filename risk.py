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
