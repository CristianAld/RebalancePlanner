from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk import annualized_cov, load_prices, portfolio_vol, risk_contributions

PRICES_CSV = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0003, [0.012, 0.015, 0.004], size=(252, 3))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    idx = pd.bdate_range("2025-01-02", periods=252, name="Date")
    return pd.DataFrame(prices, index=idx, columns=["X", "Y", "Z"])


def test_annualized_cov_is_daily_cov_times_252(synthetic_prices):
    rets = synthetic_prices.pct_change().dropna().to_numpy()
    expected = np.cov(rets, rowvar=False) * 252
    np.testing.assert_allclose(annualized_cov(synthetic_prices).to_numpy(), expected)


def test_portfolio_vol_hand_checked():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert portfolio_vol(w, cov) == pytest.approx(np.sqrt(0.0125))


def test_weights_align_by_ticker_not_position():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"B": 0.8, "A": 0.2})  # reversed order
    assert portfolio_vol(w, cov) == pytest.approx(np.sqrt(0.2**2 * 0.04 + 0.8**2 * 0.01))


def test_weight_on_unpriced_ticker_is_refused():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    with pytest.raises(ValueError, match="ARKK"):
        portfolio_vol(pd.Series({"A": 0.9, "ARKK": 0.1}), cov)


def test_contributions_hand_checked():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    sigma = np.sqrt(0.0125)
    rc = risk_contributions(w, cov)
    assert rc["A"] == pytest.approx(0.01 / sigma)
    assert rc["B"] == pytest.approx(0.0025 / sigma)


def test_contributions_sum_to_sigma(synthetic_prices):
    cov = annualized_cov(synthetic_prices)
    w = pd.Series({"X": 0.5, "Y": 0.3, "Z": 0.2})
    sigma = portfolio_vol(w, cov)
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(sigma)
    assert (rc / sigma).sum() == pytest.approx(1.0)


def test_load_prices_roundtrip(tmp_path, synthetic_prices):
    path = tmp_path / "prices.csv"
    synthetic_prices.to_csv(path)
    loaded = load_prices(path)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == ["X", "Y", "Z"]


@pytest.mark.skipif(not PRICES_CSV.exists(), reason="run scripts/fetch_prices.py first")
def test_cached_prices_csv_shape():
    """Phase 1 done-check: 6 ETFs, about a year of trading days, no gaps."""
    prices = load_prices(PRICES_CSV)
    assert list(prices.columns) == ["VTI", "VXUS", "BND", "VNQ", "GLD", "SHV"]
    assert 240 <= len(prices) <= 260
    assert not prices.isna().any().any()
