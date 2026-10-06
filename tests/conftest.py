import pandas as pd
import pytest


@pytest.fixture
def sample_holdings() -> pd.Series:
    """The hand-checked $250k demo account (matches data/sample_account.csv).

    Weights: VTI 48, VXUS 17, BND 19, VNQ 6.5, GLD 5.5, SHV 4 (%).
    """
    return pd.Series(
        {"VTI": 120_000, "VXUS": 42_500, "BND": 47_500,
         "VNQ": 16_250, "GLD": 13_750, "SHV": 10_000},
        dtype=float,
    )


@pytest.fixture
def sample_targets() -> pd.Series:
    return pd.Series(
        {"VTI": 0.40, "VXUS": 0.20, "BND": 0.25,
         "VNQ": 0.05, "GLD": 0.05, "SHV": 0.05}
    )
