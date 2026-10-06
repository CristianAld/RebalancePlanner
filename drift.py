"""Drift detection: how far each holding has moved from its target weight.

Pure functions only -- no Streamlit imports. Weights and bands are fractions
(0.05 means 5 points; 0.25 means 25% of target).
"""

import numpy as np
import pandas as pd

# Comparisons use this slack so a drift sitting exactly on the band edge
# does not breach because of floating-point noise.
EPS = 1e-12


def compute_weights(holdings: pd.Series) -> pd.Series:
    """Market values ($) indexed by ticker -> weights that sum to 1."""
    holdings = holdings.astype(float)
    total = holdings.sum()
    if total <= 0:
        raise ValueError("Account has no market value")
    return holdings / total


def compute_drift(
    holdings: pd.Series,
    targets: pd.Series,
    abs_band: float = 0.05,
    rel_band: float = 0.25,
) -> pd.DataFrame:
    """Drift table, one row per ticker in holdings or targets (missing -> 0).

    Columns:
        value           market value in $
        current_weight  value / total value
        target_weight   target fraction
        drift_pts       current_weight - target_weight
        drift_rel       drift_pts / target_weight (inf if target is 0 and held)
        band_lo/band_hi allowed weight range (the tighter of the two bands)
        breach          True if outside either band
        breach_reason   "absolute", "relative", "both", or ""

    Rules:
        absolute breach: |drift_pts| > abs_band
        relative breach: |drift_rel| > rel_band
        target 0 and held -> off-model, always a relative breach
        target 0 and not held -> no drift, no breach
    """
    # Holdings order first, then any model-only tickers
    tickers = holdings.index.append(targets.index.difference(holdings.index))
    value = holdings.reindex(tickers, fill_value=0).astype(float)
    current = compute_weights(value)
    target = targets.reindex(tickers, fill_value=0).astype(float)

    drift_pts = current - target
    has_target = target > 0
    drift_rel = pd.Series(
        np.where(has_target, drift_pts / target.where(has_target, 1.0),
                 np.where(current > 0, np.inf, 0.0)),
        index=tickers,
    )

    half_width = np.minimum(abs_band, rel_band * target)
    band_lo = (target - half_width).clip(lower=0)
    band_hi = target + half_width

    abs_breach = drift_pts.abs() > abs_band + EPS
    rel_breach = drift_rel.abs() > rel_band + EPS
    reason = np.select(
        [abs_breach & rel_breach, abs_breach, rel_breach],
        ["both", "absolute", "relative"],
        default="",
    )

    return pd.DataFrame({
        "value": value,
        "current_weight": current,
        "target_weight": target,
        "drift_pts": drift_pts,
        "drift_rel": drift_rel,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "breach": abs_breach | rel_breach,
        "breach_reason": reason,
    })
