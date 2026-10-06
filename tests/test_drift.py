import math

import pandas as pd
import pytest

from drift import compute_drift, compute_weights


def test_weights_sum_to_one(sample_holdings):
    w = compute_weights(sample_holdings)
    assert w.sum() == pytest.approx(1.0)
    assert w["VTI"] == pytest.approx(0.48)
    assert w["VNQ"] == pytest.approx(0.065)


def test_sample_account_breaches(sample_holdings, sample_targets):
    d = compute_drift(sample_holdings, sample_targets, abs_band=0.05, rel_band=0.25)

    # VTI: +8 pts, +20% -> absolute only
    assert d.loc["VTI", "drift_pts"] == pytest.approx(0.08)
    assert d.loc["VTI", "breach_reason"] == "absolute"
    # BND: -6 pts, -24% -> absolute only
    assert d.loc["BND", "drift_rel"] == pytest.approx(-0.24)
    assert d.loc["BND", "breach_reason"] == "absolute"
    # VNQ: 6.5% vs 5% -> +1.5 pts, +30% -> relative only
    assert d.loc["VNQ", "drift_pts"] == pytest.approx(0.015)
    assert d.loc["VNQ", "drift_rel"] == pytest.approx(0.30)
    assert d.loc["VNQ", "breach_reason"] == "relative"
    # VXUS -15%, GLD +10%, SHV -20% -> inside both bands
    for t in ["VXUS", "GLD", "SHV"]:
        assert not d.loc[t, "breach"]
        assert d.loc[t, "breach_reason"] == ""

    assert set(d.index[d["breach"]]) == {"VTI", "BND", "VNQ"}


def test_both_bands():
    d = compute_drift(pd.Series({"A": 20.0, "B": 80.0}),
                      pd.Series({"A": 0.10, "B": 0.90}))
    # A: +10 pts and +100%
    assert d.loc["A", "breach_reason"] == "both"


def test_on_the_band_edge_is_not_a_breach():
    d = compute_drift(pd.Series({"A": 45.0, "B": 55.0}),
                      pd.Series({"A": 0.40, "B": 0.60}),
                      abs_band=0.05, rel_band=0.25)
    assert not d["breach"].any()


def test_held_with_zero_target_is_off_model():
    # ARKK is 2% of the account (inside the 5-pt band) but not in the model
    d = compute_drift(pd.Series({"VTI": 98.0, "ARKK": 2.0}),
                      pd.Series({"VTI": 1.0}))
    assert d.loc["ARKK", "target_weight"] == 0
    assert math.isinf(d.loc["ARKK", "drift_rel"])
    assert d.loc["ARKK", "breach_reason"] == "relative"


def test_zero_target_not_held_is_quiet():
    d = compute_drift(pd.Series({"VTI": 100.0}),
                      pd.Series({"VTI": 1.0, "GLD": 0.0}))
    assert d.loc["GLD", "drift_pts"] == 0
    assert not d.loc["GLD", "breach"]


def test_allowed_range_is_the_tighter_band(sample_holdings, sample_targets):
    d = compute_drift(sample_holdings, sample_targets, abs_band=0.05, rel_band=0.25)
    # VTI 40%: abs +-5 is tighter than rel +-10 -> 35-45%
    assert (d.loc["VTI", "band_lo"], d.loc["VTI", "band_hi"]) == pytest.approx((0.35, 0.45))
    # VNQ 5%: rel +-1.25 is tighter than abs +-5 -> 3.75-6.25%
    assert (d.loc["VNQ", "band_lo"], d.loc["VNQ", "band_hi"]) == pytest.approx((0.0375, 0.0625))
    # A breach is exactly "outside the allowed range"
    outside = (d["current_weight"] < d["band_lo"] - 1e-12) | (d["current_weight"] > d["band_hi"] + 1e-12)
    assert (outside == d["breach"]).all()


def test_wider_bands_clear_breaches(sample_holdings, sample_targets):
    d = compute_drift(sample_holdings, sample_targets, abs_band=0.10, rel_band=0.50)
    assert not d["breach"].any()
