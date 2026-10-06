import pandas as pd
import pytest

from drift import compute_drift
from rebalance import corridor_trades, full_trades, turnover


@pytest.fixture
def sample_drift(sample_holdings, sample_targets):
    return compute_drift(sample_holdings, sample_targets, abs_band=0.05, rel_band=0.25)


def post_trade_weights(drift, trades):
    after = drift["value"] + trades
    return after / after.sum()


def test_full_trades_hit_every_target(sample_drift):
    t = full_trades(sample_drift)
    assert t.sum() == pytest.approx(0, abs=1e-6)
    assert t["VTI"] == pytest.approx(-20_000)
    assert t["GLD"] == pytest.approx(-1_250)
    w = post_trade_weights(sample_drift, t)
    assert w.to_dict() == pytest.approx(sample_drift["target_weight"].to_dict())


def test_corridor_hand_checked(sample_drift):
    t = corridor_trades(sample_drift)
    expected = {"VTI": -20_000, "BND": 15_000, "VNQ": -3_750,
                "VXUS": 6_562.50, "SHV": 2_187.50, "GLD": 0}
    assert t.to_dict() == pytest.approx(expected)


def test_corridor_invariants(sample_drift):
    t = corridor_trades(sample_drift)
    w = post_trade_weights(sample_drift, t)
    d = sample_drift

    # 1. Trades net to $0
    assert t.sum() == pytest.approx(0, abs=1e-6)
    # 2. Breachers land exactly on target
    for tk in d.index[d["breach"]]:
        assert w[tk] == pytest.approx(d.loc[tk, "target_weight"])
    # 3. With cash left over, overweight non-breachers are left alone:
    #    no cash flows in (the GLD bug) and nothing is sold off them either
    overweight = d.index[(d["drift_pts"] > 0) & ~d["breach"]]
    assert (t[overweight].abs() <= 1e-9).all()
    # 4. Non-breachers move toward target, never past it
    for tk in d.index[~d["breach"]]:
        before = d.loc[tk, "drift_pts"]
        after = w[tk] - d.loc[tk, "target_weight"]
        assert abs(after) <= abs(before) + 1e-12
        assert after * before >= -1e-12


def test_corridor_raises_cash_when_breachers_need_it():
    # B is 8 pts under target; A, C, D are overweight non-breachers; E is underweight
    holdings = pd.Series({"A": 44_000, "B": 17_000, "C": 23_000, "D": 12_000, "E": 4_000.0})
    targets = pd.Series({"A": 0.40, "B": 0.25, "C": 0.20, "D": 0.10, "E": 0.05})
    d = compute_drift(holdings, targets, abs_band=0.05, rel_band=0.25)
    assert list(d.index[d["breach"]]) == ["B"]

    t = corridor_trades(d)
    assert t.sum() == pytest.approx(0, abs=1e-6)
    assert t["B"] == pytest.approx(8_000)
    # $8,000 raised pro-rata to excess: A 4k, C 3k, D 2k of 9k total
    assert t["A"] == pytest.approx(-8_000 * 4 / 9)
    assert t["C"] == pytest.approx(-8_000 * 3 / 9)
    assert t["D"] == pytest.approx(-8_000 * 2 / 9)
    assert t["E"] == pytest.approx(0)  # never sell an underweight position


def test_no_breaches_no_trades(sample_holdings, sample_targets):
    d = compute_drift(sample_holdings, sample_targets, abs_band=0.10, rel_band=0.50)
    assert (corridor_trades(d) == 0).all()


@pytest.mark.parametrize("scale", [0.004, 40, 4_000])   # $1k, $10M, $1B accounts
def test_account_size_scales_trades_not_decisions(sample_holdings, sample_targets, sample_drift, scale):
    d = compute_drift(sample_holdings * scale, sample_targets, abs_band=0.05, rel_band=0.25)
    assert d["breach"].tolist() == sample_drift["breach"].tolist()
    assert d["current_weight"].to_dict() == pytest.approx(sample_drift["current_weight"].to_dict())
    assert corridor_trades(d).to_dict() == pytest.approx((corridor_trades(sample_drift) * scale).to_dict())


def test_one_fund_blowing_up_is_brought_back(sample_holdings, sample_targets):
    # GLD (5% target) goes up 10x and swamps the account
    holdings = sample_holdings.copy()
    holdings["GLD"] *= 10
    d = compute_drift(holdings, sample_targets, abs_band=0.05, rel_band=0.25)
    t = corridor_trades(d)
    w = post_trade_weights(d, t)
    assert d.loc["GLD", "breach"]
    assert w["GLD"] == pytest.approx(0.05)
    assert t.sum() == pytest.approx(0, abs=1e-6)
    assert (w >= -1e-12).all()


def test_turnover_corridor_vs_full(sample_drift):
    corridor = turnover(corridor_trades(sample_drift))
    full = turnover(full_trades(sample_drift))
    assert corridor == {"dollars": pytest.approx(47_500), "n_trades": 5}
    assert full == {"dollars": pytest.approx(50_000), "n_trades": 6}
