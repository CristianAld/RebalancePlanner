# Rebalance Planner with Risk Check

Accounts drift from their target model as markets move. This tool flags which
holdings breach tolerance, proposes the minimum trades to fix them, and shows
how portfolio risk changes before and after.

**Live app:** _TODO (Phase 6)_

_Screenshot: TODO (Phase 6)_

## The defensible decision

_TODO (Phase 6): corridor vs. full rebalance, with the turnover numbers._

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Prices come from `data/prices.csv` (about 1 year of adjusted daily closes),
generated once with `python scripts/fetch_prices.py`. The app makes no live API calls.

## Layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI only |
| `drift.py` | Weights, drift, and band breaches |
| `rebalance.py` | Corridor and full-rebalance trades, turnover |
| `risk.py` | Annualized covariance, portfolio volatility, risk contributions |
| `tests/` | pytest suite for every pure function |

## What I'd do next

_TODO (Phase 6)_
