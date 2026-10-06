"""One-off: download ~1 year of adjusted daily closes into data/prices.csv.

Run locally only -- yfinance is a dev dependency and never ships with the app.
    python scripts/fetch_prices.py
"""

from pathlib import Path

import yfinance as yf

TICKERS = ["VTI", "VXUS", "BND", "VNQ", "GLD", "SHV"]
OUT = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def main() -> None:
    raw = yf.download(TICKERS, period="1y", auto_adjust=True, progress=False)
    prices = raw["Close"][TICKERS].dropna().round(4)
    prices.index.name = "Date"

    OUT.parent.mkdir(exist_ok=True)
    prices.to_csv(OUT)
    print(f"Wrote {OUT} -- {len(prices)} rows x {len(prices.columns)} columns")
    print(f"Range: {prices.index.min().date()} to {prices.index.max().date()}")


if __name__ == "__main__":
    main()
