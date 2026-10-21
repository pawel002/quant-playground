"""Download and load tick-level trade data from Binance's public data dump.

Source: https://data.binance.vision  (free, no API key, no account)
Every file is one day of *individual trades* for one symbol, e.g.
    https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2026-10-01.zip

CSV columns (no header):
    trade_id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
`time` is in microseconds since epoch. `is_buyer_maker=True` means the buyer's
order was resting in the book, i.e. the *seller* was the aggressor (a sell tick).
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


def download_trades(symbol: str = "BTCUSDT", date: str = "2026-10-01") -> Path:
    """Download (once) and unzip one day of trades. Returns the CSV path."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{symbol}-trades-{date}.csv"
    if csv_path.exists():
        return csv_path

    zip_path = csv_path.with_suffix(".zip")
    url = f"{BASE_URL}/{symbol}/{zip_path.name}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    return csv_path


def load_trades(csv_path: Path) -> pd.DataFrame:
    """Load trades into a DataFrame indexed by timestamp.

    Adds `side`: +1 if the aggressor bought, -1 if the aggressor sold.
    """
    df = pd.read_csv(csv_path, header=None, names=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="us", utc=True)
    df["side"] = np.where(df["is_buyer_maker"], -1, 1).astype(np.int8)
    return df[["time", "price", "qty", "quote_qty", "side"]].set_index("time")

if __name__ == "__main__":

    trades = load_trades(download_trades("BTCUSDT", "2026-10-01"))
    print(trades)
