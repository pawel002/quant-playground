## Ticks

To better undestand the tick system I have decided to fetch BTC trading data. A short python script is able to download the data in a csv form.

```python
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
```

Using this script I fetched 7 columns with these values:

- trade_id - unique Binance ID for the trade. Increases by 1 with each trade on this symbol.
- price - price this trade was execute, in the quote currency (USDT per 1 BTC)
- qty - how much of the base asset (BTC) changed hands. This is the volume.
- quote_qty - how much of the quote assets changed hands (price \* qty)
- time - timestamp
- is_buyer_maker - which side is the passive one. Every order has two sides, maker and taker. Maker has a limit order resting in the order book and the taker sends an offer which crosses the spread and hits the resting order.
- is_best_match - dropped (always true on binance)
