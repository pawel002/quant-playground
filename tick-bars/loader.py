"""Load the E-mini S&P 500 (ES) tick data downloaded from Databento.

    uv run python tick-bars/loader.py
"""

from pathlib import Path

import polars as pl

DATA_PATH = Path(__file__).parent.parent / "data" / "ES-trades-2026-04-21_2026-10-21.parquet"

COLUMN_DESCRIPTIONS = {
    "ts_recv": "when Databento received the trade message (UTC, nanoseconds)",
    "ts_event": "when CME's matching engine executed the trade (UTC, nanoseconds)",
    "rtype": "record type; 0 = trade record",
    "publisher_id": "data source; 1 = CME Globex",
    "instrument_id": "CME's numeric id of the contract (changes on each roll)",
    "action": "T = trade",
    "side": "aggressor side: B = buyer hit the ask, A = seller hit the bid, N = unknown",
    "depth": "book level the trade happened at (always 0 for trades)",
    "price": "trade price in index points (1 point = $50 per contract)",
    "size": "number of contracts traded",
    "flags": "bit flags from Databento (e.g. 128 = last message in a packet)",
    "ts_in_delta": "how many nanoseconds before ts_recv CME sent the message",
    "sequence": "CME's message sequence number",
    "symbol": "the symbol we requested (ES.v.0 = front-month continuous)",
}


def load_trades(data_path: Path = DATA_PATH, columns: list[str] | None = None) -> pl.DataFrame:
    """Read the trades. Pass `columns` to load only some of them (uses less memory)."""
    return pl.read_parquet(data_path, columns=columns)


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)

    df = load_trades()

    print("=== Columns ===")
    for col, dtype in df.schema.items():
        print(f"{col:<14} {str(dtype):<26} {COLUMN_DESCRIPTIONS.get(col, '')}")

    print("\n=== Size ===")
    print(f"rows:   {df.height:,}")
    print(f"memory: {df.estimated_size('gb'):.2f} GB")

    print("\n=== First rows ===")
    print(df.head())

    print("\n=== Time range ===")
    print(f"first trade: {df['ts_recv'].first()}")
    print(f"last trade:  {df['ts_recv'].last()}")

    print("\n=== Categorical columns ===")
    for col in ["rtype", "publisher_id", "action", "side", "depth", "symbol", "flags"]:
        print(f"{col}: {dict(df[col].value_counts().iter_rows())}")

    print("\n=== Contracts (instrument_id) ===")
    contracts = df.group_by("instrument_id").agg(
        first=pl.col("ts_recv").min(),
        last=pl.col("ts_recv").max(),
        trades=pl.len(),
    )
    print(contracts.sort("first"))

    print("\n=== Price and size ===")
    print(df.select("price", "size").describe())

    print("\n=== Per trading day ===")
    daily = df.group_by(pl.col("ts_recv").dt.date().alias("date")).agg(
        trades=pl.len(),
        contracts=pl.col("size").cast(pl.Int64).sum(),
    )
    print(f"days with trades: {daily.height}")
    print(daily.select("trades", "contracts").describe())

    print("\n=== Aggressor side ===")
    by_side = df.group_by("side").agg(volume=pl.col("size").cast(pl.Int64).sum())
    print(by_side.with_columns(share=pl.col("volume") / pl.col("volume").sum()).sort("side"))
