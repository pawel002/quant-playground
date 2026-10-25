from pathlib import Path

import polars as pl
import numpy as np
import numpy.typing as npt

from numba import njit

from plots import plot_ticks
from exercise1 import apply_etf_trick

def dollar_etf(df: pl.DataFrame, dollar_bin_size: int, agg_bin_size: str) -> pl.DataFrame:

    ticks = (
        df.lazy()
        .with_columns(
            cum_dollar=(pl.col("size") * pl.col("price") * 50.0).cum_sum()
        )
        .with_columns(
            bin=(pl.col("cum_dollar") // dollar_bin_size)
        )
        .group_by("bin")
        .agg(
            ts_datetime=pl.from_epoch(pl.col("ts_recv"), time_unit="ns").first(),
            open=pl.col("continous_price").first(),
            high=pl.col("continous_price").max(),
            low=pl.col("continous_price").min(),
            close=pl.col("continous_price").last(),
            volume=pl.col("size").sum(),
        )
        .sort("ts_datetime")
    )

    daily_ticks = (
        ticks
        .group_by_dynamic("ts_datetime", every=agg_bin_size)
        .agg(
            tick_count=pl.len()
        )
    )

    return ticks.collect(), daily_ticks.collect()


def prepare_imbalance_data(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df.lazy()
        .with_columns(
            price_diff=pl.col("continous_price").diff()
        )
        .with_columns(
            tick_sign=pl.when(pl.col("price_diff") > 0).then(1)
            .when(pl.col("price_diff") < 0).then(-1)
            .otherwise(None)
        )
        .with_columns(
            tick_sign=pl.col("tick_sign").forward_fill().fill_null(1)
        )
        .with_columns(
            signed_dollar=(pl.col("tick_sign") * pl.col("size") * pl.col("price") * 50.0)
        )
        .drop("price_diff")
        .collect()
    )


@njit
def compute_imbalance_bars_ids(
    signed_dollars: npt.NDArray[np.float64], 
    threshold: float
) -> npt.NDArray[np.int64]:

    n = len(signed_dollars)
    bar_ids = np.empty(n, dtype=np.int64)

    current_bar_id = 0
    cummulative_sum = 0.0

    for i in range(n):

        cummulative_sum += signed_dollars[i]
        bar_ids[i] = current_bar_id

        if abs(cummulative_sum) >= threshold:
            current_bar_id += 1
            cummulative_sum = 0.0

    return bar_ids


def dollar_imbalance(prep_df: pl.DataFrame, agg_bin_size: str, threshold: float) -> pl.DataFrame:

    signed_dollar_arr = (
        prep_df.select(pl.col("signed_dollar"))
        .to_numpy()
        .flatten()
    )

    bar_ids = compute_imbalance_bars_ids(signed_dollar_arr, threshold)

    ticks =  (
        prep_df
        .with_columns(
            bin=pl.Series("bin", bar_ids)
        )
        .group_by("bin")
        .agg(
            ts_datetime=pl.from_epoch(pl.col("ts_recv"), time_unit="ns").first(),
            open=pl.col("continous_price").first(),
            high=pl.col("continous_price").max(),
            low=pl.col("continous_price").min(),
            close=pl.col("continous_price").last(),
            volume=pl.col("size").sum(),
        )
        .sort("ts_datetime")
    )

    daily_ticks = (
        ticks
        .group_by_dynamic("ts_datetime", every=agg_bin_size)
        .agg(
            tick_count=pl.len()
        )
    )

    return ticks, daily_ticks


def main():
    columns = ["ts_recv", "instrument_id", "price", "size"]
    data_path = Path(__file__).parent.parent / "data" / "ES-trades-2026-04-21_2026-10-21.parquet"
    df = (
        pl.scan_parquet(data_path)
        .select(columns)
        .with_row_index()
    )
    # start the continuous series at the first traded price, so both lines match until the first roll
    first_price = (
        df.select(pl.col("price").first())
        .collect()
        .item()
    )
    df = apply_etf_trick(df, starting_base=first_price)

    agg_interval = "1d"
    bin_size = 10000

    dollar_bin_size = (
        df.select(
            (pl.col("price") * pl.col("size") * 50.0).mean()
        )
        .collect()
        .item()
    ) * bin_size
    dollar_ticks, daily_dollar_ticks_count = dollar_etf(df, dollar_bin_size, agg_interval)
    plot_ticks(dollar_ticks, daily_dollar_ticks_count, "output/ex2-dollar-ticks.png")

    prep_df = prepare_imbalance_data(df)
    threshold = (
        prep_df
        # Group into fixed windows of `bin_size` ticks
        .with_columns(chunk_id=pl.col("index") // bin_size)
        .group_by("chunk_id")
        # Sum the signed volume within each chunk to see the net imbalance
        .agg(net_chunk_imbalance=pl.col("signed_dollar").sum())
        # Take the average of the absolute imbalances across all chunks
        .select(pl.col("net_chunk_imbalance").abs().mean())
        .item()
    )
    dollar_imbalance_ticks, dollar_imbalance_ticks_count = dollar_imbalance(prep_df, agg_interval, threshold)
    plot_ticks(dollar_imbalance_ticks, dollar_imbalance_ticks_count, "output/ex2-dollar-imbalance-ticks.png")


if __name__ == "__main__":
    main()

