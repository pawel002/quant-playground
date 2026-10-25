from pathlib import Path

import polars as pl
import numpy as np
import numpy.typing as npt

from numba import njit

from plots import plot_ticks, plot_imbalance_threshold
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
def compute_dynamic_imbalance_bars(
    signed_dollars: npt.NDArray[np.float64], 
    init_T: float, 
    init_d: float, 
    alpha: float = 0.01  # EWMA smoothing factor (e.g., 0.01 for ~200 bar lookback)
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:

    n = len(signed_dollars)
    bar_ids = np.empty(n, dtype=np.int64)
    # threshold used by each bar (at most n bars), trimmed before returning
    thresholds = np.empty(n, dtype=np.float64)

    current_bar_id = 0
    current_imbalance = 0.0
    current_T = 0

    ewma_T = init_T
    ewma_d = init_d

    for i in range(n):

        current_imbalance += signed_dollars[i]
        bar_ids[i] = current_bar_id
        current_T += 1

        # Calculate dynamic threshold: E[T] * |E[d_t]|
        threshold = ewma_T * abs(ewma_d)

        if threshold < 1e-5:
            threshold = init_T * abs(init_d)

        thresholds[current_bar_id] = threshold

        if abs(current_imbalance) >= threshold:

            # update EWMA
            ewma_T = (alpha * current_T) + ((1.0 - alpha) * ewma_T)

            bar_mean_d = current_imbalance / current_T
            ewma_d = (alpha * bar_mean_d) + ((1.0 - alpha) * ewma_d)

            current_imbalance = 0.0
            current_T = 0
            current_bar_id += 1

    return bar_ids, thresholds[:current_bar_id + 1]


def dollar_imbalance(
    prep_df: pl.DataFrame, 
    agg_bin_size: str, 
    init_T: float,
    init_d: float,
) -> pl.DataFrame:

    signed_dollar_arr = (
        prep_df.select(pl.col("signed_dollar"))
        .to_numpy()
        .flatten()
    )

    bar_ids, thresholds = compute_dynamic_imbalance_bars(
        signed_dollar_arr, init_T, init_d
    )
    bar_thresholds = pl.DataFrame({"bin": np.arange(len(thresholds)), "threshold": thresholds})

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
        .join(bar_thresholds, on="bin")
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
    init_T = float(bin_size)
    init_d = (
        prep_df.lazy()
        .select(pl.col("signed_dollar").mean().abs())
        .collect()
        .item()
    )
    dollar_imbalance_ticks, dollar_imbalance_ticks_count = dollar_imbalance(prep_df, agg_interval, init_T, init_d)
    plot_ticks(dollar_imbalance_ticks, dollar_imbalance_ticks_count, "output/ex2-dollar-imbalance-ticks.png")
    plot_imbalance_threshold(dollar_imbalance_ticks, "output/ex2-dollar-imbalance-threshold.png")


if __name__ == "__main__":
    main()

