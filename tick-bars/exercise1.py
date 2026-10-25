from pathlib import Path

import polars as pl
import numpy as np

from plots import plot_continuous, plot_ticks, plot_daily_stability

def apply_etf_trick(df: pl.DataFrame, starting_base: float = 1000.0) -> pl.DataFrame:
    """
    Applies the ETF trick to raw futures tick data to create a continuous, strictly positive price series.
    Assumes df is sorted chronologically and contains 'instrument_id' and 'price'.
    """

    return (
        df.lazy()
        .with_columns(
            price_diff=pl.col("price").diff()
        )
        .with_columns(
            true_pnl=(
                pl.when(pl.col("instrument_id") != pl.col("instrument_id").shift(1))
                .then(0.0)
                .otherwise(pl.col("price_diff"))
            )
        )
        .with_columns(
            continous_price=(
                pl.col("true_pnl").fill_null(0.0).cum_sum() + starting_base
            )
        )
        .drop(["price_diff", "true_pnl"])
    )


def timestamp(df: pl.DataFrame, bin_size: int, agg_bin_size: str) -> tuple[pl.DataFrame, pl.DataFrame]:

    ticks = (
        df.lazy()
        .with_columns(
            bin=(pl.col("index") // bin_size)
        )
        .group_by("bin")
        .agg(
            ts_datetime=pl.from_epoch(pl.col("ts_recv"), time_unit="ns").first(),
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close=pl.col("price").last(),
            volume=pl.col("size").sum(),
            continuous_close=pl.col("continous_price").last(),
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


def volume(df: pl.DataFrame, volume_bin_size: int, agg_bin_size: str) -> pl.DataFrame:

    ticks = (
        df.lazy()
        .with_columns(
            cum_volume=pl.col("size").cum_sum()
        )
        .with_columns(
            bin=(pl.col("cum_volume") // volume_bin_size)
        )
        .group_by("bin")
        .agg(
            ts_datetime=pl.from_epoch(pl.col("ts_recv"), time_unit="ns").first(),
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close=pl.col("price").last(),
            volume=pl.col("size").sum(),
            continuous_close=pl.col("continous_price").last(),
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


def dollar(df: pl.DataFrame, dollar_bin_size: int, agg_bin_size: str) -> pl.DataFrame:

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
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close=pl.col("price").last(),
            volume=pl.col("size").sum(),
            continuous_close=pl.col("continous_price").last(),
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


def asses_daily_stability(df: pl.DataFrame, window_size: int) -> pl.DataFrame:

    return (
        df.lazy()
        .with_columns(
            deltas=pl.col("tick_count").diff().fill_null(0.0)
        )
        .with_columns(
            rolling_mean=pl.col("deltas").rolling_mean(window_size=window_size),
            rolling_stds=pl.col("deltas").rolling_std(window_size=window_size),
        )
        .collect()
    )


def calculate_serial_correlation(df: pl.DataFrame, lag: int = 1) -> float:

    return (
        df.lazy()
        .with_columns(
            log_return=(pl.col("close") / pl.col("close").shift(1)).log()
        )
        .with_columns(
            lagged_return=pl.col("log_return").shift(lag)
        )
        .select(
            serial_corr=pl.corr("log_return", "lagged_return", method="pearson")
        )
        .collect()
        .select("serial_corr")
        .item()
    )


def jarque_bera_test(df: pl.DataFrame) -> float:

    returns = (
        df.lazy()
        .with_columns(
            log_return=(pl.col("close") / pl.col("close").shift(1)).log()
        )
        .collect()
    )

    skew = (
        returns
        .select(pl.col("log_return").skew())
        .item()
    )

    kurtosis = (
        returns
        .select(pl.col("log_return").kurtosis())
        .item()
    )

    return len(returns) * (skew ** 2 + kurtosis ** 2 / 4) / 6


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
    timestamp_ticks, daily_timestamp_ticks_count = timestamp(df, bin_size, agg_interval)
    plot_ticks(timestamp_ticks, daily_timestamp_ticks_count, "output/timestamp-ticks.png")
    plot_continuous(timestamp_ticks, "output/timestamp-continuous.png")

    avg_trade_size = (
        df.select(pl.col("size").mean())
        .collect()
        .item()
    )
    volume_bin_size = bin_size * avg_trade_size
    volume_ticks, daily_volume_ticks_count = volume(df, volume_bin_size, agg_interval)
    plot_ticks(volume_ticks, daily_volume_ticks_count, "output/volume-ticks.png")

    dollar_bin_size = (
        df.select(
            (pl.col("price") * pl.col("size") * 50.0).mean()
        )
        .collect()
        .item()
    ) * bin_size
    dollar_ticks, daily_dollar_ticks_count = dollar(df, dollar_bin_size, agg_interval)
    plot_ticks(dollar_ticks, daily_dollar_ticks_count, "output/dollar-ticks.png")


    for window_size in [5, 10, 15]: 
        plot_daily_stability(
            {
                "timestamp": asses_daily_stability(daily_timestamp_ticks_count, window_size),
                "volume": asses_daily_stability(daily_volume_ticks_count, window_size),
                "dollar": asses_daily_stability(daily_dollar_ticks_count, window_size),
            },
            f"output/daily-stability-{window_size}.png",
        )

    for k in [1, 5, 10, 15, 20]:
        print(f"Lag value k = {k}")
        print(f"\tTimestamp serial correlation: {calculate_serial_correlation(timestamp_ticks, k):.4f}")
        print(f"\tVolume serial correlation:    {calculate_serial_correlation(volume_ticks, k):.4f}")
        print(f"\tDollar serial correlation:    {calculate_serial_correlation(dollar_ticks, k):.4f}")

    print("Jarque-Bera test")
    for name, ticks in zip(
        ["timestamp", "volume", "dollars"],
        [timestamp_ticks, volume_ticks, dollar_ticks]
    ):
        print(f"\tValue for {name} ticks is {jarque_bera_test(ticks):.2f}")


if __name__ == "__main__":
    main()