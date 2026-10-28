from pathlib import Path

import polars as pl
import numpy as np

from plots import plot_omegas

def read_dataset(filename: str, columns: list[str]) -> pl.DataFrame:
    path = Path(__file__).parent.parent / "data" / filename
    return pl.read_parquet(path, columns=columns)
    

def roll_times(df: pl.DataFrame) -> pl.Series:
    return (
        df.filter(
            pl.col("instrument_id") != pl.col("instrument_id").shift(1)
        )
        .get_column("ts_recv")
    )


def insert_roll_index(sp500: pl.DataFrame, niy: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:

    # 1. Combine, deduplicate, and sort the roll dates
    roll_series = pl.concat(
        [roll_times(sp500), roll_times(niy)]
    ).unique().sort()

    # 2. Use the evaluated Series (df["col"]) instead of the lazy Expression (pl.col)
    # segment before first rollover gets index 0, before second 1, and so on...
    return (
        sp500
        .with_columns(
            segment=roll_series.search_sorted(sp500["ts_recv"], side="right")
        )
    ), (
        niy
        .with_columns(
            segment=roll_series.search_sorted(niy["ts_recv"], side="right")
        )
    )


def get_bars(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df
        # separate the sp500 into dollar volume beans
        .with_columns(
            bin=((pl.col("price") * pl.col("size") * 50).cum_sum() // 10e9)
        )
        .group_by("segment", "bin")
        .agg(
            ts_start=pl.col("ts_recv").first(),
            ts_end=pl.col("ts_recv").last(),
            es_open=pl.col("price").first(),
            es_close=pl.col("price").last(),
        )
        .sort("ts_start")
        .with_row_index("bar_id")
    )


def get_joined_bars(sp500_bars: pl.DataFrame, niy: pl.DataFrame) -> pl.DataFrame:

    niy_bars = (
        niy
        .join_asof(
            sp500_bars.select("ts_start", "bar_id", "segment"),
            left_on="ts_recv",
            right_on="ts_start",
            by="segment",
            strategy="backward",
        )
        .drop_nulls("bar_id")
        .group_by("bar_id")
        .agg(
            niy_open=pl.col("price").first(),
            niy_close=pl.col("price").last(),
        )
    )

    return (
        sp500_bars
        .join(niy_bars, on="bar_id", how="left")
        .with_columns(
            pl.col("niy_close").forward_fill().over("segment")
        )
        .with_columns(
            pl.col("niy_open").fill_null(pl.col("niy_close").shift().over("segment"))
        )
    )


def pca_weights(cov: np.ndarray, risk_dist: np.ndarray | None = None, risk_target: float = 1.0) -> np.ndarray:
    """AFML Snippet 2.1: weights that spread the portfolio's risk across principal components."""
    # eigen-decomposition of the covariance matrix: cov = e_vec @ diag(e_val) @ e_vec.T
    e_val, e_vec = np.linalg.eigh(cov)

    # sort the components from largest to smallest eigenvalue (eigh returns them ascending)
    order = e_val.argsort()[::-1]
    e_val, e_vec = e_val[order], e_vec[:, order]

    # default: put all the risk on the last component (smallest eigenvalue) -> the spread
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0

    # beta_n = sigma * sqrt(R_n / lambda_n): how much to load on each component
    loads = risk_target * np.sqrt(risk_dist / e_val)

    # back from component coordinates to instrument weights: omega = W @ beta
    w = e_vec @ loads

    # eigenvectors are only defined up to a sign; keep the ES leg positive so the spread doesn't flip
    return w if w[0] > 0 else -w


def calculate_omegas(bars: pl.DataFrame):

    o    = bars.select("es_open", "niy_open").to_numpy()
    p    = bars.select("es_close", "niy_close").to_numpy()
    phi  = bars.select("es_phi", "niy_phi").to_numpy()
    rets = bars.select("es_ret", "niy_ret").to_numpy()
    reb  = bars["rebalance"].to_numpy()

    window, T = 500, len(bars)
    K =      np.full(T, np.nan) 
    omegas = np.full((T, 2), np.nan)
    h =      np.zeros(2)

    started = just_rebalanced = False

    for t in range(T - 1):

        if started:
            delta = p[t] - o[t] if just_rebalanced else p[t] - p[t - 1]
            K[t] = K[t - 1] + np.sum(h * phi[t] * delta)

        if reb[t] and t > window:
            r = rets[t - window + 1 : t + 1]
            r = r[~np.isnan(r).any(axis=1)]

            w = pca_weights(np.cov(r, rowvar=False))
            if not started:
                K[t], started = 1.0, True

            h = w * K[t] / (o[t + 1] * phi[t] * np.abs(w).sum())
            omegas[t] = w
            just_rebalanced = True

        else:
            just_rebalanced = False

    return K, omegas


def main():
    stock_columns = ["ts_recv", "instrument_id", "price", "size"]
    exchange_columns = ["ts_event", "close"]
    sp500 = read_dataset(
        "ES-trades-2026-04-21_2026-10-21.parquet",
        stock_columns,
    )

    niy = read_dataset(
        "NIY-trades-2026-04-21_2026-10-21.parquet",
        stock_columns
    )

    exchange = read_dataset(
        "6J-ohlcv-1d-2026-04-21_2026-10-21.parquet",
        exchange_columns
    )

    sp500, niy = insert_roll_index(sp500, niy)
    sp500_bars = get_bars(sp500)
    bars = get_joined_bars(sp500_bars, niy)

    fx = (
        exchange
        .select(
            ts_known=pl.col("ts_event").dt.offset_by("1d"),
            usd_per_jpy=pl.col("close"),
        )
        .sort("ts_known")
    )

    bars = (
        bars
        .join_asof(
            fx, 
            left_on="ts_start", 
            right_on="ts_known", 
            strategy="backward",
        )
        .with_columns(
            pl.col("usd_per_jpy").backward_fill()
        )
        .with_columns(
            es_phi=pl.lit(50.0), 
            niy_phi=500 * pl.col("usd_per_jpy")
        )
        .with_columns(
            es_ret=pl.col("es_close").log().diff().over("segment"),
            niy_ret=pl.col("niy_close").log().diff().over("segment"),
            week=pl.col("ts_start").dt.truncate("1w"),
        )
        .with_columns(
            rebalance=(
                (pl.col("segment") != pl.col("segment").shift(-1)) |
                (pl.col("week") != pl.col("week").shift(-1))
            )
            .fill_null(False)
        )
    )

    K, omegas = calculate_omegas(bars)
    bars = bars.with_columns(
        K=pl.Series(K),
        omega_es=pl.Series(omegas[:, 0]),
        omega_niy=pl.Series(omegas[:, 1]),
    )
    plot_omegas(bars, "output/ex3-omegas.png")



if __name__ == "__main__":
    main()
