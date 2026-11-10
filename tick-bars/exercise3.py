"""AFML exercise 2.3: the ETF trick on a PCA-weighted ES / NIY basket, then an ADF test of the spread.

ES  - E-mini S&P 500 futures (USD)
NIY - Nikkei 225 futures in yen, our stand-in for the Euro Stoxx 50, converted to USD with 6J

    uv run python tick-bars/exercise3.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

from plots import plot_adf, plot_omegas, plot_spread

ES_POINT_VALUE_USD = 50.0    # one ES index point is worth $50 per contract
NIY_POINT_VALUE_JPY = 500.0  # one NIY index point is worth ¥500 per contract
BAR_DOLLARS = 10e8           # ES dollar volume per bar
COV_WINDOW = 500             # bars of returns used to estimate the covariance for the PCA weights
REBALANCE_EVERY = "1w"       # besides every roll, re-estimate the weights at this interval


@dataclass(frozen=True)
class AdfResult:
    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: dict[str, float]


# --- bars ---------------------------------------------------------------------------------------

def read_trades(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=["ts_recv", "instrument_id", "price", "size"])


def roll_times(trades: pl.DataFrame) -> pl.Series:
    """Timestamp of the first trade of every new contract."""
    return (
        trades
        .filter(pl.col("instrument_id") != pl.col("instrument_id").shift())
        .get_column("ts_recv")
    )


def with_segment(trades: pl.DataFrame, rolls: pl.Series) -> pl.DataFrame:
    """Number the stretches between rolls: 0 before the first roll, 1 after it, and so on.
    side="right" puts the first trade of a new contract into the new segment."""
    return trades.with_columns(
        segment=rolls.search_sorted(trades["ts_recv"], side="right")
    )


def es_dollar_bars(es: pl.DataFrame) -> pl.DataFrame:
    """ES dollar bars, the bar clock shared by both instruments. A bar never spans a roll."""
    dollar_volume = pl.col("price") * pl.col("size") * ES_POINT_VALUE_USD
    return (
        es
        .with_columns(bin=dollar_volume.cum_sum() // BAR_DOLLARS)
        .group_by("segment", "bin")
        .agg(
            ts_start=pl.col("ts_recv").first(),
            ts_end=pl.col("ts_recv").last(),
            es_open=pl.col("price").first(),
            es_close=pl.col("price").last(),
        )
        .sort("ts_start")
        .with_row_index("bar_id")
        .drop("bin")
    )


def add_niy_prices(bars: pl.DataFrame, niy: pl.DataFrame) -> pl.DataFrame:
    """NIY open and close inside each ES bar. Every NIY trade goes to the latest ES bar that
    started before it, within the same segment. Bars without an NIY trade keep the last NIY price."""
    niy_bars = (
        niy
        .join_asof(
            bars.select("ts_start", "bar_id", "segment"),
            left_on="ts_recv",
            right_on="ts_start",
            by="segment",
            strategy="backward",
            check_sortedness=False,  # both sides are sorted by time; polars can't verify it with `by`
        )
        .drop_nulls("bar_id")
        .group_by("bar_id")
        .agg(niy_open=pl.col("price").first(), niy_close=pl.col("price").last())
    )
    return (
        bars
        .join(niy_bars, on="bar_id", how="left")
        .with_columns(pl.col("niy_close").forward_fill().over("segment"))
        .with_columns(pl.col("niy_open").fill_null(pl.col("niy_close").shift().over("segment")))
    )


def add_point_values(bars: pl.DataFrame, usd_jpy: pl.DataFrame) -> pl.DataFrame:
    """phi: the USD value of one index point. NIY uses yesterday's USD/JPY close (no look-ahead)."""
    fx = usd_jpy.select(
        known_from=pl.col("ts_event").dt.offset_by("1d"),
        usd_per_jpy=pl.col("close"),
    ).sort("known_from")
    return (
        bars
        .join_asof(fx, left_on="ts_start", right_on="known_from", strategy="backward")
        .with_columns(pl.col("usd_per_jpy").backward_fill())  # the first day has no "yesterday"
        .with_columns(
            es_phi=pl.lit(ES_POINT_VALUE_USD),
            niy_phi=NIY_POINT_VALUE_JPY * pl.col("usd_per_jpy"),
        )
        .drop("known_from")
    )


def add_returns_and_rebalances(bars: pl.DataFrame) -> pl.DataFrame:
    """Log returns that never span a roll (the first bar of a segment gets null), and the
    rebalance flag: the last bar before every roll and the last bar of every period."""
    period = pl.col("ts_start").dt.truncate(REBALANCE_EVERY)
    is_last_before_roll = pl.col("segment") != pl.col("segment").shift(-1)
    is_last_of_period = period != period.shift(-1)
    return bars.with_columns(
        es_ret=pl.col("es_close").log().diff().over("segment"),
        niy_ret=pl.col("niy_close").log().diff().over("segment"),
        rebalance=(is_last_before_roll | is_last_of_period).fill_null(False),
    )


# --- ETF trick ----------------------------------------------------------------------------------

def pca_weights(cov: np.ndarray, risk_dist: np.ndarray | None = None, risk_target: float = 1.0) -> np.ndarray:
    """AFML Snippet 2.1: weights that spread the portfolio's risk across principal components.
    By default all the risk goes on the smallest component, i.e. the hedged spread."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]  # largest first
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    if risk_dist is None:
        risk_dist = np.zeros(len(cov))
        risk_dist[-1] = 1.0

    loads = risk_target * np.sqrt(risk_dist / eigenvalues)  # beta_n = sigma * sqrt(R_n / lambda_n)
    weights = eigenvectors @ loads                          # omega = W @ beta
    return weights if weights[0] > 0 else -weights          # eigenvectors have no fixed sign: keep ES long


def etf_trick(bars: pl.DataFrame) -> pl.DataFrame:
    """AFML 2.4.1: K, the value of $1 held in the ES / NIY basket. At every rebalance the PCA weights
    omega are re-estimated from the last COV_WINDOW bars and turned into holdings at the next bar's open.
    Adds K, omega_es and omega_niy (NaN where there was no rebalance)."""
    opens = bars.select("es_open", "niy_open").to_numpy()
    closes = bars.select("es_close", "niy_close").to_numpy()
    phi = bars.select("es_phi", "niy_phi").to_numpy()
    returns = bars.select("es_ret", "niy_ret").to_numpy()
    rebalance = bars["rebalance"].to_numpy()

    n_bars = len(bars)
    K = np.full(n_bars, np.nan)
    omegas = np.full((n_bars, 2), np.nan)
    holdings = np.zeros(2)
    started = rebalanced_last_bar = False

    for t in range(n_bars):
        # 1. mark to market with the holdings set at the previous rebalance
        if started:
            price_change = closes[t] - opens[t] if rebalanced_last_bar else closes[t] - closes[t - 1]
            K[t] = K[t - 1] + holdings @ (phi[t] * price_change)
        rebalanced_last_bar = False

        # 2. rebalance: new weights from past returns only, new holdings at the next bar's open
        if rebalance[t] and t + 1 >= COV_WINDOW and t + 1 < n_bars:
            window = returns[t - COV_WINDOW + 1 : t + 1]
            window = window[~np.isnan(window).any(axis=1)]
            omega = pca_weights(np.cov(window, rowvar=False))
            if not started:
                K[t], started = 1.0, True
            holdings = omega * K[t] / (opens[t + 1] * phi[t] * np.abs(omega).sum())
            omegas[t] = omega
            rebalanced_last_bar = True

    return bars.with_columns(K=K, omega_es=omegas[:, 0], omega_niy=omegas[:, 1])


# --- stationarity -------------------------------------------------------------------------------

def adf_test(series: np.ndarray) -> AdfResult:
    """Augmented Dickey-Fuller test. H0: the series has a unit root (it is NOT stationary).
    A statistic below the critical value (p-value < 0.05) rejects H0, i.e. the series is stationary."""
    statistic, p_value, lags, n_obs, critical_values, _ = adfuller(
        series, regression="c", autolag="AIC", result_object=False
    )
    return AdfResult(statistic, p_value, lags, n_obs, critical_values)


def test_stationarity(spread_bars: pl.DataFrame) -> dict[str, AdfResult]:
    """ADF test of the spread K, and of each leg alone over the same bars for comparison.
    The legs are roll-free log prices: cumulative within-segment log returns."""
    series = {
        "Spread K": spread_bars["K"],
        "ES alone": spread_bars["es_ret"].fill_null(0.0).cum_sum(),
        "NIY alone": spread_bars["niy_ret"].fill_null(0.0).cum_sum(),
    }
    return {name: adf_test(values.to_numpy()) for name, values in series.items()}


def print_adf_results(results: dict[str, AdfResult]) -> None:
    print("ADF test (H0: unit root, i.e. not stationary)")
    for name, r in results.items():
        print(f"\t{name:<10} stat = {r.statistic:7.3f}   p-value = {r.p_value:.4f}   lags = {r.lags:3d}   n = {r.n_obs}")
    critical = next(iter(results.values())).critical_values
    print("\tcritical values:", {level: round(value, 3) for level, value in critical.items()})


def main():
    data_dir = Path(__file__).parent.parent / "data"
    es_trades = data_dir / "ES-trades-2026-04-21_2026-10-21.parquet"
    niy_trades = data_dir / "NIY-trades-2026-04-21_2026-10-21.parquet"
    usd_jpy = data_dir / "6J-ohlcv-1d-2026-04-21_2026-10-21.parquet"

    columns_trades = ["ts_recv", "instrument_id", "price", "size"]
    columns_exchange = ["ts_event", "close"]
    es_trades = pl.read_parquet(es_trades, columns=columns_trades)
    niy_trades = pl.read_parquet(niy_trades, columns=columns_trades)
    usd_jpy = pl.read_parquet(usd_jpy, columns=columns_exchange)

    rolls = pl.concat([roll_times(es_trades), roll_times(niy_trades)]).unique().sort()
    es_trades = with_segment(es_trades, rolls)
    niy_trades = with_segment(niy_trades, rolls)

    bars = es_dollar_bars(es_trades)
    bars = add_niy_prices(bars, niy_trades)
    bars = add_point_values(bars, usd_jpy)
    bars = add_returns_and_rebalances(bars)
    bars = etf_trick(bars)

    plot_omegas(bars, "output/ex3-omegas.png")
    plot_spread(bars, "output/ex3-spread.png")

    # part c): is the spread stationary?
    spread_bars = bars.filter(pl.col("K").is_not_nan())
    results = test_stationarity(spread_bars)
    print_adf_results(results)
    plot_adf(spread_bars, results, "output/ex3-adf.png")


if __name__ == "__main__":
    main()
