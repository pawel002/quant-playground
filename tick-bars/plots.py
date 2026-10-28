from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_ticks(result_df: pl.DataFrame, weekly_ticks: pl.DataFrame, output_path: str | Path = "output/tick-bars.png") -> Path:
    """
    Plots tick bars as a candlestick chart with a volume panel and saves it to disk.
    """
    plt.switch_backend("Agg")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]}
    )

    bins = result_df["bin"]
    opens = result_df["open"]
    closes = result_df["close"]
    highs = result_df["high"]
    lows = result_df["low"]
    volumes = result_df["volume"]

    # Determine colors: Green if Close >= Open, Red if Close < Open
    # Using Polars to compute colors or standard python lists
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(opens, closes)]

    # 1. Candlestick Wicks (High to Low)
    ax1.vlines(bins, lows, highs, color=colors, linewidth=1)

    # 2. Candlestick Bodies (Open to Close)
    # width of the candle bar (e.g., 0.6 of the bin spacing)
    bar_width = 0.6
    
    # We use bar to plot the rectangle between open and close
    bottoms = [min(o, c) for o, c in zip(opens, closes)]
    heights = [abs(c - o) if abs(c - o) > 0 else 0.001 for o, c in zip(opens, closes)] # minimum height to avoid invisible zero-range candles
    
    ax1.bar(bins, heights, bottom=bottoms, width=bar_width, color=colors, edgecolor=colors)

    ax1.set_ylabel("Price")
    ax1.set_title("Tick Bars Candlestick Overview")
    ax1.grid(True, alpha=0.3)

    # 3. Tick count panel
    bins = weekly_ticks["ts_datetime"]
    count = weekly_ticks["tick_count"]

    ax2.bar(bins, count, alpha=0.7, width=bar_width)
    ax2.set_ylabel("Tick count per week")
    ax2.set_xlabel("Week")
    ax2.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_continuous(result_df: pl.DataFrame, output_path: str | Path) -> Path:
    """
    Overlays the raw bar close price and the continuous (roll-adjusted) close price.
    The lines match until the first roll; after each roll they differ by the roll gap.
    """
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result_df["ts_datetime"], result_df["close"], color="#8a8984", linewidth=1, label="raw price")
    ax.plot(result_df["ts_datetime"], result_df["continuous_close"], color="#2a78d6", linewidth=1, label="continuous price")

    ax.set_ylabel("Price")
    ax.set_title("Raw vs continuous price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_daily_stability(daily_stats: dict[str, pl.DataFrame], output_path: str | Path) -> Path:
    """
    Rolling mean (top) and rolling std (bottom) of the daily tick counts, one line per tick type.
    `daily_stats` maps a name (e.g. "volume") to the output of `asses_daily_stability`
    (columns `ts_datetime`, `rolling_mean`, `rolling_stds`).
    """
    plt.switch_backend("Agg")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]

    for (name, stats), color in zip(daily_stats.items(), colors):
        ax1.plot(stats["ts_datetime"], stats["rolling_mean"], color=color, linewidth=1.5, label=name)
        ax2.plot(stats["ts_datetime"], stats["rolling_stds"], color=color, linewidth=1.5, label=name)

    ax1.set_ylabel("Rolling mean")
    ax1.set_title("Daily tick count stability")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Rolling std")
    ax2.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_imbalance_threshold(result_df: pl.DataFrame, output_path: str | Path) -> Path:
    """
    Threshold |theta| had to reach to close each imbalance bar, plotted at the bar's start time.
    Log scale, because the adaptive threshold can change by orders of magnitude.
    """
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result_df["ts_datetime"], result_df["threshold"], color="#2a78d6", linewidth=1)
    ax.set_yscale("log")

    ax.set_ylabel("Threshold [$]")
    ax.set_title("Dollar imbalance bar threshold")
    ax.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_omegas(bars: pl.DataFrame, output_path: str | Path) -> Path:
    """
    PCA weights at each rebalance, as a share of capital: omega_i / sum(|omega|).
    `bars` needs `ts_start`, `omega_es` and `omega_niy` (NaN on bars without a rebalance).
    Drawn as steps, because the weights stay fixed until the next rebalance.
    """
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(10, 4))

    weights = (
        bars
        .filter(pl.col("omega_es").is_not_nan())
        .with_columns(gross=pl.col("omega_es").abs() + pl.col("omega_niy").abs())
        .with_columns(
            es_share=pl.col("omega_es") / pl.col("gross"),
            niy_share=pl.col("omega_niy") / pl.col("gross"),
        )
    )

    ax.step(weights["ts_start"], weights["es_share"], where="post", color="#2a78d6", linewidth=1.5, label="ES")
    ax.step(weights["ts_start"], weights["niy_share"], where="post", color="#eb6834", linewidth=1.5, label="NIY")
    ax.axhline(0, color="#8a8984", linewidth=0.8)

    ax.set_ylabel("Share of capital")
    ax.set_title("PCA weights at each rebalance")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_spread(bars: pl.DataFrame, output_path: str | Path) -> Path:
    """
    Value K of $1 invested in the PCA-weighted ES/NIY basket (the ETF trick output).
    `bars` needs `ts_start`, `K` and `segment`; dotted lines mark the rolls (segment changes).
    """
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(10, 4))

    spread = bars.filter(pl.col("K").is_not_nan())
    rolls = bars.filter(pl.col("segment") != pl.col("segment").shift(1))["ts_start"]

    ax.plot(spread["ts_start"], spread["K"], color="#2a78d6", linewidth=1)
    ax.axhline(1.0, color="#8a8984", linewidth=0.8)
    for roll in rolls:
        ax.axvline(roll, color="#8a8984", linewidth=0.8, linestyle=":")

    ax.set_ylabel("K (value of $1)")
    ax.set_title("ES / NIY spread from the ETF trick (dotted: rolls)")
    ax.grid(True, alpha=0.3)

    return save(fig, output_path)


def plot_adf(spread_bars: pl.DataFrame, results: dict[str, dict], output_path: str | Path) -> Path:
    """
    Top: the spread K around its mean, with a +-2 std band.
    Bottom: ADF test statistic of each series vs the critical values.
    A bar left of a dashed line rejects "not stationary" at that confidence level.
    `results` maps a series name to the output of `adf_test` (keys: stat, pvalue, crit).
    """
    plt.switch_backend("Agg")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})

    k = spread_bars["K"]
    mean, std = k.mean(), k.std()
    ax1.plot(spread_bars["ts_start"], k, color="#2a78d6", linewidth=1)
    ax1.axhline(mean, color="#52514e", linewidth=1)
    ax1.axhspan(mean - 2 * std, mean + 2 * std, color="#2a78d6", alpha=0.08)
    spread = results["Spread K"]
    ax1.set_title(f"Spread K: ADF statistic {spread['stat']:.2f}, p-value {spread['pvalue']:.3f}")
    ax1.set_ylabel("K (value of $1)")
    ax1.grid(True, alpha=0.3)

    names = list(results)
    stats = [results[n]["stat"] for n in names]
    colors = ["#2a78d6" if n == "Spread K" else "#8a8984" for n in names]
    ax2.barh(names, stats, color=colors, height=0.5)
    for n, s in zip(names, stats):
        # label inside the bar, just right of its (negative) end
        ax2.annotate(f"{s:.2f}  (p={results[n]['pvalue']:.3f})", (s, n), xytext=(6, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=8, color="white")
    for level, value in spread["crit"].items():
        ax2.axvline(value, color="#52514e", linewidth=1, linestyle="--")
        # x in data units, y just above the axes
        ax2.text(value, 1.02, f"{level} critical", transform=ax2.get_xaxis_transform(),
                 ha="center", fontsize=8, color="#52514e")
    ax2.invert_yaxis()
    ax2.set_xlabel("ADF test statistic (more negative = more evidence of stationarity)")
    ax2.grid(True, axis="x", alpha=0.3)

    return save(fig, output_path)


def save(fig, output_path: str | Path) -> Path:
    """Save a figure relative to this folder and close it."""
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path