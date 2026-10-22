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

    plt.tight_layout()

    # Save output
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path