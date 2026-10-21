"""Build time / tick / volume / dollar / imbalance bars from one day of BTCUSDT
trades and compare them.

    uv run python tick-bars/main.py
    uv run python tick-bars/main.py --symbol ETHUSDT --date 2026-09-15

Outputs: a stats table on stdout and charts in tick-bars/output/.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bars as B
from data import download_trades, load_trades

OUT_DIR = Path(__file__).parent / "output"

# Categorical palette, fixed order (one colour per bar type, never cycled).
COLORS = {
    "time": "#8a8984",          # neutral grey: the baseline
    "tick": "#2a78d6",
    "volume": "#eb6834",
    "dollar": "#1baf7a",
    "tick imbalance": "#4a3aa7",
    "volume imbalance": "#e87ba4",
}


def build_all_bars(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Thresholds are picked so most bar types give ~1-2k bars/day, i.e. roughly
    comparable to 1-minute time bars (1440/day). Tune them for other symbols."""
    n_minutes = 24 * 60
    ticks_per_bar = round(len(trades) / n_minutes, -2)
    volume_per_bar = round(trades["qty"].sum() / n_minutes, 1)
    dollars_per_bar = round(trades["quote_qty"].sum() / n_minutes, -4)

    ids = {
        "time": B.time_bar_ids(trades, "1min"),
        "tick": B.tick_bar_ids(trades, int(ticks_per_bar)),
        "volume": B.volume_bar_ids(trades, volume_per_bar),
        "dollar": B.dollar_bar_ids(trades, dollars_per_bar),
        "tick imbalance": B.tick_imbalance_bar_ids(trades, threshold=300),
        "volume imbalance": B.volume_imbalance_bar_ids(trades, threshold=volume_per_bar / 2),
    }
    print(f"thresholds: {ticks_per_bar:.0f} ticks | {volume_per_bar} units | "
          f"${dollars_per_bar:,.0f} | TIB 300 net ticks | VIB {volume_per_bar / 2:.2f} net units\n")
    return {name: B.aggregate(trades, i) for name, i in ids.items()}


def return_stats(bars: pd.DataFrame) -> dict:
    """Stats of log returns between consecutive bar closes.

    AFML's claim: sampling by activity (ticks/volume/dollars) gives returns that
    are closer to IID normal than time bars. Look at excess kurtosis (0 for a
    normal) and the Jarque-Bera statistic (lower = more normal).
    """
    r = np.log(bars["close"]).diff().dropna().to_numpy()
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean() - 3
    duration = (bars.index.to_series().diff().dt.total_seconds()).median()
    return {
        "bars": len(bars),
        "ticks/bar (median)": int(bars["n_ticks"].median()),
        "secs/bar (median)": round(duration, 1),
        "ret std (bp)": round(r.std() * 1e4, 2),
        "skew": round(skew, 2),
        "excess kurtosis": round(kurt, 1),
        "Jarque-Bera": round(len(r) / 6 * (skew ** 2 + kurt ** 2 / 4)),
        "lag-1 autocorr": round(np.corrcoef(r[:-1], r[1:])[0, 1], 3),
    }


def show_adaptive_imbalance(trades: pd.DataFrame) -> None:
    """Run AFML's adaptive imbalance bars to see their instability."""
    print("AFML adaptive imbalance bars (expected_ticks_init=2000):")
    for name, fn in [("tick", B.tick_imbalance_bar_ids), ("volume", B.volume_imbalance_bar_ids)]:
        for clamp in [None, 200]:
            ids = fn(trades, expected_ticks_init=2000, min_ticks=clamp or 1)
            sizes = np.bincount(ids)
            q10, q50, q90 = np.quantile(sizes, [0.1, 0.5, 0.9]).astype(int)
            label = f"min_ticks={clamp}" if clamp else "no clamp"
            print(f"  {name:>6} imbalance, {label:<13} -> {len(sizes):>9,} bars, "
                  f"ticks/bar p10/p50/p90 = {q10}/{q50}/{q90}")
    print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax):
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bdbcb6")
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_sampling(trades, all_bars, path):
    """Same price path, sampled six ways, over the busiest hour of the day.
    Dots mark bar closes: activity-based bars cluster where the action is."""
    busiest = trades["qty"].resample("1h").sum().idxmax()
    window = slice(busiest, busiest + pd.Timedelta("1h"))
    raw = trades.loc[window, "price"]

    fig, axes = plt.subplots(len(all_bars), 1, figsize=(11, 13), sharex=True)
    for ax, (name, bars) in zip(axes, all_bars.items()):
        b = bars.loc[window]
        ax.plot(raw.index, raw.values, color="#c9c8c2", linewidth=0.6)
        ax.plot(b.index, b["close"], color=COLORS[name], linewidth=1.5)
        ax.scatter(b.index, b["close"], color=COLORS[name], s=10, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.set_title(f"{name} bars  ({len(b)} bars in this hour)", loc="left",
                     fontsize=10, color="#0b0b0b")
        _style(ax)
    fig.suptitle(f"Bar closes over the busiest hour ({busiest:%Y-%m-%d %H:%M} UTC); "
                 "grey = every trade", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars_per_hour(all_bars, path):
    """How many bars each method produces per hour, indexed to its daily average
    (1.0 = average). Time bars are flat by construction; the others follow
    market activity."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, bars in all_bars.items():
        per_hour = bars["n_ticks"].resample("1h").size()
        idx = per_hour / per_hour.mean()
        ax.plot(idx.index, idx.values, color=COLORS[name], linewidth=2,
                label=name, linestyle="--" if name == "time" else "-")
    ax.set_ylabel("bars per hour / daily average", color="#52514e", fontsize=9)
    ax.set_title("Activity-based bars sample more when the market is busy",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_return_distributions(all_bars, path):
    """Histogram of standardised bar returns vs the standard normal pdf."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    grid = np.linspace(-6, 6, 300)
    normal = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    for ax, (name, bars) in zip(axes.flat, all_bars.items()):
        r = np.log(bars["close"]).diff().dropna()
        z = (r - r.mean()) / r.std()
        ax.hist(z.clip(-6, 6), bins=np.linspace(-6, 6, 61), density=True,
                color=COLORS[name], edgecolor="white", linewidth=0.5)
        ax.plot(grid, normal, color="#0b0b0b", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
        kurt = (z ** 4).mean() - 3
        ax.set_title(f"{name}  (excess kurtosis {kurt:.1f})", loc="left", fontsize=10)
        _style(ax)
    fig.suptitle("Standardised bar returns vs normal (black). Log y-axis shows the fat tails.",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--date", default="2026-10-01")
    args = p.parse_args()

    trades = load_trades(download_trades(args.symbol, args.date))
    print(f"{args.symbol} {args.date}: {len(trades):,} trades, "
          f"{trades['qty'].sum():,.1f} units, ${trades['quote_qty'].sum():,.0f}")
    tr_agree = (B.tick_rule(trades["price"].to_numpy()) == trades["side"].to_numpy()).mean()
    print(f"tick rule matches the true aggressor side on {tr_agree:.1%} of trades")

    all_bars = build_all_bars(trades)
    stats = pd.DataFrame({name: return_stats(b) for name, b in all_bars.items()}).T
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(stats, "\n")

    show_adaptive_imbalance(trades)

    OUT_DIR.mkdir(exist_ok=True)
    plot_sampling(trades, all_bars, OUT_DIR / "1_sampling.png")
    plot_bars_per_hour(all_bars, OUT_DIR / "2_bars_per_hour.png")
    plot_return_distributions(all_bars, OUT_DIR / "3_return_distributions.png")
    print(f"charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
