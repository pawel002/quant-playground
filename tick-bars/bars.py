"""Information-driven bars, following López de Prado, *Advances in Financial
Machine Learning* (AFML), chapter 2.

Every bar type here works the same way:
  1. walk through the trades and assign each one a `bar_id`
  2. group by `bar_id` and aggregate into OHLCV  (see `aggregate`)

Only step 1 differs between bar types - it's the rule for *when to close a bar*:

  time bars              every N seconds                       (baseline)
  tick bars              every N trades
  volume bars            every V units of the asset traded
  dollar bars            every D dollars traded
  tick imbalance bars    when |#buy ticks - #sell ticks| gets large
  volume imbalance bars  when |buy volume - sell volume| gets large
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aggregation: bar_id per trade  ->  one OHLCV row per bar
# ---------------------------------------------------------------------------

def aggregate(trades: pd.DataFrame, bar_ids: np.ndarray) -> pd.DataFrame:
    """Turn trades + a bar id for each trade into OHLCV bars.

    The bar's timestamp is the time of its *last* trade (the moment the bar
    closed), which is the only timestamp you could have acted on without
    look-ahead.
    """
    g = trades.assign(
        bar_id=bar_ids,
        time=trades.index,
        buy_qty=trades["qty"].where(trades["side"] > 0, 0.0),
    ).groupby("bar_id", sort=True)
    bars = pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["qty"].sum(),
        "dollar_volume": g["quote_qty"].sum(),
        "buy_volume": g["buy_qty"].sum(),
        "n_ticks": g["price"].size(),
        "start": g["time"].first(),
        "end": g["time"].last(),
    })
    return bars.set_index("end")


# ---------------------------------------------------------------------------
# Standard bars: fixed threshold of time / ticks / volume / dollars
# ---------------------------------------------------------------------------

def time_bar_ids(trades: pd.DataFrame, freq: str = "1min") -> np.ndarray:
    """Baseline: bucket trades by wall-clock time."""
    return trades.index.floor(freq).asi8


def tick_bar_ids(trades: pd.DataFrame, ticks_per_bar: int) -> np.ndarray:
    """Close a bar every `ticks_per_bar` trades. Trivially vectorised."""
    return np.arange(len(trades)) // ticks_per_bar


def _threshold_bar_ids(values: np.ndarray, threshold: float) -> np.ndarray:
    """Close a bar as soon as the running sum of `values` reaches `threshold`,
    then reset the sum to zero.

    (A tempting shortcut is `cumsum(values) // threshold`, but that lets the
    overshoot of one bar leak into the next. The explicit loop is exact.)
    """
    ids = np.empty(len(values), dtype=np.int64)
    bar, running = 0, 0.0
    for i, v in enumerate(values):
        ids[i] = bar
        running += v
        if running >= threshold:
            bar += 1
            running = 0.0
    return ids


def volume_bar_ids(trades: pd.DataFrame, volume_per_bar: float) -> np.ndarray:
    """Close a bar every `volume_per_bar` units of the asset (e.g. BTC)."""
    return _threshold_bar_ids(trades["qty"].to_numpy(), volume_per_bar)


def dollar_bar_ids(trades: pd.DataFrame, dollars_per_bar: float) -> np.ndarray:
    """Close a bar every `dollars_per_bar` of notional (price * qty).
    Robust to big price moves, unlike volume bars."""
    return _threshold_bar_ids(trades["quote_qty"].to_numpy(), dollars_per_bar)


# ---------------------------------------------------------------------------
# Imbalance bars (AFML 2.3.2)
# ---------------------------------------------------------------------------

def tick_rule(prices: np.ndarray) -> np.ndarray:
    """AFML's tick rule: classify each trade as buy (+1) or sell (-1) from
    prices alone.

        b_t = sign(p_t - p_{t-1})   if the price changed
        b_t = b_{t-1}               if it didn't

    Use this when your data has no aggressor flag. Binance does give us the
    aggressor side (`trades["side"]`), which is strictly better - compare the
    two to see how often the tick rule is right.
    """
    signs = np.sign(np.diff(prices, prepend=prices[0]))
    # forward-fill zeros with the previous non-zero sign
    s = pd.Series(signs).replace(0, np.nan).ffill().fillna(1.0)
    return s.to_numpy().astype(np.int8)


def fixed_imbalance_bar_ids(signed: np.ndarray, threshold: float) -> np.ndarray:
    """Simplest imbalance bar: close when |sum of b_t * v_t| >= `threshold`.

    `signed[i]` is b_i * v_i, where b_i is +1/-1 (buy/sell) and v_i is 1 for
    tick imbalance or the trade size for volume imbalance. A bar therefore
    closes when one side has *dominated* the flow by a fixed amount - quiet,
    two-sided markets produce few bars, one-sided pressure produces many.
    """
    ids = np.empty(len(signed), dtype=np.int64)
    bar, theta = 0, 0.0
    for i, x in enumerate(signed):
        ids[i] = bar
        theta += x
        if abs(theta) >= threshold:
            bar += 1
            theta = 0.0
    return ids


def adaptive_imbalance_bar_ids(
    signed: np.ndarray,
    expected_ticks_init: int,
    ewma_bars: int = 20,
    ewma_ticks: int | None = None,
    min_ticks: int = 1,
    max_ticks: int | None = None,
) -> np.ndarray:
    """AFML's imbalance bar with an *adaptive* threshold.

    Within the current bar we accumulate the imbalance
        theta_T = sum_{t=1..T} b_t * v_t
    and close the bar at the first T where
        |theta_T| >= E[T] * |E[b_t * v_t]|
    i.e. when the imbalance exceeds what a "normal" bar is expected to produce.

    E[T]        - expected ticks per bar: EWMA over past bar lengths
    E[b_t v_t]  - expected signed flow per tick: EWMA over past ticks

    Warning - this recipe is famously unstable. When flow is balanced,
    E[b v] ~ 0, the threshold is tiny, bars get short, E[T] shrinks, the
    threshold shrinks further... and you end up with 1-tick bars. In the
    other direction E[T] can run away to a handful of enormous bars.
    `min_ticks` / `max_ticks` clamp E[T] to limit the damage. Run
    `main.py` to see it happen on real data.
    """
    n = len(signed)
    ewma_ticks = ewma_ticks or ewma_bars * expected_ticks_init
    max_ticks = max_ticks or n
    a_bars = 2.0 / (ewma_bars + 1)
    a_ticks = 2.0 / (ewma_ticks + 1)

    # Warm-up: seed E[b v] from the first chunk of data.
    exp_imbalance = signed[:expected_ticks_init * ewma_bars].mean()
    exp_ticks = float(expected_ticks_init)

    ids = np.empty(n, dtype=np.int64)
    bar, theta, ticks = 0, 0.0, 0
    for i in range(n):
        x = signed[i]
        ids[i] = bar
        theta += x
        ticks += 1
        exp_imbalance += a_ticks * (x - exp_imbalance)

        if abs(theta) >= exp_ticks * abs(exp_imbalance):
            exp_ticks += a_bars * (ticks - exp_ticks)
            exp_ticks = min(max(exp_ticks, min_ticks), max_ticks)
            bar += 1
            theta, ticks = 0.0, 0
    return ids


def _trade_signs(trades: pd.DataFrame, use_tick_rule: bool) -> np.ndarray:
    if use_tick_rule:
        return tick_rule(trades["price"].to_numpy()).astype(float)
    return trades["side"].to_numpy().astype(float)


def tick_imbalance_bar_ids(trades: pd.DataFrame, threshold: float | None = None,
                           expected_ticks_init: int | None = None,
                           use_tick_rule: bool = False, **kw) -> np.ndarray:
    """TIBs: bar closes when (#buy ticks - #sell ticks) gets large.

    Pass `threshold` for the fixed version, or `expected_ticks_init` for AFML's
    adaptive version.
    """
    signed = _trade_signs(trades, use_tick_rule)
    if threshold is not None:
        return fixed_imbalance_bar_ids(signed, threshold)
    return adaptive_imbalance_bar_ids(signed, expected_ticks_init, **kw)


def volume_imbalance_bar_ids(trades: pd.DataFrame, threshold: float | None = None,
                             expected_ticks_init: int | None = None,
                             use_tick_rule: bool = False, **kw) -> np.ndarray:
    """VIBs: bar closes when (buy volume - sell volume) gets large.

    Pass `threshold` (in units of the asset, e.g. BTC) for the fixed version,
    or `expected_ticks_init` for AFML's adaptive version.
    """
    signed = _trade_signs(trades, use_tick_rule) * trades["qty"].to_numpy()
    if threshold is not None:
        return fixed_imbalance_bar_ids(signed, threshold)
    return adaptive_imbalance_bar_ids(signed, expected_ticks_init, **kw)
