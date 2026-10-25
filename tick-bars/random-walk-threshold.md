## Scaling the imbalance threshold like a random walk

Why AFML's adaptive threshold either explodes or collapses on ES data, and a threshold that doesn't.

### 1. Setup

Let $d_t = b_t \, p_t v_t \cdot 50$ be the signed dollar flow of trade $t$, and

$$
    \theta_T = \sum_{t=1}^{T} d_t
$$

the imbalance after $T$ trades of the current bar. Write each trade as a mean plus noise:

$$
    d_t = \mu + \varepsilon_t, \qquad \mu = \mathbb{E}[d_t], \qquad \sigma^2 = \operatorname{Var}(d_t).
$$

Assuming for now that the $\varepsilon_t$ are independent:

$$
    \mathbb{E}[\theta_T] = T\mu, \qquad \operatorname{Var}(\theta_T) = T\sigma^2,
$$

so by the central limit theorem

$$
    \theta_T \approx \underbrace{T\mu}_{\text{drift}} + \underbrace{\sigma\sqrt{T}\, Z}_{\text{noise}}, \qquad Z \sim \mathcal{N}(0, 1).
$$

### 2. Drift or noise: which one moves $\theta$?

The drift grows like $T$, the noise like $\sqrt{T}$. They are equal at

$$
    T|\mu| = \sigma\sqrt{T} \quad\Longrightarrow\quad T^* = \left(\frac{\sigma}{\mu}\right)^2 .
$$

For bars much shorter than $T^*$, $\theta_T$ is a driftless random walk. On our 6 months of ES:

| quantity | value |
|---|---|
| $\mu$ | $-\$389$ |
| $\sigma$ | $\$2{,}413{,}694$ |
| $\lvert\mu\rvert / \sigma$ | $1.6 \times 10^{-4}$ |
| $T^*$ | **38.5M trades** (two-thirds of the whole dataset) |

So **every bar we care about lives in the noise-dominated regime**. Measured over blocks of $T$ trades, the typical imbalance is about 100× larger than the drift term:

| $T$ | $\mathbb{E}\lvert\theta_T\rvert$ (measured) | $T\lvert\mu\rvert$ (drift) |
|---|---|---|
| 1,000 | \$128M | \$0.4M |
| 10,000 | \$517M | \$3.9M |
| 100,000 | \$1,880M | \$38.9M |

### 3. Why AFML's threshold is unstable here

AFML closes a bar when $|\theta_T| \ge b$ with

$$
    b = \mathbb{E}[T] \cdot |\mathbb{E}[d_t]| = \mathbb{E}[T]\,|\mu| .
$$

This compares $\theta$ to its **drift**. But in the noise regime, $\theta$ reaches $\pm b$ through random fluctuations, so we need the time a random walk takes to first hit $\pm b$.

**First-passage time.** For a driftless walk with step variance $\sigma^2$, the process $\theta_t^2 - \sigma^2 t$ is a martingale. Stopping it at $\tau$, the first time $|\theta_t| \ge b$ (optional stopping theorem):

$$
    \mathbb{E}[\theta_\tau^2] - \sigma^2\,\mathbb{E}[\tau] = 0
    \quad\Longrightarrow\quad
    \mathbb{E}[\tau] = \frac{\mathbb{E}[\theta_\tau^2]}{\sigma^2} \approx \frac{b^2}{\sigma^2},
$$

where $\theta_\tau \approx \pm b$ (ignoring the overshoot of the last trade). **Bar length grows with the square of the threshold.**

**The feedback loop.** AFML then sets the next $\mathbb{E}[T]$ to the observed bar length. Plugging in $b = \mathbb{E}[T]\,|\mu|$:

$$
    \mathbb{E}[T]_{\text{next}} \approx \frac{\mathbb{E}[T]^2 \mu^2}{\sigma^2} = \frac{\mathbb{E}[T]^2}{T^*} .
$$

This map $x \mapsto x^2 / T^*$ has two fixed points, $0$ and $T^*$, and $T^*$ is **unstable** (the slope there is $2 > 1$):

- start below $T^*$ → every bar is shorter than expected → $\mathbb{E}[T]$ shrinks → **collapse to 1-trade bars**;
- start above $T^*$ → every bar is longer → $\mathbb{E}[T]$ grows → **explosion**.

Since $T^* = 38.5$M, any sensible starting value collapses. That is what happened when $\mathbb{E}[d_t]$ was estimated over all trades: 57.8M bars of 1 trade each.

Your version estimates $\mathbb{E}[d_t]$ as $\theta_T / T$ of *bars that just closed*. A bar closes because $|\theta_T| \ge b$, so this estimate is biased upwards: $|\theta_T/T| \ge b/T$. Each threshold inflates the next one, which is the explosion branch: 622 bars, with the last one containing 27.6M trades.

### 4. The fix: compare $\theta$ to its noise, not its drift

If $\theta$ moves like a random walk, measure it in units of its own standard deviation. Pick a target bar length $T_0$ and set

$$
    b = c\,\sigma\sqrt{T_0} .
$$

From the first-passage formula:

$$
    \mathbb{E}[\tau] \approx \frac{b^2}{\sigma^2} = c^2\, T_0 ,
$$

so with $c = 1$ the bars average about $T_0$ trades, **independent of $\mu$**. There is nothing to feed back. The rule reads like a z-score:

$$
    \text{close the bar when}\quad \frac{|\theta_T|}{\sigma\sqrt{T_0}} \ge c,
$$

i.e. "the one-sided flow is unusually large compared to normal two-sided noise". That is the information imbalance bars are meant to capture.

### 5. Correction: trades are not independent

The table in section 2 already hints at this: measured $\mathbb{E}|\theta_T|$ is about 2–3× bigger than the i.i.d. prediction $\sigma\sqrt{2T/\pi}$. One market order sweeping the book prints many same-side trades, so the $d_t$ are positively autocorrelated. The variance then becomes

$$
    \operatorname{Var}(\theta_T) = T\sigma^2 \left( 1 + 2\sum_{k=1}^{T-1}\left(1 - \frac{k}{T}\right)\rho_k \right) = T\,\sigma_{\text{LR}}^2 ,
$$

where $\rho_k$ is the autocorrelation at lag $k$ and $\sigma_{\text{LR}}$ is the long-run standard deviation. Measured variance ratio $\operatorname{Var}(\theta_T)/(T\sigma^2)$:

| $T$ | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| ratio | 3.97 | 5.18 | 8.27 | 11.16 |

Using the plain $\sigma$ would underestimate the noise and give bars ~4× shorter than $T_0$. The simplest robust estimate skips $\rho_k$ entirely: split a warm-up period (the first 3M trades, so there is no look-ahead) into blocks of $T_0$ trades and take the standard deviation of their imbalances:

$$
    b = c \cdot \operatorname{std}\big(\theta^{(1)}_{T_0}, \theta^{(2)}_{T_0}, \dots\big) \approx c\,\sigma_{\text{LR}}\sqrt{T_0} .
$$

### 6. Results on ES (6 months, 57.8M trades)

| threshold | bars | trades per bar (mean / median / max) |
|---|---|---|
| AFML, $\mathbb{E}[d]$ from closed bars (current code) | 622 | 92,866 / 2,609 / **27.6M** |
| AFML, $\mathbb{E}[d]$ over all trades | 57.8M | 1 / 1 / 46 |
| $b = \sigma\sqrt{T_0}$, $T_0 = 10{,}000$ (i.i.d. $\sigma$) | 21,011 | 2,749 / 1,751 / 39,596 |
| $b = \operatorname{std}(\theta_{T_0})$, $T_0 = 2{,}000$ | 22,703 | 2,544 / 1,620 / 42,722 |
| $b = \operatorname{std}(\theta_{T_0})$, $T_0 = 10{,}000$ | 4,441 | 13,007 / 8,186 / 121,168 |

The block-calibrated threshold gives bars about $1.3\,T_0$ long on average. That is close to the $c^2 T_0$ prediction; the gap comes from the variance ratio still rising with $T$. Nothing collapses or explodes.

### 7. Caveats

- This is my adjustment, not AFML's recipe. A version that keeps both ideas is $b = \mathbb{E}[T]|\mu| + c\,\sigma_{\text{LR}}\sqrt{\mathbb{E}[T]}$. It reduces to AFML when the drift dominates ($T \gg T^*$) and to the rule above when the noise dominates.
- The first-passage result is exact for Brownian motion. For real trades, the last trade overshoots $b$ and the tails are heavy, so treat $\mathbb{E}[\tau] \approx b^2/\sigma^2$ as an approximation.
- $\sigma_{\text{LR}}$ is estimated once from the warm-up. If market activity shifts a lot, it can be updated with an EWMA over past blocks, without reintroducing the feedback loop, since it doesn't depend on the bars themselves.
