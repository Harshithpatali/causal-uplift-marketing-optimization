"""Power analysis and sample size calculations."""

from __future__ import annotations

import numpy as np
from scipy import stats


def sample_size_proportion(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """
    Required sample size *per group* for two-proportion z-test.
    """
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n = numerator / (p2 - p1) ** 2
    return int(np.ceil(n))


def power_proportion(
    n_per_group: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Statistical power of a two-proportion z-test.
    """
    se_null = np.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2) / n_per_group)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)
    effect = abs(p2 - p1)

    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        # Power ≈ P(|Z| > z_crit | H1)
        z = effect / se_alt
        power = 1 - stats.norm.cdf(z_crit - z) + stats.norm.cdf(-z_crit - z)
    else:
        z_crit = stats.norm.ppf(1 - alpha)
        z = effect / se_alt
        power = 1 - stats.norm.cdf(z_crit - z)
    return float(np.clip(power, 0.0, 1.0))


def power_mean(
    n_per_group: int,
    mu1: float,
    mu2: float,
    sigma: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Power of a two-sample t-test (equal variance)."""
    effect = abs(mu2 - mu1)
    se = sigma * np.sqrt(2 / n_per_group)
    df = 2 * n_per_group - 2

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ncp = effect / se
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)
        ncp = effect / se
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    return float(np.clip(power, 0.0, 1.0))


def power_curve_proportion(
    p1: float,
    absolute_uplift: float,
    n_range: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return power for a range of sample sizes."""
    p2 = np.clip(p1 + absolute_uplift, 0.0, 1.0)
    return np.array([power_proportion(int(n), p1, p2, alpha) for n in n_range])
