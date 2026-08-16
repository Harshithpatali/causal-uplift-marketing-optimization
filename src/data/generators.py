"""
Synthetic data generators for A/B testing and sequential experimentation.

All generators return ground-truth parameters so that Type I error,
power, and calibration can be rigorously evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuration for synthetic A/B experiments."""

    n_per_group: int = 1000
    baseline_rate: float = 0.10          # for Bernoulli
    absolute_uplift: float = 0.02        # treatment - control
    mu_control: float = 0.0              # for Normal
    sigma: float = 1.0
    random_seed: Optional[int] = 42
    metric_type: Literal["binary", "continuous"] = "binary"

    def __post_init__(self):
        if self.metric_type == "binary":
            self.treatment_rate = np.clip(
                self.baseline_rate + self.absolute_uplift, 0.0, 1.0
            )


def generate_bernoulli_ab(
    n_control: int = 1000,
    n_treatment: int = 1000,
    p_control: float = 0.10,
    p_treatment: float = 0.12,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate a classic two-group Bernoulli (conversion) A/B test.

    Returns
    -------
    df : DataFrame with columns [user, conversion]
    truth : dict with ground-truth parameters
    """
    rng = np.random.default_rng(seed)

    control = rng.binomial(1, p_control, size=n_control)
    treatment = rng.binomial(1, p_treatment, size=n_treatment)

    df = pd.DataFrame(
        {
            "group": ["control"] * n_control + ["treatment"] * n_treatment,
            "conversion": np.concatenate([control, treatment]),
        }
    )

    truth = {
        "metric_type": "binary",
        "p_control": p_control,
        "p_treatment": p_treatment,
        "absolute_uplift": p_treatment - p_control,
        "relative_uplift": (p_treatment - p_control) / p_control if p_control > 0 else np.nan,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "true_effect_exists": abs(p_treatment - p_control) > 1e-12,
    }
    return df, truth


def generate_normal_ab(
    n_control: int = 1000,
    n_treatment: int = 1000,
    mu_control: float = 0.0,
    mu_treatment: float = 0.2,
    sigma: float = 1.0,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate a two-group continuous (Normal) A/B test.
    """
    rng = np.random.default_rng(seed)

    control = rng.normal(mu_control, sigma, size=n_control)
    treatment = rng.normal(mu_treatment, sigma, size=n_treatment)

    df = pd.DataFrame(
        {
            "group": ["control"] * n_control + ["treatment"] * n_treatment,
            "value": np.concatenate([control, treatment]),
        }
    )

    truth = {
        "metric_type": "continuous",
        "mu_control": mu_control,
        "mu_treatment": mu_treatment,
        "sigma": sigma,
        "absolute_uplift": mu_treatment - mu_control,
        "cohen_d": (mu_treatment - mu_control) / sigma if sigma > 0 else np.nan,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "true_effect_exists": abs(mu_treatment - mu_control) > 1e-12,
    }
    return df, truth


def generate_sequential_stream(
    n_total_per_group: int = 5000,
    p_control: float = 0.10,
    p_treatment: float = 0.12,
    batch_size: int = 100,
    seed: Optional[int] = 42,
) -> Tuple[list[pd.DataFrame], dict]:
    """
    Generate data as sequential batches (for sequential testing demos).

    Returns a list of cumulative DataFrames and the ground-truth dict.
    """
    rng = np.random.default_rng(seed)

    control_all = rng.binomial(1, p_control, size=n_total_per_group)
    treatment_all = rng.binomial(1, p_treatment, size=n_total_per_group)

    batches = []
    for end in range(batch_size, n_total_per_group + 1, batch_size):
        df = pd.DataFrame(
            {
                "group": ["control"] * end + ["treatment"] * end,
                "conversion": np.concatenate(
                    [control_all[:end], treatment_all[:end]]
                ),
            }
        )
        batches.append(df)

    truth = {
        "metric_type": "binary",
        "p_control": p_control,
        "p_treatment": p_treatment,
        "absolute_uplift": p_treatment - p_control,
        "n_total_per_group": n_total_per_group,
        "batch_size": batch_size,
        "n_batches": len(batches),
    }
    return batches, truth


def generate_multi_variant(
    n_per_group: int = 800,
    rates: Optional[list[float]] = None,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate multi-variant (A/B/n) Bernoulli data for multiple-testing demos.
    """
    if rates is None:
        rates = [0.10, 0.11, 0.105, 0.13, 0.09]  # control + 4 treatments

    rng = np.random.default_rng(seed)
    frames = []
    for i, p in enumerate(rates):
        label = "control" if i == 0 else f"treatment_{i}"
        conversions = rng.binomial(1, p, size=n_per_group)
        frames.append(
            pd.DataFrame({"group": [label] * n_per_group, "conversion": conversions})
        )

    df = pd.concat(frames, ignore_index=True)
    truth = {
        "metric_type": "binary",
        "rates": rates,
        "n_per_group": n_per_group,
        "n_variants": len(rates),
    }
    return df, truth
