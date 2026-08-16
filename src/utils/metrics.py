"""Utility metrics and effect size helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def effect_size_cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def effect_size_cohen_d(mu1: float, mu2: float, sigma: float) -> float:
    """Cohen's d for two means."""
    return abs(mu2 - mu1) / sigma if sigma > 0 else np.nan


def summarize_ab_result(df: pd.DataFrame, metric_col: str = "conversion") -> dict:
    """Quick summary statistics for an A/B dataframe."""
    summary = {}
    for g, sub in df.groupby("group"):
        vals = sub[metric_col]
        summary[g] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "sum": float(vals.sum()) if metric_col == "conversion" else None,
        }
    return summary
