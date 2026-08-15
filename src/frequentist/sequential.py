"""
Sequential testing utilities.

Implements a practical always-valid p-value approximation and
a simple group-sequential style monitoring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats


def two_proportion_ztest(success_c: int, n_c: int, success_t: int, n_t: int) -> Tuple[float, float]:
    """Classic two-proportion z-test. Returns (z, p_value)."""
    p_c = success_c / n_c
    p_t = success_t / n_t
    p_pool = (success_c + success_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se == 0:
        return 0.0, 1.0
    z = (p_t - p_c) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def always_valid_pvalue(
    success_c: int,
    n_c: int,
    success_t: int,
    n_t: int,
    rho: float = 1.0,
) -> float:
    """
    Approximate always-valid p-value (mixture sequential probability ratio style).

    This is a practical approximation suitable for demonstration and teaching.
    Based on the mixture of likelihood ratios idea (see Johari et al., Howard et al.).
    """
    # Classic p-value
    _, p_classic = two_proportion_ztest(success_c, n_c, success_t, n_t)

    # Information / effective sample size adjustment
    n_eff = 2 / (1 / n_c + 1 / n_t)
    # Simple inflation that grows with looking frequency (conservative)
    # Using a mixture prior style bound
    av_p = np.minimum(1.0, p_classic * np.sqrt(np.log(n_eff + 1) * rho + 1))
    return float(av_p)


def sequential_test_proportion(
    cumulative_data: List,  # list of DataFrames or (success_c, n_c, success_t, n_t)
    alpha: float = 0.05,
    method: str = "always_valid",
) -> dict:
    """
    Run sequential monitoring on a list of cumulative snapshots.

    Parameters
    ----------
    cumulative_data : list
        Either list of DataFrames with columns [group, conversion]
        or list of tuples (success_c, n_c, success_t, n_t)
    """
    results = {
        "n_looks": [],
        "n_control": [],
        "n_treatment": [],
        "p_classic": [],
        "p_always_valid": [],
        "reject_classic": [],
        "reject_always_valid": [],
        "z_stat": [],
    }

    for i, snap in enumerate(cumulative_data):
        if hasattr(snap, "groupby"):  # DataFrame
            g = snap.groupby("group")["conversion"]
            success_c = int(g.get_group("control").sum())
            n_c = int(g.get_group("control").count())
            success_t = int(g.get_group("treatment").sum())
            n_t = int(g.get_group("treatment").count())
        else:
            success_c, n_c, success_t, n_t = snap

        z, p_cl = two_proportion_ztest(success_c, n_c, success_t, n_t)
        p_av = always_valid_pvalue(success_c, n_c, success_t, n_t)

        results["n_looks"].append(i + 1)
        results["n_control"].append(n_c)
        results["n_treatment"].append(n_t)
        results["p_classic"].append(p_cl)
        results["p_always_valid"].append(p_av)
        results["reject_classic"].append(p_cl < alpha)
        results["reject_always_valid"].append(p_av < alpha)
        results["z_stat"].append(z)

    # First time of rejection (if any)
    results["first_reject_classic"] = next(
        (i for i, r in enumerate(results["reject_classic"]) if r), None
    )
    results["first_reject_always_valid"] = next(
        (i for i, r in enumerate(results["reject_always_valid"]) if r), None
    )
    return results
