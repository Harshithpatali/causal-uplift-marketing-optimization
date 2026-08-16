"""Multiple testing correction procedures."""

from __future__ import annotations

from typing import List, Literal

import numpy as np
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    pvalues: List[float],
    alpha: float = 0.05,
    method: Literal["bonferroni", "holm", "fdr_bh"] = "fdr_bh",
) -> dict:
    """
    Apply multiple testing correction.

    Returns adjusted p-values and rejection decisions.
    """
    pvals = np.asarray(pvalues, dtype=float)
    reject, pvals_corrected, _, _ = multipletests(
        pvals, alpha=alpha, method=method
    )
    return {
        "method": method,
        "alpha": alpha,
        "raw_pvalues": pvals.tolist(),
        "adjusted_pvalues": pvals_corrected.tolist(),
        "reject": reject.tolist(),
        "n_rejected": int(reject.sum()),
    }
