"""
Bayesian A/B testing (Beta-Binomial conjugate model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class BayesianABResult:
    # Posterior parameters
    alpha_c: float
    beta_c: float
    alpha_t: float
    beta_t: float
    # Decision metrics
    prob_treatment_better: float
    expected_loss_choose_treatment: float
    expected_loss_choose_control: float
    # Credible intervals
    ci_control: Tuple[float, float]
    ci_treatment: Tuple[float, float]
    # Point estimates
    mean_control: float
    mean_treatment: float
    # Simulation diagnostics
    n_sim: int


def bayesian_ab_test(
    success_c: int,
    n_c: int,
    success_t: int,
    n_t: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_sim: int = 50_000,
    ci_level: float = 0.95,
    seed: Optional[int] = 42,
) -> BayesianABResult:
    """
    Bayesian A/B test under Beta-Binomial model with independent Beta priors.

    Decision quantities
    -------------------
    - P(p_t > p_c | data)
    - Expected loss of choosing treatment (or control)
    """
    rng = np.random.default_rng(seed)

    # Posterior parameters
    alpha_c = prior_alpha + success_c
    beta_c = prior_beta + (n_c - success_c)
    alpha_t = prior_alpha + success_t
    beta_t = prior_beta + (n_t - success_t)

    # Monte Carlo samples from posteriors
    samples_c = rng.beta(alpha_c, beta_c, size=n_sim)
    samples_t = rng.beta(alpha_t, beta_t, size=n_sim)

    # Probability that treatment is better
    prob_better = float(np.mean(samples_t > samples_c))

    # Expected loss (opportunity rate scale)
    # Loss of choosing treatment when control is actually better
    loss_choose_t = np.maximum(samples_c - samples_t, 0)
    loss_choose_c = np.maximum(samples_t - samples_c, 0)
    expected_loss_t = float(np.mean(loss_choose_t))
    expected_loss_c = float(np.mean(loss_choose_c))

    # Credible intervals
    alpha_ci = (1 - ci_level) / 2
    ci_c = (
        float(np.quantile(samples_c, alpha_ci)),
        float(np.quantile(samples_c, 1 - alpha_ci)),
    )
    ci_t = (
        float(np.quantile(samples_t, alpha_ci)),
        float(np.quantile(samples_t, 1 - alpha_ci)),
    )

    return BayesianABResult(
        alpha_c=alpha_c,
        beta_c=beta_c,
        alpha_t=alpha_t,
        beta_t=beta_t,
        prob_treatment_better=prob_better,
        expected_loss_choose_treatment=expected_loss_t,
        expected_loss_choose_control=expected_loss_c,
        ci_control=ci_c,
        ci_treatment=ci_t,
        mean_control=float(alpha_c / (alpha_c + beta_c)),
        mean_treatment=float(alpha_t / (alpha_t + beta_t)),
        n_sim=n_sim,
    )


def simulate_bayesian_decision(
    true_p_c: float,
    true_p_t: float,
    n_per_group: int,
    threshold: float = 0.95,
    n_monte_carlo: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Simulate many experiments to estimate the frequentist properties
    of a Bayesian decision rule: "declare treatment better if P(p_t > p_c) > threshold".
    """
    rng = np.random.default_rng(seed)
    decisions = []
    probs = []

    for i in range(n_monte_carlo):
        s_c = rng.binomial(n_per_group, true_p_c)
        s_t = rng.binomial(n_per_group, true_p_t)
        res = bayesian_ab_test(s_c, n_per_group, s_t, n_per_group, n_sim=5_000, seed=None)
        decisions.append(res.prob_treatment_better > threshold)
        probs.append(res.prob_treatment_better)

    decisions = np.array(decisions)
    true_effect = true_p_t > true_p_c

    return {
        "true_effect_exists": true_effect,
        "declaration_rate": float(decisions.mean()),
        "mean_prob_better": float(np.mean(probs)),
        "threshold": threshold,
        "n_simulations": n_monte_carlo,
    }
