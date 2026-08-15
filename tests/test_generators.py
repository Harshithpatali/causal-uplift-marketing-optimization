"""Basic smoke tests for synthetic generators and core stats."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.generators import generate_bernoulli_ab, generate_normal_ab, generate_sequential_stream
from src.frequentist.power_analysis import sample_size_proportion, power_proportion
from src.bayesian.ab_testing import bayesian_ab_test
from src.frequentist.sequential import two_proportion_ztest


def test_bernoulli_generator():
    df, truth = generate_bernoulli_ab(500, 500, 0.1, 0.15, seed=1)
    assert len(df) == 1000
    assert set(df["group"].unique()) == {"control", "treatment"}
    assert abs(truth["absolute_uplift"] - 0.05) < 1e-9


def test_normal_generator():
    df, truth = generate_normal_ab(300, 300, 0.0, 0.5, 1.0, seed=2)
    assert len(df) == 600
    assert truth["cohen_d"] == 0.5


def test_power_and_sample_size():
    n = sample_size_proportion(0.1, 0.12, alpha=0.05, power=0.8)
    assert n > 1000
    pwr = power_proportion(n, 0.1, 0.12, alpha=0.05)
    assert 0.75 < pwr < 0.90


def test_bayesian_ab():
    res = bayesian_ab_test(100, 1000, 130, 1000, n_sim=5000, seed=0)
    assert 0.5 < res.prob_treatment_better < 1.0
    assert res.mean_treatment > res.mean_control


def test_ztest():
    z, p = two_proportion_ztest(100, 1000, 100, 1000)
    assert abs(z) < 0.5
    assert p > 0.5


if __name__ == "__main__":
    test_bernoulli_generator()
    test_normal_generator()
    test_power_and_sample_size()
    test_bayesian_ab()
    test_ztest()
    print("All smoke tests passed.")
