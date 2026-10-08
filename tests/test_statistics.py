"""Checks for the block-averaging error estimate and seed aggregation."""

import numpy as np
import pytest

from qpmhc import core


def ar1_series(n, phi, seed=0):
    """AR(1) process x_t = phi * x_{t-1} + noise, with unit stationary variance."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_block_average_independent_samples_matches_naive_sem():
    x = np.random.default_rng(1).normal(size=4096)
    out = core.block_average(x)
    assert out["mean"] == pytest.approx(x.mean())
    # For independent samples blocking should not inflate the error much.
    assert out["sem"] == pytest.approx(out["sem_naive"], rel=0.35)


def test_block_average_correlated_samples_inflates_sem():
    phi = 0.9                                   # statistical inefficiency g = (1+phi)/(1-phi) = 19
    x = ar1_series(2**15, phi)
    out = core.block_average(x)
    g_expected = (1 + phi) / (1 - phi)
    assert out["sem"] > 3.0 * out["sem_naive"]
    assert out["sem"] == pytest.approx(out["sem_naive"] * np.sqrt(g_expected), rel=0.35)
    assert out["n_eff"] < x.size / 5


def test_block_average_constant_series():
    out = core.block_average(np.full(100, -3.0))
    assert out["mean"] == -3.0
    assert out["sem"] == 0.0


def test_summarize_seeds():
    runs = [{"mean": m, "acc_rate": a} for m, a in [(-10.0, 20.0), (-12.0, 30.0), (-11.0, 25.0)]]
    agg = core.summarize_seeds(runs)
    assert agg["n_seeds"] == 3
    assert agg["mean"] == pytest.approx(-11.0)
    assert agg["std"] == pytest.approx(1.0)
    assert agg["sem"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert agg["acc_mean"] == pytest.approx(25.0)


def test_monte_carlo_reports_acceptance_and_sem():
    res = core.run_monte_carlo(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]),
                               np.array([[4.0, 0.0, 0.0]]), np.array([-1.0]),
                               n_steps=200, rng=np.random.default_rng(3))
    assert 0.0 <= res["acc_rate"] <= 100.0
    assert np.isfinite(res["sem"]) and res["sem"] >= 0.0
    assert res["n_production"] == 201 - int(0.2 * 201)
