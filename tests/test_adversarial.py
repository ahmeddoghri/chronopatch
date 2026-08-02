"""Tests for the single-series generalization finding."""

from __future__ import annotations

from chronopatch.adversarial import ADVERSARIAL_SEEDS, HOLDOUT_SEEDS, make_adversarial_series
from chronopatch.eval_v2 import _measure, _summarize, build_report
from chronopatch.forecast import conformal_forecast, coverage, mase, patch_forecast, seasonal_naive
from chronopatch.series import make_series

# --- the finding: the headline gain is real but not stable at n=14 ---------

def test_coverage_estimate_has_a_wide_confidence_interval_at_n14():
    """With only 14 test points, the observed 0.93 coverage has a 95% CI
    spanning roughly [0.79, 1.00] -- the precision the headline implies
    isn't there."""
    import math

    n, hits = 14, 13
    p_hat = hits / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    ci_low = p_hat - 1.96 * se
    assert ci_low < 0.80  # the CI reaches well below the 0.90 nominal target


def test_original_series_gain_is_real_and_stable_across_seeds():
    result = _summarize(make_series, range(40))
    assert result["mean_gain_pct"] > 15.0
    assert result["negative_gain_seeds"] == 0


# --- the finding: the gain does not survive a different series shape -------

def test_adversarial_series_shows_a_much_smaller_gain():
    original = _summarize(make_series, range(40))
    adversarial = _summarize(make_adversarial_series, ADVERSARIAL_SEEDS)
    assert adversarial["mean_gain_pct"] < original["mean_gain_pct"] / 2


def test_adversarial_series_produces_negative_gain_seeds():
    """On the bundled series family, gain is never negative across 40
    seeds. On the adversarial family it regularly is: the method
    sometimes does worse than seasonal naive."""
    result = _summarize(make_adversarial_series, ADVERSARIAL_SEEDS)
    assert result["negative_gain_seeds"] > 0


def test_adversarial_series_coverage_drops_below_target_on_some_seeds():
    result = _summarize(make_adversarial_series, ADVERSARIAL_SEEDS)
    assert result["coverage_below_0.70"] > 0


# --- held out, evaluated once ------------------------------------------------

def test_holdout_seeds_are_disjoint_from_tuning_seeds():
    assert not (set(ADVERSARIAL_SEEDS) & set(HOLDOUT_SEEDS))


def test_holdout_confirms_the_pattern():
    result = _summarize(make_adversarial_series, HOLDOUT_SEEDS)
    original = _summarize(make_series, range(40))
    assert result["mean_gain_pct"] < original["mean_gain_pct"] / 2
    assert result["negative_gain_seeds"] > 0


# --- the original benchmark is unaffected -----------------------------------

def test_original_forecast_module_untouched():
    import chronopatch.forecast as forecast_module

    assert not hasattr(forecast_module, "make_adversarial_series")


def test_original_benchmark_still_reproduces():
    values = make_series()
    train = values[:170]
    calibration = values[170:226]
    test = values[226:240]
    patch = patch_forecast(train + calibration)
    naive = seasonal_naive(train + calibration)
    interval = conformal_forecast(train, calibration)
    patch_mase = mase(test, patch, train)
    naive_mase = mase(test, naive, train)
    cov = coverage(test, interval.lower, interval.upper)
    assert round(naive_mase, 3) == 1.167
    assert round(patch_mase, 3) == 0.987
    assert round(cov, 2) == 0.93


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()


def test_measure_is_deterministic():
    a = _measure(make_adversarial_series, seed=7)
    b = _measure(make_adversarial_series, seed=7)
    assert a == b
