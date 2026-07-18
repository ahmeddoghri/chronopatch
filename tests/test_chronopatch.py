from chronopatch import conformal_forecast, make_series, patch_forecast, seasonal_naive
from chronopatch.forecast import coverage, mase


def test_series_is_deterministic() -> None:
    assert make_series(seed=3) == make_series(seed=3)


def test_patch_forecast_has_horizon() -> None:
    pred = patch_forecast(make_series()[:200], horizon=14)
    assert len(pred) == 14


def test_patch_beats_seasonal_on_fixture() -> None:
    values = make_series()
    train = values[:170]
    calibration = values[170:226]
    test = values[226:240]
    patch = patch_forecast(train + calibration)
    naive = seasonal_naive(train + calibration)
    assert mase(test, patch, train) < mase(test, naive, train)


def test_conformal_interval_covers_some_points() -> None:
    values = make_series()
    interval = conformal_forecast(values[:170], values[170:226])
    assert coverage(values[226:240], interval.lower, interval.upper) >= 0.75
