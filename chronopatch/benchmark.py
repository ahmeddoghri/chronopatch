from __future__ import annotations

from .forecast import conformal_forecast, coverage, mase, patch_forecast, seasonal_naive
from .series import make_series


def main() -> None:
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
    width = sum(hi - lo for lo, hi in zip(interval.lower, interval.upper)) / len(interval.point)

    print("chronopatch benchmark: patch forecast with conformal intervals")
    print(f"seasonal_naive_mase  {naive_mase:.3f}")
    print(f"patch_knn_mase       {patch_mase:.3f}")
    print(f"relative_gain        {(naive_mase - patch_mase) / naive_mase:.2%}")
    print(f"interval_coverage    {cov:.2f}")
    print(f"mean_interval_width  {width:.2f}")


if __name__ == "__main__":
    main()
