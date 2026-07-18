from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Forecast:
    point: list[float]
    lower: list[float]
    upper: list[float]


def seasonal_naive(history: list[float], horizon: int = 14, season: int = 7) -> list[float]:
    return [history[-season + (idx % season)] for idx in range(horizon)]


def _distance(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def patch_forecast(
    history: list[float],
    horizon: int = 14,
    lookback: int = 21,
    k: int = 6,
) -> list[float]:
    if len(history) < lookback + horizon + 1:
        return seasonal_naive(history, horizon)
    seasonal = seasonal_naive(history, horizon)
    recent_slope = (sum(history[-7:]) / 7.0 - sum(history[-28:-21]) / 7.0) / 21.0
    target = history[-lookback:]
    matches = []
    last_level = target[-1]
    target_centered = [v - last_level for v in target]
    for start in range(0, len(history) - lookback - horizon):
        context = history[start : start + lookback]
        level = context[-1]
        centered = [v - level for v in context]
        dist = _distance(target_centered, centered)
        future = history[start + lookback : start + lookback + horizon]
        delta = [value - level for value in future]
        matches.append((dist, delta))
    matches.sort(key=lambda item: item[0])
    chosen = matches[:k]
    forecast = []
    for step in range(horizon):
        patch_value = last_level + sum(delta[step] for _, delta in chosen) / len(chosen)
        trend_value = seasonal[step] + recent_slope * (step + 1)
        forecast.append(0.35 * patch_value + 0.65 * trend_value)
    return forecast


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def conformal_forecast(history: list[float], calibration: list[float], alpha: float = 0.1) -> Forecast:
    horizon = 14
    residuals = []
    window = history[:]
    for idx in range(0, len(calibration) - horizon + 1, horizon):
        truth = calibration[idx : idx + horizon]
        pred = patch_forecast(window, horizon=horizon)
        residuals.extend(abs(a - b) for a, b in zip(truth, pred))
        window.extend(truth)
    qhat = 1.25 * quantile(residuals, 1.0 - alpha)
    point = patch_forecast(history + calibration, horizon=horizon)
    return Forecast(point, [p - qhat for p in point], [p + qhat for p in point])


def mase(actual: list[float], pred: list[float], train: list[float], season: int = 7) -> float:
    denom = sum(abs(train[i] - train[i - season]) for i in range(season, len(train))) / (len(train) - season)
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual) / max(1e-9, denom)


def coverage(actual: list[float], lower: list[float], upper: list[float]) -> float:
    return sum(1 for a, lo, hi in zip(actual, lower, upper) if lo <= a <= hi) / len(actual)
