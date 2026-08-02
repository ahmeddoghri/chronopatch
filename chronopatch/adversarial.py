"""A second, structurally different synthetic series family, to check
whether the headline gain and coverage generalize past the one series the
benchmark is measured on.

``series.make_series`` bakes in weekly (period-7) seasonality, and
``seasonal_naive``/``patch_forecast`` both hardcode ``season=7`` as their
default. The benchmark measures whether the model can exploit a period-7
signal in data that was constructed to have a period-7 signal. That is
not a fair test of whether the patch-KNN idea, comparing the current
window to similar windows earlier in the same series and forecasting from
what came after those, actually generalizes.

``make_adversarial_series`` keeps the same overall shape (trend, a
periodic component, noise) but changes every parameter that could make
the comparison easy: monthly (period-30) seasonality instead of weekly,
no discrete promo spikes or level shift, and roughly 4x the noise-to-
signal ratio. It is not an adversarial construction designed to break the
method, just a different, equally plausible real-world series shape.
"""
from __future__ import annotations

import math
import random


def make_adversarial_series(n: int = 260, seed: int = 5) -> list[float]:
    rng = random.Random(seed)
    values = []
    for t in range(n):
        monthly = 6.0 * math.sin(2 * math.pi * t / 30.0)
        trend = 50.0 + 0.02 * t
        noise = rng.gauss(0.0, 3.0)
        values.append(trend + monthly + noise)
    return values


# Seeds used while characterizing the finding above.
ADVERSARIAL_SEEDS = list(range(20))

# A disjoint set of seeds, chosen after the finding was characterized,
# evaluated exactly once in eval_v2.
HOLDOUT_SEEDS = list(range(1000, 1015))
