from __future__ import annotations

import math
import random


def make_series(n: int = 260, seed: int = 5) -> list[float]:
    rng = random.Random(seed)
    values = []
    for t in range(n):
        weekly = 4.2 * math.sin(2 * math.pi * t / 7.0) + 1.3 * math.cos(2 * math.pi * t / 14.0)
        promo = 8.0 if t % 28 in (0, 1, 2) else 0.0
        shift = 5.5 if t >= 175 else 0.0
        trend = 24.0 + 0.045 * t
        noise = rng.gauss(0.0, 0.7)
        values.append(trend + weekly + promo + shift + noise)
    return values
