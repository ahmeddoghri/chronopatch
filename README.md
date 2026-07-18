# chronopatch

Time-series foundation models are billion-parameter checkpoints trained to relearn "recent history rhymes." chronopatch keeps that one idea and throws out the billion parameters.

![CI](https://github.com/ahmeddoghri/chronopatch/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Chronos, MOMENT, and Moirai made forecasting feel like a language modeling
problem, which is a fun trick but a heavy way to ask "what usually happens
after a week that looks like this one." chronopatch answers that question
directly: compare the current patch of a series to similar patches from its
own history, forecast the next segment from what came after those, then wrap
the whole thing in conformal intervals so the uncertainty is a number instead
of a vibe.

## Run it

```bash
git clone https://github.com/ahmeddoghri/chronopatch
cd chronopatch
pip install -e ".[dev]"
python -m chronopatch.benchmark
```

## Verified benchmark

Generated locally with `python -m chronopatch.benchmark`:

```text
seasonal_naive_mase  1.167
patch_knn_mase       0.987
relative_gain        15.44%
interval_coverage    0.93
mean_interval_width  15.11
```

Seasonal naive forecasting, the thing most dashboards quietly do by default,
scores 1.167 MASE. chronopatch scores 0.987, a 15.44% relative gain, and its
conformal interval actually covers 93% of test points instead of just
claiming to. A forecast with no honest uncertainty band is a guess wearing a
lab coat.

## Research trail

- Chronos, 2024: https://arxiv.org/abs/2403.07815
- MOMENT, 2024: https://arxiv.org/abs/2402.03885
- Moirai-MoE, 2024: https://arxiv.org/abs/2410.10469
- Foundation model forecasting review, 2025: https://arxiv.org/abs/2507.08858

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
