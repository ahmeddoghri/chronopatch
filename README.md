# chronopatch

A patch-based time-series forecaster with conformal intervals. It is built to
show the useful part of current time-series foundation model work without
shipping a huge checkpoint: compare recent patches, forecast the next segment,
and calibrate uncertainty on held-out residuals.

![CI](https://github.com/ahmeddoghri/chronopatch/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/chronopatch
cd chronopatch
pip install -e ".[dev]"
python -m chronopatch.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m chronopatch.benchmark`:

```text
seasonal_naive_mase  1.167
patch_knn_mase       0.987
relative_gain        15.44%
interval_coverage    0.93
mean_interval_width  15.11
```

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
