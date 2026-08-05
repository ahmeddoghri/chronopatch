# chronopatch

Time-series foundation models are billion-parameter checkpoints trained to relearn "recent history rhymes." chronopatch keeps that one idea and throws out the billion parameters.

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

**Update:** that 15.44% gain and 93% coverage are both measured on one
hardcoded series with weekly seasonality, exactly matching this model's
hardcoded `season=7` default, and the coverage claim rests on just 14 test
points. Across 40 seeds of the same series family the gain holds up; on a
different but equally realistic series shape (monthly seasonality, more
noise, no promo spikes), mean gain drops from 19% to single digits and
some runs go negative. `python -m chronopatch.eval_v2` runs the honest
comparison. Details below.

## The headline is measured on one series that matches the model's own assumptions

`series.make_series` bakes in weekly (period-7) seasonality. `seasonal_naive`
and `patch_forecast` both hardcode `season=7` as their default. The
benchmark measures whether the model can exploit a period-7 signal in
data built to have a period-7 signal, and the coverage figure rests on
just 14 test points, small enough that a 95% confidence interval on the
observed 0.93 spans roughly [0.79, 1.00].

```bash
python -m chronopatch.eval_v2
```
```
series family                  n   mean gain  neg. gain  mean cov  cov<0.70
original (bundled)            40      19.26%        0/40     0.914        0/40
adversarial (tuning)          20       6.30%        4/20     0.861        2/20
adversarial (holdout)         15       6.75%        2/15     0.890        1/15
```

Across 40 seeds of the *same* series family (weekly seasonality, promo
spikes, a discrete level shift), the ~19% gain and ~0.91 coverage are real
and stable, not a lucky draw on seed=5. On a structurally different but
equally plausible series, monthly seasonality instead of weekly, no
promo/shift, roughly 4x the noise, mean gain drops to single digits and a
meaningful share of runs show *negative* gain (the patch method doing
worse than plain seasonal naive) and coverage well under the 0.90 target.
Manually correcting the hardcoded `season=7` to the true period (30) on
the harder series does not reliably fix this either; the patch-matching
approach itself doesn't transfer cleanly to a noisier, less repetitive
signal.

None of this is a code bug: `forecast.py` and `benchmark.py` do exactly
what they say, and the published numbers above still reproduce exactly.
It's an honest scope statement the original benchmark didn't provide:
this method's real strength is on data that looks like the series it was
tuned against, and the gap between "works here" and "works in general" is
larger than one series and 14 test points can tell you.

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
