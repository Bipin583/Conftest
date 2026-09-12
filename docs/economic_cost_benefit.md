# Economic Cost-Benefit Projection

The economic report applies measured benchmark rates to a configurable organization scenario. Its dollar values are projections, not realized savings, invoices, or validated industry averages.

## Model

For annual commits `N`, full-suite minutes `T`, runner price `r_ci`, developer hourly cost `r_dev`, and assumed blocked fraction `beta`:

```text
annual CI cost   = N * T * r_ci
annual wait cost = N * (T / 60) * beta * r_dev
gross savings    = projected CI savings + projected wait-cost savings
```

`reports/economic_analysis.json` uses these scenario inputs: 25 developers, 3 commits per developer-day, 250 working days, a 45-minute suite, `$0.016` per runner-minute, `$75` per developer-hour, and a 30% wait fraction. These are editable assumptions, not measurements from a ConfTest deployment.

## Measured inputs

The producer reads duration reduction, failure recall, and escaped-commit counts from `reports/baseline_comparison.csv`, covering 183 held-out mutation observations. Only those benchmark rates and counts are measured. For the shipped selective point the input is:

- 0.0% ETR after CSV rounding;
- 100.0% observed failure recall;
- 0 observed escaped observations out of 183.

Applied to the scenario, this gives `$0` projected gross annual savings because measured wall-clock reduction rounds to zero. The point's 3.11% test-count reduction is not substituted for duration reduction.

## Escape-cost treatment

The scenario includes `$3,500` per escaped bug and four baseline annual escapes, but the report deliberately does not price or annually extrapolate benchmark escapes. An injected mutant is not a production regression, and 183 observations do not establish a reliable annual escape rate.

Instead, each strategy reports:

- observed escaped observations in the benchmark;
- projected gross savings before escape cost; and
- a break-even number of additional annual escaped bugs that would consume those savings.

Therefore the report does not establish net annual benefit. In particular, zero observed escapes must not be converted into zero expected production escape cost.

## Reading strategy comparisons

Aggressive baselines show larger projected gross savings because they measured larger ETR, but they also missed many failures in the held-out benchmark. For example, random-k projects `$259,636.22` gross savings under the scenario while recording 133 escaped observations out of 183. That dollar amount is not a recommendation; it omits a priced escape consequence by design.

## Reproduction

```bash
python scripts/evaluate.py --run economics
```

Change assumptions in the producer or its CLI inputs, then regenerate `reports/economic_analysis.json`. Report the assumptions, measured source rates, and unpriced risk together. See [Selective prediction](selective_prediction.md), [Experiments](experiments.md), and [Limitations](limitations.md).
