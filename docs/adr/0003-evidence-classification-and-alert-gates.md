# ADR 0003: Evidence Classification and Alert Severity Gating

## Status
Accepted

## Context
Standard monitoring setups produce alert fatigue because they treat input shifts as model failures:
1. *Unlabeled data cannot detect concept drift*: If ground truth is delayed (such as retail sales receipts or returns), observing $P(X_{\text{curr}}) \neq P(X_{\text{ref}})$ only tells us the inputs moved. It cannot tell us whether $P(Y \mid X)$ changed.
2. *Alert fatigue*: Firing high-priority alerts on input drift causes on-call engineers to ignore alerts. In practice, models are often invariant to minor shifts in unimportant features.

## Decision
Classify all telemetry into four evidence tiers:

1. `STATISTICAL_SHIFT`:
   - Measured on unlabeled inputs ($X$) using univariate tests (PSI, KS, JS, Wasserstein) or domain classifier AUC.
   - Max severity is `WARNING`. If the shifted feature has $< 2\%$ predictive importance, demote severity to `INFO`.

2. `PREDICTION_SHIFT`:
   - Measured on model outputs ($\hat{y}$) without labels.
   - Max severity is `WARNING`.

3. `ESTIMATED_PERFORMANCE`:
   - Measured via density-ratio importance weighting or on windows with $< 80\%$ label coverage.
   - Max severity is `INFO`. Stamped with explicit assumptions (covariate shift only; invalid under concept drift).

4. `VERIFIED_PERFORMANCE`:
   - Measured on matched predictions and delayed ground truth with $\ge 80\%$ coverage.
   - Required for `CRITICAL` alerts. A `CRITICAL` alert requires the lower bound of the 95% stationary block bootstrap confidence interval to breach the baseline error threshold.

## Alternatives
- *Single alert severity with tunable thresholds*: Rejected. Threshold tuning without evidence gating still treats input shifts as outages.
- *Estimate target values using heuristic labels*: Rejected. Inventing labels introduces unquantified bias.

## Consequences
- No `CRITICAL` alerts without ground truth.
- Covariate shifts raise warnings that prompt pipeline inspection rather than emergency retraining.
