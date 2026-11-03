# ADR 0003: Evidence Classification and Alert Severity Gating

## Status
Accepted

## Context
A ubiquitous flaw in production ML monitoring systems is alert fatigue caused by conflating input feature distribution shifts with actual model performance decay:
1. **The Fallacy of Concept Drift Detection from Unlabeled Data**: When true ground-truth labels are delayed (e.g. retail demand receipts, returns, delivery logs), observing $P(X_{\text{curr}}) \neq P(X_{\text{ref}})$ only establishes covariate shift. It cannot confirm that the conditional target relationship $P(Y \mid X)$ has altered.
2. **Alert Fatigue**: Treating every feature drift breach as a high-severity incident causes operations teams to ignore alarms. In high-dimensional models, benign changes in irrelevant features frequently breach uncorrected hypothesis tests.

## Decision
We enforce a strict four-tier **Evidence Classification** across all monitoring evaluations, database records, and user interfaces:

1. `STATISTICAL_SHIFT`:
   - Derived exclusively from unlabeled input features ($X$) via univariate tests (PSI, KS, JS, Wasserstein) or multivariate domain classifiers.
   - **Severity Gate**: Maximum severity is `WARNING`. If the drifted feature possesses $< 2\%$ predictive importance in the active champion model, severity is automatically demoted to `INFO` (Drift-Impact Attribution).

2. `PREDICTION_SHIFT`:
   - Derived exclusively from the model's own output distribution ($\hat{Y}$) without labels.
   - **Severity Gate**: Maximum severity is `WARNING`.

3. `ESTIMATED_PERFORMANCE`:
   - Derived from importance-weighted density ratio estimation or windows with inadequate ground-truth label maturity ($< 80\%$ coverage).
   - **Severity Gate**: Maximum severity is `INFO`. Stamped with explicit stated assumptions (Covariate Shift Invariance) and failure modes (Concept Drift blindness).

4. `VERIFIED_PERFORMANCE`:
   - Derived strictly from matched operational predictions and verified delayed ground-truth labels ($Y$) where label coverage $\ge 80\%$.
   - **Severity Gate**: Required prerequisite for `CRITICAL` severity alerts. A `CRITICAL` alert requires a statistically significant breach of the stationary block bootstrap 95% confidence interval across consecutive monitoring windows.

## Consequences
- Zero false `CRITICAL` alerts on unlabeled feature drift.
- Transparent mathematical provenance for all operational alerts.
- Operations teams are never misled into treating covariate shifts as confirmed model failure.
