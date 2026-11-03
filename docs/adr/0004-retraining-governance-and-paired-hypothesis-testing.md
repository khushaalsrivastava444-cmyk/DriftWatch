# ADR 0004: Retraining Governance, Job Locks, and Paired Statistical Testing

## Status
Accepted

## Context
Automating retraining without statistical gates creates real failure modes:
1. *Drift-induced retraining loops*: If retraining fires on input drift alone, models retrain on transient noise, burning compute and risking regression.
2. *Assumption that recent models are better*: Models fit on fresh data sometimes overfit recent fluctuations and perform worse on holdouts.
3. *Concurrent jobs*: Overlapping retraining runs can collide on model artifact paths.

## Decision
Enforce four operational gates:

1. **Reject drift-only retraining**:
   - `evaluate_retraining_trigger(DRIFT_ONLY)` returns `False`.
   - Retraining only fires on `SUSTAINED_DEGRADATION` (verified performance breaches across $k$-of-$n$ windows), authenticated `MANUAL` requests, or `SCHEDULED` cadences.

2. **Atomic job locks**:
   - Before starting, the pipeline queries the DB for `RUNNING` retraining jobs. If one is found, it raises a `RuntimeError` rather than running concurrently.

3. **Paired Wilcoxon signed-rank test against active champion**:
   - Candidate models are evaluated on an identical temporal holdout split against the active champion.
   - We pair absolute errors by record: $d_i = |y_i - \hat{y}_i^{\text{champ}}| - |y_i - \hat{y}_i^{\text{cand}}|$.
   - The candidate is approved only if:
     - The Wilcoxon test rejects the null hypothesis at $\alpha=0.05$ (statistically significant error reduction).
     - Candidate MAE is strictly lower than champion MAE.

4. **Decoupled promotion and rollback**:
   - An approved candidate is registered with `is_active_champion = False`.
   - Promotion is a distinct, audited API call that updates the champion pointer in the database, reloads the forecaster in memory, and writes to `audit_logs`.
   - Rollback re-activates the previous champion with an audit log record.

## Alternatives
- *Auto-promote whenever candidate validation loss is lower*: Rejected. A small difference on a single validation set can be random noise. The paired Wilcoxon test guards against regression.
- *Async message queue (Celery/RabbitMQ)*: Deferred. In-process locking is simpler and adequate for current workloads.

## Consequences
- Statistical drift alone never triggers model replacement.
- Every promotion and rollback has an immutable database audit trail.
