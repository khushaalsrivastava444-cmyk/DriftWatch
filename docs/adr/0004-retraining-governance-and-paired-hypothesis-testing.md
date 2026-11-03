# ADR 0004: Retraining Governance, Job Locks, and Paired Statistical Hypothesis Testing

## Status
Accepted

## Context
Automated ML retraining pipelines in production present severe operational risks:
1. **Uncontrolled Retraining Loops**: If retraining triggers automatically upon statistical drift alone, models may be retrained continuously on noisy transient fluctuations, consuming compute and introducing regression risk.
2. **The "Newer Is Better" Fallacy**: Promoting candidate models solely because their training loss is low or because they were trained on recent data often degrades production performance.
3. **Race Conditions**: Concurrent retraining jobs triggered by overlapping windows can corrupt artifacts and produce non-deterministic deployments.

## Decision
We enforce three strict governance pillars:

### 1. Trigger Policy Governance & Rejection of Drift-Only Retraining
- The `DRIFT_ONLY` trigger policy is explicitly rejected by policy evaluation:
  $$\text{evaluate\_retraining\_trigger}(\text{DRIFT\_ONLY}) \implies \text{should\_retrain} = \text{False}$$
- Retraining triggers strictly require either:
  - `SUSTAINED_DEGRADATION`: $k$-of-$n$ (e.g. 2 of 3) verified performance windows breaching the 95% bootstrap CI band.
  - `MANUAL`: Audited operator trigger with authenticated credentials.
  - `SCHEDULED`: Calendar maintenance window.

### 2. Atomic Retraining Job Locks & Idempotency
- Before retraining initiates, the database is queried for active `RUNNING` jobs. If a job is active, subsequent trigger requests are rejected immediately:
  $$\text{select}(\text{RetrainingJobModel}).\text{where}(\text{status} == \text{"RUNNING"}) \implies \text{RuntimeError}$$

### 3. Paired Statistical Hypothesis Comparison vs Active Champion
- Candidate models are evaluated on an identical, chronologically isolated temporal holdout test partition against the active Champion.
- We evaluate paired absolute forecast errors:
  $$e_{\text{champ}, i} = |y_i - \hat{y}_{\text{champ}, i}|, \quad e_{\text{cand}, i} = |y_i - \hat{y}_{\text{cand}, i}|$$
- We test the non-parametric Wilcoxon Signed-Rank Test with one-sided alternative hypothesis:
  $$H_0: e_{\text{cand}} \ge e_{\text{champ}} \quad \text{vs} \quad H_1: e_{\text{cand}} < e_{\text{champ}}$$
- Candidate models are marked `APPROVED` only if:
  1. $p\text{-value} < 0.05$ (statistically significant error reduction).
  2. $\text{MAE}_{\text{cand}} < \text{MAE}_{\text{champ}}$.
  3. Candidate passes all validation rules.

### 4. Decoupled Promotion & Audited Rollback
- Registration of an `APPROVED` candidate **never** auto-promotes to active champion:
  $$\text{ModelVersion}.\text{is\_active\_champion} = \text{False}$$
- Promotion is a distinct, authenticated action that updates the champion pointer in the database, hot-reloads memory forecasters, and logs a permanent entry in `audit_logs`.
- Rollback restores the prior stable champion version with an audited transaction.

## Consequences
- No silent model regressions.
- Transparent mathematical audit trail for all deployments.
- Guaranteed idempotency and thread-safety during retraining operations.
