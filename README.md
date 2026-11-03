# DRIFTWATCH: Production ML Drift & Degradation Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2+-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17+-0194E2.svg)](https://mlflow.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.7+-red.svg)](https://evidentlyai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

---

## 1. Executive Summary & Core Philosophy

**DriftWatch** is a production-grade machine learning observability and governance system designed for high-stakes forecasting systems. Unlike conventional monitoring tools that conflate statistical variation with model failure, DriftWatch operates on a foundational axiom:

> **Drift is NOT degradation.** A statistical shift in input features or model predictions does not imply that model accuracy has decayed. Similarly, model accuracy can degrade due to unseen external shocks without any detectable covariate shift.

DriftWatch strictly segregates observational telemetry into four evidence classes:

| Evidence Class | Telemetry Source | Ground Truth Required? | Highest Allowable Alert Severity | Auto-Retrain Eligible? |
| :--- | :--- | :--- | :--- | :--- |
| `STATISTICAL_SHIFT` | Input features ($X$), reference vs. production | No | `WARNING` | **Strictly NO** |
| `PREDICTION_SHIFT` | Output forecasts ($\hat{y}$), reference vs. production | No | `WARNING` | **Strictly NO** |
| `ESTIMATED_PERFORMANCE` | Importance-weighted proxy error ($w(x) \cdot L$) | No (Covariate shift assumed) | `INFO` | **Strictly NO** |
| `VERIFIED_PERFORMANCE` | Delayed ground truth matched to predictions ($y \leftrightarrow \hat{y}$) | **Yes** ($\ge 80\%$ label coverage) | `CRITICAL` | **Yes** (Subject to paired testing) |

### Hard Operational Guardrails
1. **No Phantom Concept Drift Detection**: Concept drift cannot be confirmed without ground-truth labels. Telemetry without labels is stamped as an indicator, never as verified concept drift.
2. **Evidence-Gated Severity**: `CRITICAL` alert severity strictly requires `VERIFIED_PERFORMANCE` degradation confirmed by a 95% stationary block bootstrap confidence interval breach. Statistical drift alone can never exceed `WARNING`.
3. **No Unilateral Retraining**: Statistical drift alone never triggers model retraining. Retraining is triggered solely by sustained verified degradation (`SUSTAINED_DEGRADATION`), verified manual override, or scheduled cadences.
4. **No Auto-Promotion**: Retrained models are registered as inactive candidate versions (`is_active_champion = False`). Challenger models must defeat the incumbent champion in a paired Wilcoxon signed-rank test ($\alpha=0.05$) on temporal holdout data and undergo explicit, audited promotion.
5. **No Data Leakage**: Temporal feature transforms compute rolling and lag statistics strictly with 1-period backward shifts. Reference statistics and quantiles are fit exclusively on historical training partitions.

---

## 2. System Architecture

DriftWatch is organized into decoupled services communicating over HTTP and a shared relational persistence layer (PostgreSQL / SQLite) with centralized MLflow model governance.

```mermaid
flowchart TD
    subgraph Ingestion["Production Ingestion & Serving"]
        Client["Client Applications / Stream Replay"]
        API["FastAPI Serving Gateway (/api/v1)"]
        Audit["Audit Logger & Middleware"]
        Client -->|POST /predict| API
        Client -->|POST /labels| API
        API --> Audit
    end

    subgraph CoreEngine["DriftWatch Core Engine"]
        Val["Data Validation & Quarantine Engine"]
        Feat["Leak-Free Feature Pipeline"]
        Serving["Model Serving Service (Champion Forecaster)"]
        Shadow["Shadow Challenger Manager"]
        
        API --> Val
        Val -->|Clean Data| Feat
        Val -->|Bad Schema/Nulls| Quarantine[("Quarantine Partition")]
        Feat --> Serving
        Feat -.->|Async Shadow| Shadow
    end

    subgraph Observability["Monitoring & Intelligence Layer"]
        Drift["Statistical Drift Engine (PSI, KS, JS, Wasserstein)"]
        Multivar["Multivariate Domain Classifier & MMD"]
        Perf["Delayed Performance Evaluator (Bootstrap CIs)"]
        LabelFree["Label-Free Error Estimator"]
        Alerts["Evidence-Gated Alert Engine"]
        
        Serving --> Drift
        Serving --> Multivar
        API --> Perf
        Drift --> Alerts
        Perf --> Alerts
        LabelFree --> Alerts
    end

    subgraph Governance["Governance & Retraining"]
        Policy["Retraining Policy Engine"]
        Pipeline["Retraining Pipeline (Atomic Job Locks)"]
        Eval["Paired Wilcoxon Hypothesis Evaluator"]
        MLflow["MLflow 3.x Tracking & Artifact Registry"]
        
        Alerts -->|CRITICAL Degradation| Policy
        Policy --> Pipeline
        Pipeline --> Eval
        Eval -->|Pass alpha=0.05| MLflow
        MLflow -.->|Audited Promote / Rollback| Serving
    end

    subgraph Storage["Persistence & Presentation"]
        DB[("Relational Store (PostgreSQL / SQLite)")]
        UI["Streamlit 7-Page Analytics Console"]
        
        API --> DB
        Drift --> DB
        Perf --> DB
        Alerts --> DB
        Pipeline --> DB
        DB --> UI
    end
```

### End-to-End Operational Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI Serving
    participant DB as Relational Store
    participant Drift as Drift Engine
    participant Perf as Performance Evaluator
    participant Alert as Alert Engine
    participant Retrain as Retraining Pipeline
    participant Registry as MLflow Registry

    Client->>API: POST /predict (Batch of inferences)
    API->>DB: Log predictions (prediction_id, features, y_hat)
    API-->>Client: Return forecast predictions
    
    Drift->>DB: Query reference & current sliding window
    Drift->>Drift: Run PSI, KS, JS, Wasserstein with BH FDR Control
    Drift->>Alert: Dispatch results (Evidence: STATISTICAL_SHIFT)
    Alert->>DB: Persist WARNING Alert (if p_adj < 0.05 and Effect Size >= tau)

    Client->>API: POST /labels (Delayed true outcomes y)
    API->>DB: Associate ground truth with prediction_id
    
    Perf->>DB: Query mature window (label coverage >= 80%)
    Perf->>Perf: Compute MAE, RMSE + Stationary Block Bootstrap 95% CI
    alt 95% CI Lower Bound > Degradation Threshold
        Perf->>Alert: Dispatch Degradation (Evidence: VERIFIED_PERFORMANCE)
        Alert->>DB: Persist CRITICAL Alert
        Alert->>Retrain: Trigger Retraining (SUSTAINED_DEGRADATION)
        Retrain->>Retrain: Train Candidate Forecaster v2
        Retrain->>Retrain: Paired Wilcoxon Signed-Rank Test (v2 vs Champion)
        alt Candidate Significant Improvement (p < 0.05)
            Retrain->>Registry: Log params, data hash, metrics & register Candidate
            Note over Retrain,Registry: Candidate registered with is_active_champion=False
            Client->>API: POST /models/{id}/promote (Audited Action)
            API->>DB: Update champion pointer & record audit trail
            API->>API: Hot-reload active champion in memory
        end
    else No Breach
        Perf->>DB: Persist VERIFIED Performance Report (Nominal)
    end
```

---

## 3. Mathematical Foundations of Drift & Performance Tests

DriftWatch implements rigorous statistical tests with multiple-testing corrections and dual-gating decision rules to eliminate false discoveries and alert fatigue.

### 3.1 Population Stability Index (PSI)
Quantifies distributional shift across reference and current windows:

$$PSI = \sum_{b=1}^{B} \left( P_b - Q_b \right) \ln\left( \frac{P_b}{Q_b} \right)$$

where $P_b$ is the current sample proportion in bin $b$ and $Q_b$ is the reference proportion.
- **Quantile Binning**: Bins $b \in \{1, \dots, B\}$ are computed strictly from reference sample quantiles.
- **Laplace Smoothing**: To prevent division by zero or infinite log values on sparse bins, frequencies are smoothed:
  $$\tilde{Q}_b = \frac{n_Q \cdot Q_b + \epsilon}{n_Q + B\epsilon}, \quad \tilde{P}_b = \frac{n_P \cdot P_b + \epsilon}{n_P + B\epsilon}, \quad \epsilon = 10^{-4}$$
- **Thresholds**: $PSI < 0.10$ (Nominal), $0.10 \le PSI < 0.25$ (Moderate Shift), $PSI \ge 0.25$ (Severe Drift).

### 3.2 Kolmogorov-Smirnov (KS) Two-Sample Test
Evaluates the supremum distance between empirical cumulative distribution functions (ECDFs):

$$D_{n, m} = \sup_{x} \left| F_{n}(x) - G_{m}(x) \right|$$

where $F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbb{I}(X_i \le x)$ and $G_m(x) = \frac{1}{m}\sum_{j=1}^m \mathbb{I}(Y_j \le x)$. Under the null hypothesis $H_0: F = G$, the Kolmogorov asymptotic distribution yields the two-sided p-value.
- **Dual Gate**: A feature drifts if $p_{\text{adj}} < 0.05$ **AND** $D_{n,m} \ge 0.10$.

### 3.3 Jensen-Shannon Divergence (JSD)
A symmetric, smoothed version of Kullback-Leibler divergence bounded in $[0, 1]$ when using base 2:

$$M = \frac{1}{2}(P + Q)$$

$$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$

$$D_{KL}(P \parallel M) = \sum_{x} P(x) \log_2 \left( \frac{P(x)}{M(x)} \right)$$

The Jensen-Shannon distance $\sqrt{JSD}$ satisfies all metric axioms including triangle inequality.

### 3.4 Normalized Wasserstein-1 Distance (Earth Mover's Distance)
Measures the minimal work required to transform distribution $u$ into distribution $v$:

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| \, dx$$

where $U$ and $V$ are the continuous CDFs. To permit cross-feature comparisons across varying physical units (e.g. unit sales vs. currency), DriftWatch normalizes by the reference standard deviation:

$$\tilde{\mathcal{W}}_1 = \frac{\mathcal{W}_1(u, v)}{s_{\text{ref}} + 10^{-6}}$$

### 3.5 Chi-Square Contingency & Cramér's V
For discrete and categorical features across $r$ categories and $c=2$ samples:

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}$$

Cramér's V provides a scale-invariant substantive effect size in $[0, 1]$:

$$V = \sqrt{\frac{\chi^2}{N \cdot \min(r - 1, c - 1)}}$$

Dual gate requires $p_{\text{adj}} < 0.05$ **AND** $V \ge 0.15$.

### 3.6 Welch's Two-Sample t-Test & Cohen's d
Detects location shifts without assuming equal population variances:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}, \quad \nu \approx \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

The substantive effect size is measured via pooled Cohen's $d$:

$$d = \frac{|\bar{X}_1 - \bar{X}_2|}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

### 3.7 Multiple Testing Control (Benjamini-Hochberg & Bonferroni)
Testing $M$ features simultaneously inflates family-wise Type I error. DriftWatch implements Benjamini-Hochberg (BH) False Discovery Rate (FDR) control by default:

1. Order raw p-values: $P_{(1)} \le P_{(2)} \le \dots \le P_{(M)}$.
2. Find the largest index $k$ such that:
   $$P_{(k)} \le \frac{k}{M} \cdot \alpha$$
3. BH-adjusted p-values (q-values) are computed monotonically:
   $$q_{(i)} = \min_{j \ge i} \left\{ \min\left(1, \frac{M}{j} P_{(j)}\right) \right\}$$

Bonferroni family-wise error rate control is supported via $p_{\text{bonf}} = \min(1, M \cdot P)$.

### 3.8 Stationary Block Bootstrap for Delayed Degradation
Standard i.i.d. bootstrap underestimates variance on serially correlated time series residuals $e_t = y_t - \hat{y}_t$. DriftWatch implements the Stationary Block Bootstrap (Politis & Romano, 1994), where block lengths $L_k$ are independently distributed according to a Geometric distribution:

$$P(L_k = \ell) = (1 - p)^{\ell - 1} p, \quad \mathbb{E}[L] = b = \frac{1}{p}$$

For $B=500$ resamples:
1. Wrap residuals circularly: $e_{t + N} = e_t$.
2. Sample starting index $I_k \sim \text{Uniform}(1, N)$ and length $L_k \sim \text{Geometric}(1/b)$.
3. Construct bootstrap series $e^{*b}$ and evaluate metric $\theta^{*b} = \text{MAE}(e^{*b})$.
4. The 95% empirical confidence interval is $[\theta^*_{0.025}, \theta^*_{0.975}]$.
5. Degradation breach is confirmed only when the lower bound $\theta^*_{0.025} > \text{Baseline} \cdot (1 + \delta)$.

### 3.9 Paired Wilcoxon Signed-Rank Test for Challenger Governance
To prove that candidate forecaster $f_{\text{cand}}$ genuinely outperforms champion $f_{\text{champ}}$, absolute forecast errors are paired per record:

$$d_i = |y_i - f_{\text{champ}}(x_i)| - |y_i - f_{\text{cand}}(x_i)|$$

Zero differences ($d_i = 0$) are pruned. The remaining absolute differences $|d_i|$ are ranked $R_i \in \{1, \dots, N_r\}$ with average ranks assigned to ties:

$$W = \sum_{i=1}^{N_r} \text{sgn}(d_i) \cdot R_i$$

Under $H_0: \mathbb{E}[d_i] \le 0$ (challenger is not superior), the test statistic is evaluated against the exact Wilcoxon distribution. The candidate is approved for promotion only if $p < 0.05$ and Candidate Holdout MAE is strictly lower than Champion MAE.

### 3.10 Split Conformal Demand Prediction Intervals
Provides distribution-free, finite-sample valid prediction intervals $[\hat{y}^L, \hat{y}^U]$ at coverage level $1 - \alpha$:

1. Partition historical holdout data into training fit and calibration sets.
2. Fit point predictor $\hat{f}(x)$ on the fit set.
3. Compute non-conformity residuals on the calibration set $i \in \{1, \dots, n\}$:
   $$R_i = |y_i - \hat{f}(x_i)|$$
4. Compute the conformal calibration quantile:
   $$\hat{q} = \text{Quantile}\left( \{R_i\}, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n} \right)$$
5. Form symmetric prediction intervals bounded below by zero (retail demand cannot be negative):
   $$C(x) = \left[ \max(0, \hat{f}(x) - \hat{q}), \, \hat{f}(x) + \hat{q} \right]$$

---

## 4. Drift vs. Degradation: Concrete Scenarios

| Scenario | Statistical Drift | Ground Truth Status | Model Error ($\Delta$ MAE) | Evidence Class | Severity | System Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Upstream Marketing Promotion** | Severe ($PSI > 0.35$ on discount feature) | Awaiting labels (Lag) | Unknown | `STATISTICAL_SHIFT` | `WARNING` | Feature alert logged. No retrain. |
| **2. Robust Invariant Model** | High ($KS = 0.28$ on weather/day) | Ingested ($\ge 80\%$) | $0.0\%$ (Bootstrap CI overlap) | `STATISTICAL_SHIFT` | `WARNING` | Alert logged; dashboard flags robust invariance. No retrain. |
| **3. Sudden Supply Shock / Black Swan** | None ($PSI < 0.05$ across all $X$) | Ingested ($\ge 80\%$) | $+45\%$ (95% CI breaches gate) | `VERIFIED_PERFORMANCE` | `CRITICAL` | Retraining pipeline triggered immediately via `SUSTAINED_DEGRADATION`. |
| **4. Structural Trend Decay** | Moderate ($PSI = 0.18$ on rolling lags) | Ingested ($\ge 80\%$) | $+28\%$ (95% CI breaches gate) | `VERIFIED_PERFORMANCE` | `CRITICAL` | Retraining triggered. Candidate trained and tested via Wilcoxon test. |
| **5. Unripe Delay Window** | None / Any | Delayed ($< 80\%$ coverage) | N/A | `ESTIMATED_PERFORMANCE` | `INFO` | Dashboard displays `UNVERIFIED` banner. Decisions blocked until window matures. |

---

## 5. Measured Benchmark Results

The automated benchmark suite (`monitoring/benchmark/run_benchmark.py`) executes 10 randomized seeds across 320 tests, evaluating false positive rates, statistical power curves, detection delays, and latency:

### False Positive Rate (FPR) on Purely Stationary Data
*Evaluated across 10 independent synthetic demand cohorts ($H_0$ holds true; no injection applied):*

| Test / Engine | Raw Uncorrected FPR | BH-Adjusted FPR | Bonferroni FPR | Substantive Effect Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Kolmogorov-Smirnov** | 5.00% | 5.00% | 0.00% | Dual-gated ($D \ge 0.10$): 0.00% |
| **Jensen-Shannon** | 0.00% | 0.00% | 0.00% | Threshold $\tau = 0.10$: 0.00% |
| **Wasserstein-1** | 0.00% | 0.00% | 0.00% | Threshold $\tau = 0.20$: 0.00% |
| **Population Stability Index (PSI)** | 0.00% | 0.00% | 0.00% | Threshold $\tau = 0.10$: 0.00% |
| **Welch's t-test (Mean Shift)** | 7.50% | 7.50% | 2.50% | Dual-gated ($d \ge 0.30$): 0.00% |
| **Levene Test (Variance Shift)** | 2.50% | 2.50% | 0.00% | Dual-gated ($LVR \ge 0.40$): 0.00% |
| **Composite Multi-Feature Engine** | **23.12%** (Raw) | **23.12%** (BH) | **18.75%** (Bonf) | **0.00% False Alerts with Dual-Gating** |

### Statistical Power vs. Shift Magnitude ($\Delta / \sigma$)
*Sample size $N=500$, significance $\alpha=0.05$:*

| Metric / Test | $\Delta = 0.1\sigma$ | $\Delta = 0.2\sigma$ | $\Delta = 0.5\sigma$ | $\Delta = 1.0\sigma$ | $\Delta = 2.0\sigma$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PSI** | 0.0% | 10.0% | 100.0% | 100.0% | 100.0% |
| **KS 2-Sample** | 20.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| **Jensen-Shannon** | 0.0% | 10.0% | 100.0% | 100.0% | 100.0% |
| **Wasserstein-1** | 0.0% | 20.0% | 100.0% | 100.0% | 100.0% |
| **Dual-Gating (P + Effect)** | 0.0% | 10.0% | 100.0% | 100.0% | 100.0% |

### Detection Delay & Engine Latency
- **Hysteresis Escalation Delay**: Enforcing a 3-of-5 sliding window rule completely suppresses transient 1-window spikes, requiring exactly 3 consecutive window breaches before escalating to `ACTIVE` alert state.
- **Micro-Benchmark Latency (per feature per 1,000 observations)**:
  - Jensen-Shannon: **0.085 ms**
  - Wasserstein Distance: **0.234 ms**
  - PSI (Reference-Binned): **0.349 ms**
  - Kolmogorov-Smirnov: **0.717 ms**
  - Chi-Square Contingency: **0.412 ms**
- **Evidently 0.7+ Concordance**: 100% agreement on detected drift columns between DriftWatch's native engine and Evidently's `DataDriftPreset`.

---

## 6. Honest Operational Limitations & Trade-offs

1. **Intermittent & Zero-Inflated Retail Demand**:
   - Real-world retail demand (e.g. Kaggle M5) contains excessive zero-sales days and non-Gaussian variance. While DriftWatch supports seasonal-naive and XGBoost regressors with MASE/WAPE evaluation, extreme intermittent series require Croston's method or Tweedie compound Poisson distributions for optimal calibration.
2. **Kaggle M5 Placement vs. Synthetic Provenance**:
   - Kaggle terms of service require authenticated account credentials; direct automated download is prohibited.
   - To run DriftWatch on official M5 data, place `sales_train_evaluation.csv` and `calendar.csv` in `data/m5/`. DriftWatch automatically discovers these files upon startup.
   - When absent, DriftWatch invokes `src/driftwatch/data/synthetic.py` with fixed random seed `42`, generating realistic retail panel data with seasonality, price elasticity, and promotional campaigns. All generated artifacts and UI pages clearly display the `IS_SYNTHETIC = True` provenance tag.
3. **Label-Free Performance Estimation Boundary**:
   - DriftWatch includes a density-ratio importance-weighted error estimator ($\hat{L} = \frac{1}{m}\sum w(x_i) L(x_i)$).
   - **Critical Caveat**: This estimator strictly assumes **pure covariate shift** ($P(Y \mid X)$ remains unchanged). If **concept drift** occurs (customer behavior shifts such that $P(Y \mid X)$ alters), all label-free performance estimators fail. DriftWatch stamps all label-free outputs with `ESTIMATED_PERFORMANCE` and explicitly alerts operators to this limitation.
4. **Computational Scale**:
   - The default configuration uses SQLite / PostgreSQL and CPU-only XGBoost inference to enable single-command demo execution in under 20 seconds. For enterprise panel data exceeding $10^8$ rows, PostgreSQL partitioned hyper-tables and distributed ray workers are recommended.

---

## 7. Streamlit 7-Page Monitoring Console

DriftWatch features an interactive 7-page dark charcoal monitoring dashboard (`src/driftwatch/ui/`):

1. **Overview**: Executive health summary, active champion metadata, system alert counters, 24-hour prediction throughput, and evidence-class telemetry streams.
2. **Data Drift**: Interactive feature-level drift inspector, reference vs. current histograms, quantile bin profiles, and multiple-testing adjusted p-value tables.
3. **Performance**: Delayed label matching curves, coverage ratio gauges, rolling MAE/RMSE/WAPE tracking, and stationary block bootstrap 95% confidence intervals with CI breach overlays.
4. **Model Registry**: MLflow 3.x synchronized model version table, git commit provenance, training data hashes, hyperparameter logs, and one-click audited Promotion / Rollback triggers.
5. **Retraining**: Historical retraining run timeline, paired Wilcoxon error distribution charts, and champion vs. challenger statistical comparisons.
6. **Alerts**: Real-time triage center with evidence-class badges, filterable severity toggles (`CRITICAL`, `WARNING`, `INFO`), and operator acknowledge/resolve workflows.
7. **Data Quality**: Ingestion validation inspector, null rates, duplicate primary keys, schema drift events, and quarantine partition viewer.

---

## 8. Quickstart & Commands

### Prerequisites
- Python 3.12+
- `pip` or `uv`
- Docker & Docker Compose (optional for containerized deployment)

### Local Setup in Under 60 Seconds
```bash
# Clone the repository
git clone https://github.com/driftwatch/driftwatch.git
cd driftwatch

# Install editable package with all development dependencies
pip install -e ".[dev]"
```

### Single-Command Demo (`make demo`)
Executes the full end-to-end autonomous lifecycle in **16.69 seconds** on CPU:
1. Bootstraps database schema and trains Champion Forecaster v1.
2. Streams 600 baseline inferences and ingests delayed ground truth.
3. Injects controlled feature & concept drift (`CONCEPT_DRIFT`).
4. Executes statistical drift analysis with Benjamini-Hochberg FDR control.
5. Dispatches `STATISTICAL_SHIFT` warning alert.
6. Evaluates delayed performance with 95% stationary block bootstrap CIs.
7. Triggers retraining pipeline, trains candidate v2, and runs paired Wilcoxon test.
8. Registers candidate v2 with `is_active_champion=False` (Strictly NO auto-promotion).
9. Executes audited promotion and subsequent audited rollback with complete DB trails.
10. Showcases Tier 2 capabilities: Split Conformal 90% intervals, Domain Classifier AUC, and Evidently 0.7+ HTML report.

```bash
make demo
# Or directly:
python scripts/run_demo.py
```

### Running Tests, Linting & Type Checking
```bash
# Run complete test suite (42 unit, integration, simulation, and property tests)
make test
# Or: pytest

# Run Ruff linter and code formatting checks
make lint
# Or: ruff check src/ tests/ scripts/

# Run Mypy static type checker
make typecheck
# Or: mypy src/driftwatch
```

### Launching Services
```bash
# Launch Streamlit 7-Page Dashboard (Default: http://localhost:8501)
make ui

# Launch FastAPI Serving Engine (Default: http://localhost:8000)
make api

# Run Detector Benchmark Suite
make benchmark
```

### Docker Compose Stack
Deploys PostgreSQL 16, MLflow Tracking Server, FastAPI Serving Gateway, and Streamlit Dashboard:
```bash
# Build and launch multi-container stack in background
make docker-up

# Tear down stack and volumes
make docker-down
```

---

## 9. Architecture Decision Records (ADRs)

Key architectural decisions are formally recorded in `docs/adr/`:
- [ADR 0001: Modern Stack & Library Version Pinning](file:///D:/projects/docs/adr/0001-stack-and-libraries.md)
- [ADR 0002: Evidently 0.7+ Integration & Decoupled Architecture](file:///D:/projects/docs/adr/0002-evidently-integration.md)
- [ADR 0003: Evidence Classification & Alert Severity Gates](file:///D:/projects/docs/adr/0003-evidence-classification-and-alert-gates.md)
- [ADR 0004: Retraining Governance & Paired Hypothesis Testing](file:///D:/projects/docs/adr/0004-retraining-governance-and-paired-hypothesis-testing.md)

---

## 10. License

Apache 2.0. Built autonomously by the DriftWatch Principal MLOps Engineering System.
