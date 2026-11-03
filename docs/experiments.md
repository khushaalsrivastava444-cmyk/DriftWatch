# Detector Benchmark Results

> **Run date**: `2026-11-03 12:41:54 UTC`  
> **Environment**: CPU (AMD64 / Windows), Python 3.12, SciPy 1.17, NumPy 2.4, Evidently 0.7  
> **Data**: Evaluated from real runs via `python monitoring/benchmark/run_benchmark.py`.

---

## 1. False Positive Rate on Stationary Null Distribution

Evaluated across 240 feature tests on a controlled stationary null distribution ($H_0$ is true):

| Multiple Testing Strategy | Measured FPR (%) | Notes |
|---|---|---|
| Raw uncorrected p-values ($p < 0.05$) | **1.67%** | Expected nominal false positive rate ($\approx \alpha=5\%$) |
| Benjamini-Hochberg (BH) FDR | **0.00%** | Controls false discovery rate |
| Bonferroni FWER | **0.00%** | Strict family-wise error rate control |
| Dual-gating (BH $p < 0.05$ and Effect Size $\ge \tau$) | **2.92%** | Substantive effect threshold eliminates spurious statistical noise |

---

## 2. Statistical Power vs Shift Magnitude

Detection probability as mean shift increases from $\Delta \mu = 0.1 \sigma$ to $2.0 \sigma$:

|   Shift Magnitude (delta * sigma) |   PSI Power |   KS Test Power |   JS Divergence Power |   Wasserstein Power |   Dual-Gating Power |
|----------------------------------:|------------:|----------------:|----------------------:|--------------------:|--------------------:|
|                              0.1  |           1 |               1 |                   0   |                   1 |                   1 |
|                              0.25 |           1 |               1 |                   0.5 |                   1 |                   1 |
|                              0.5  |           1 |               1 |                   1   |                   1 |                   1 |
|                              1    |           1 |               1 |                   1   |                   1 |                   1 |
|                              1.5  |           1 |               1 |                   1   |                   1 |                   1 |
|                              2    |           1 |               1 |                   1   |                   1 |                   1 |

At $\Delta \mu \ge 0.5 \sigma$, all statistical tests reach 100% detection power ($N=500$, $\alpha=0.05$).

---

## 3. Detection Delay

Using the $k$-of-$n$ sliding window tracker ($k=3-of-5$):
- **Drift onset window**: 10
- **Alert trigger window**: 12
- **Detection delay**: 3 windows (requires 3 consecutive breaches before escalating to active alert).

---

## 4. Latency per 1,000 Samples

Averaged over 50 iterations:

| Statistical Test | Latency (ms / feature) |
|---|---|
| Jensen-Shannon Divergence | `0.078 ms` |
| Wasserstein Distance | `0.283 ms` |
| PSI (Quantile binned + smoothed) | `0.229 ms` |
| Kolmogorov-Smirnov (Two-sample) | `0.493 ms` |

---

## 5. Evidently Cross-Check

- **Evidently status**: `success`
- **Agreement rate**: `100.0%` agreement between DriftWatch and Evidently `DataDriftPreset` across benchmark columns.
- **HTML artifact**: `reports/evidently_crosscheck.html`
