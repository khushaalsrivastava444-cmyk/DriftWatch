# ADR 0001: Technology Stack and Version Pinning

## Status
Accepted

## Context
Fast-moving dependencies broke past monitoring setups when minor versions drifted:
- Pydantic v2 moved `BaseSettings` into `pydantic-settings`.
- SQLAlchemy 2.0 removed legacy `Query` methods in favor of `select()`.
- MLflow 3.x restructured client dependencies and deprecated file-backed tracking without explicit environment flags.
- Evidently 0.7 refactored presets and dropped legacy column mapping.

## Decision
Pin exact versions and test on Python 3.12:
- **API and Validation**: FastAPI `0.136.3`, Pydantic `2.13.4`, `pydantic-settings` `2.14.2`.
- **Database**: SQLAlchemy `2.0.51` with Alembic `1.18.5`. Local runs default to SQLite (`driftwatch.db`) for zero setup; Docker Compose uses PostgreSQL via `psycopg2-binary` 2.9.12.
- **Modeling**: XGBoost `3.2.0` (CPU mode, `hist` tree method for deterministic speed), scikit-learn `1.8.0`, SciPy `1.17.1`.
- **Drift calculation**: Native SciPy and NumPy implementations for PSI, KS (`scipy.stats.ks_2samp`), Jensen-Shannon, Wasserstein, and Chi-Square. Benjamini-Hochberg FDR control is coded directly to avoid extra dependencies.
- **Cross-check**: Evidently `0.7.23` for independent secondary validation and HTML export.
- **Registry**: MLflow `3.17.2` using SQLite backend (`sqlite:///artifacts/mlflow.db`).
- **Dashboard**: Streamlit `1.58.0` with Plotly `5.24.1`.

## Alternatives
- *Use Evidently as the primary engine*: Rejected. Evidently's overhead is too high for sub-millisecond per-feature evaluation during inference. Native SciPy runs in under 0.8 ms per 1,000 samples.
- *Default to PostgreSQL locally*: Rejected. Requiring a running Postgres instance complicates quick evaluation. SQLite runs tests in under 20 seconds.

## Consequences
- No hidden library incompatibilities.
- The entire demo and test suite run locally on CPU without external services.
