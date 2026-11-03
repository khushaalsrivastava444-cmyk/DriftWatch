# ADR 0001: Technology Stack and Version Pinning

## Status
Accepted

## Context
DriftWatch monitors deployed ML models in production against statistical drift and performance degradation. Fast-moving libraries (FastAPI, Pydantic, SQLAlchemy, MLflow, Evidently, XGBoost) frequently make breaking changes across minor and major releases:
- Pydantic v2 changed `BaseSettings` location to `pydantic-settings`, validation decorators, and JSON schema hooks.
- SQLAlchemy 2.0 eliminated legacy query APIs in favor of typed `select()` and explicit transaction contexts.
- MLflow 3.x restructured tracking and registry endpoints and skinny client dependencies.
- Evidently 0.7+ redesigned report APIs, metrics definitions, and UI test suites.

## Decisions
1. **Python Runtime**: Python 3.12.10.
2. **API Framework**: FastAPI `0.136.3` with Pydantic `2.13.4` and `pydantic-settings` `2.14.2`. Strict Pydantic v2 schemas.
3. **Database Layer**: SQLAlchemy `2.0.51` + Alembic `1.18.5`. Use 2.0-style statements (`select(...)`, session context managers). Default to SQLite file (`driftwatch.db`) for zero-configuration, deterministic local runs and unit testing; PostgreSQL driver (`psycopg2-binary` 2.9.12) for Docker Compose.
4. **Machine Learning & Forecasting**: XGBoost `3.2.0` (CPU mode, `hist` tree method for deterministic and fast training) alongside scikit-learn `1.8.0` and SciPy `1.17.1`.
5. **Drift & Statistics**: Direct, zero-magic SciPy and NumPy implementations of PSI, KS 2-sample (`scipy.stats.ks_2samp`), Jensen-Shannon divergence (`scipy.spatial.distance.jensenshannon`), Wasserstein distance (`scipy.stats.wasserstein_distance`), and Chi-Square contingency (`scipy.stats.chi2_contingency`). Benjamini-Hochberg FDR control implemented mathematically to avoid black-box library coupling.
6. **Cross-Check Monitoring**: Evidently `0.7.23` used specifically as an independent cross-check validator and HTML report exporter.
7. **Tracking & Registry**: MLflow `3.17.2` configured with local file/sqlite backend (`./mlruns` or `sqlite:///mlflow.db`).
8. **UI**: Streamlit `1.58.0` + Plotly `5.24.1` with a dark charcoal theme.

## Consequences
- Guaranteed API stability and reproducible test results across development and CI.
- Fast, local, CPU-only execution under 10 minutes without external service dependencies.
