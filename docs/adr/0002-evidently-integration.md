# ADR 0002: Evidently 0.7+ Integration Patterns

## Status
Accepted

## Context
Evidently 0.7 refactored its package structure:
- Root presets now live in `evidently.presets` (e.g. `DataDriftPreset`).
- Reports run directly on pandas DataFrames: `report.run(reference_data=ref, current_data=curr)`.
- Legacy `ColumnMapping` is replaced with standard column names.
- HTML export works via `report.save_html(path)`.

## Decision
1. Import `Report` directly from `evidently`, and `DataDriftPreset` from `evidently.presets`.
2. Keep Evidently as an optional cross-check and HTML report generator. The core monitoring loops use native SciPy and NumPy functions.

## Alternatives
- *Replace native drift engine with Evidently entirely*: Rejected. Evidently runs full suites of tests that take hundreds of milliseconds per batch. Native tests complete in under 1 ms per feature.
- *Drop Evidently completely*: Rejected. Having an independent tool verify our drift decisions provides an objective sanity check on boundary cases.

## Consequences
- Clean imports without deprecation warnings.
- Fast serving path remains decoupled from Evidently's dependency footprint.
