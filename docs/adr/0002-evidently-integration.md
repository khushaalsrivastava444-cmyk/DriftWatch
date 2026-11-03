# ADR 0002: Evidently 0.7+ Integration Patterns

## Status
Accepted

## Context
Evidently released version 0.7.x which refactored import hierarchies:
- `ColumnMapping` and legacy metrics presets were moved into `evidently.legacy` or simplified in the root namespace `from evidently import Report`.
- Metric presets are located under `from evidently.presets import DataDriftPreset`.
- Report execution operates on raw pandas DataFrames: `report.run(reference_data=..., current_data=...)`.
- HTML export is supported via `report.save_html(path)`.

## Decision
1. Import `Report` directly from `evidently`: `from evidently import Report`.
2. Import `DataDriftPreset` from `evidently.presets`: `from evidently.presets import DataDriftPreset`.
3. Use Evidently strictly for independent cross-check validation and static HTML exports, preserving pure SciPy/NumPy implementations for DriftWatch's internal sub-millisecond production inference path.

## Consequences
- Clean execution without legacy deprecation warnings.
- Full compatibility with Evidently 0.7.23+.
