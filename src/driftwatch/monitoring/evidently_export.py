"""Evidently 0.7+ Cross-Check Reporting and HTML Export Utility."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)


def generate_evidently_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    output_html_path: str = "reports/evidently_drift_report.html",
) -> dict[str, Any]:
    """Generate standalone Evidently 0.7+ Data Drift Report and export to HTML."""
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter only available numeric features
    cols = [c for c in features if c in reference_df.columns and c in current_df.columns]
    ref_sub = reference_df[cols].dropna()
    curr_sub = current_df[cols].dropna()

    logger.info("Generating Evidently 0.7+ report across %d features...", len(cols))
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_sub, current_data=curr_sub)

    # Save HTML output
    snapshot.save_html(str(out_path))
    logger.info("Evidently HTML report exported to %s", out_path)

    summary = {
        "html_path": str(out_path),
        "total_features": len(cols),
        "status": "success",
    }
    return summary
