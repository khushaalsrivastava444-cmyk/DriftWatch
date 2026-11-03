"""Page 7: Data Quality & Schema Validation Manifest."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftwatch.validation.engine import DataValidationEngine, ValidationConfig

st.set_page_config(page_title="Data Quality - DriftWatch", layout="wide")

st.title("Data Quality")
st.markdown(
    "Pre-inference schema and range checks. Validates types, nulls, "
    "primary key uniqueness, and numeric bounds before data reaches the model."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 📋 Active Schema Rules")
    st.markdown(
        r"""
        - **Required Fields**: `store_id`, `item_id`, `date`, `sales`, `sell_price`
        - **Type Constraints**: `store_id` (str), `item_id` (str), `sales` (numeric $\ge 0$), `sell_price` (> 0)
        - **Null Tolerance**: 0% nulls allowed on operational primary keys
        - **Primary Key**: (`store_id`, `item_id`, `date`) uniqueness strictly checked
        """
    )
with c2:
    st.markdown("### 🚫 Outlier & Range Fencing")
    st.markdown(
        """
        - **Price Bounds**: Allowed range: `[0.01, 1000.0]`
        - **Sales Bounds**: Allowed range: `[0, 100,000]`
        - **IQR Fencing**: Tukey $k=3.0$ extreme outlier detection on rolling sales volumes
        - **Calendar Continuity**: Timestamps must be valid ISO-8601 UTC dates
        """
    )
with c3:
    st.markdown("### 📦 Quarantine Isolation")
    st.markdown(
        """
        - **Zero Data Leakage**: Quarantined rows are isolated from feature engineering and training
        - **Defensive Ingestion**: Contaminated records are quarantined into separate tables
        - **Severity Routing**:
          - `ERROR`: Quarantines offending rows
          - `WARNING`: Flags row but permits pipeline continuation
        """
    )

st.divider()

st.subheader("Validation Test")
st.markdown("Test the validation engine with sample rows:")

sample_input = {
    "store_id": ["CA_1", "CA_1", "CA_2"],
    "item_id": ["FOODS_1", "FOODS_1", "HOBBIES_1"],
    "date": ["2026-01-01", "2026-01-01", "2026-01-02"],  # Duplicate key in rows 1 and 2
    "sales": [10.0, -5.0, 25.0],  # Negative sales in row 2
    "sell_price": [2.50, 0.00, 12.00],  # Zero price in row 2
}

df_sample = pd.DataFrame(sample_input)
st.write("Sample Batch Ingestion:")
st.dataframe(df_sample, use_container_width=True)

if st.button("Run Ingestion Validation On Sample"):
    engine = DataValidationEngine(ValidationConfig())
    report, clean_df, quarantined_df = engine.validate(df_sample)

    st.markdown(f"**Passed Validation:** {'✅ True' if report.is_valid else '❌ False'}")
    st.markdown(f"**Clean Records Retained:** {len(clean_df)} / {len(df_sample)}")
    st.markdown(f"**Quarantined Records:** {len(quarantined_df)}")

    if report.has_errors:
        st.error("**Validation Errors Detected:**")
        for err in [v for v in report.violations if v.severity.value == "ERROR"]:
            st.markdown(f"- **{err.rule_name}** on column `{err.affected_column}`: {err.message}")

    if report.has_warnings:
        st.warning("**Validation Warnings Detected:**")
        for warn in [v for v in report.violations if v.severity.value == "WARNING"]:
            st.markdown(f"- **{warn.rule_name}** on column `{warn.affected_column}`: {warn.message}")

    if not quarantined_df.empty:
        st.markdown("#### Quarantined Partition")
        st.dataframe(quarantined_df, use_container_width=True)
