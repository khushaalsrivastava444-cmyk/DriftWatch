"""Page 2: Statistical Data Drift Analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from driftwatch.ui.db_helper import (
    get_db_session,
    query_drift_reports,
    query_feature_drift_results,
)

st.set_page_config(page_title="Data Drift - DriftWatch", layout="wide")

st.title("Data Drift")
st.markdown(
    "Monitors distribution shifts across input features and model predictions. "
    "Uses Benjamini-Hochberg FDR control and dual-gating rules (significance + effect size)."
)

db = get_db_session()
reports = query_drift_reports(db, limit=20)

if not reports:
    st.info("No drift reports in the database yet. Run the simulation or benchmark to generate telemetry.")
    st.stop()

# Select Report Window
report_options = {
    f"{r.window_id} ({r.timestamp.strftime('%Y-%m-%d %H:%M UTC')}) - {r.drifted_features_count} drifted": r.id
    for r in reports
}
selected_label = st.selectbox("Select Monitoring Window Report:", list(report_options.keys()))
selected_report_id = report_options[selected_label]

feature_results = query_feature_drift_results(db, report_id=selected_report_id)

if not feature_results:
    st.info("No feature-level test results recorded for this report.")
    st.stop()

# Build DataFrame
data = []
for res in feature_results:
    data.append({
        "Feature": res.feature,
        "Test": res.test,
        "Statistic": res.statistic,
        "Effect Size": res.effect_size,
        "Threshold": res.threshold,
        "p-value": f"{res.p_value:.4f}" if res.p_value is not None else "N/A",
        "Adj p-value (BH)": f"{res.adjusted_p_value:.4f}" if res.adjusted_p_value is not None else "N/A",
        "Decision": "DRIFT" if res.decision else "STABLE",
        "Ref Samples": res.sample_size_ref,
        "Curr Samples": res.sample_size_curr,
        "Evidence Class": res.evidence_class,
    })

df = pd.DataFrame(data)

# Summary Cards
c1, c2, c3 = st.columns(3)
total_tests = len(df)
drifted_tests = len(df[df["Decision"] == "DRIFT"])
drifted_features_unique = df[df["Decision"] == "DRIFT"]["Feature"].nunique()

with c1:
    st.metric("Total Tests Evaluated", total_tests)
with c2:
    st.metric("Drifted Tests", drifted_tests)
with c3:
    st.metric("Unique Features Shifted", drifted_features_unique)

st.divider()

# Plotly Visualizations: Bar chart of Statistics
st.subheader("Test Statistics vs Thresholds")

test_types = df["Test"].unique().tolist()
selected_test = st.radio("Select Statistical Test to Plot:", test_types, horizontal=True)

df_test = df[df["Test"] == selected_test].sort_values("Statistic", ascending=False)

if not df_test.empty:
    fig = px.bar(
        df_test,
        x="Feature",
        y="Statistic",
        color="Decision",
        color_discrete_map={"DRIFT": "#e53935", "STABLE": "#00d26a"},
        title=f"{selected_test} Test Statistic per Feature",
        labels={"Statistic": f"{selected_test} Statistic", "Feature": "Feature Name"},
    )
    # Add horizontal threshold line
    threshold_val = df_test["Threshold"].iloc[0]
    fig.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="#ffa000",
        annotation_text=f"Dual-Gating Threshold ({threshold_val:.2f})",
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1d21",
        paper_bgcolor="#121417",
        font_color="#f0f2f5",
    )
    st.plotly_chart(fig, use_container_width=True)

# Full Results Table
st.subheader("Test Results")
st.dataframe(df, use_container_width=True)
