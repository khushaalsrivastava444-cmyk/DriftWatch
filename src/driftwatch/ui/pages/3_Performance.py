"""Page 3: Model Performance & Delayed Ground Truth Analytics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driftwatch.ui.db_helper import (
    get_db_session,
    query_performance_reports,
)

st.set_page_config(page_title="Performance — DriftWatch", page_icon="📉", layout="wide")

st.title("📉 Ground-Truth Performance Monitoring")
st.markdown(
    "Monitors forecasting error on delayed ground-truth retail demand. "
    "Calculates **stationary block bootstrap 95% confidence intervals**. "
    "Windows with <80% label maturity remain strictly **UNVERIFIED**."
)

db = get_db_session()
reports = query_performance_reports(db, limit=50)

if not reports:
    st.warning("⚠️ No performance reports recorded in database. Ground-truth labels must be matched and evaluated.")
    st.stop()

# Convert to DataFrame
rows = []
for r in reports:
    rows.append({
        "Window ID": r.id,
        "Start": r.window_start,
        "End": r.window_end,
        "Total Predictions": r.sample_size,
        "Labeled": r.labeled_count,
        "Coverage": r.label_coverage_ratio,
        "Status": r.status,
        "MAE": r.mae,
        "RMSE": r.rmse,
        "WAPE": r.wape,
        "MASE": r.mase,
        "Bias": r.bias,
        "CI Lower": r.ci_lower,
        "CI Upper": r.ci_upper,
        "Degradation Breached": r.degradation_breached,
    })

df = pd.DataFrame(rows).sort_values("Start", ascending=True)

# Top KPI metrics from latest window
latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Latest Window Status", latest["Status"])
with c2:
    st.metric("Label Coverage", f"{latest['Coverage']:.1%}")
with c3:
    mae_val = f"{latest['MAE']:.2f}" if pd.notnull(latest['MAE']) else "UNVERIFIED"
    st.metric("Point MAE", mae_val)
with c4:
    ci_val = f"[{latest['CI Lower']:.2f}, {latest['CI Upper']:.2f}]" if pd.notnull(latest['CI Lower']) else "N/A"
    st.metric("95% Bootstrap CI", ci_val)

if latest["Status"] == "UNVERIFIED":
    st.warning(
        "⏳ **Status UNVERIFIED**: Delayed ground-truth coverage is below the 80% threshold. "
        "Performance degradation cannot be verified until retail delivery confirmations arrive."
    )

st.divider()

# Plotly Rolling MAE with 95% Bootstrap Confidence Interval
st.subheader("📊 Temporal Error Tracking with 95% Bootstrap Bands")

df_verified = df.dropna(subset=["MAE", "CI Lower", "CI Upper"])

if not df_verified.empty:
    fig = go.Figure()

    # CI upper band
    fig.add_trace(
        go.Scatter(
            x=df_verified["Start"],
            y=df_verified["CI Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # CI lower band + shaded fill
    fig.add_trace(
        go.Scatter(
            x=df_verified["Start"],
            y=df_verified["CI Lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 210, 106, 0.15)",
            name="95% Bootstrap CI",
        )
    )

    # Point estimate line
    fig.add_trace(
        go.Scatter(
            x=df_verified["Start"],
            y=df_verified["MAE"],
            mode="lines+markers",
            line=dict(color="#00d26a", width=2.5),
            marker=dict(size=6),
            name="Point MAE",
        )
    )

    # Breach points highlight
    df_breach = df_verified[df_verified["Degradation Breached"] == True]
    if not df_breach.empty:
        fig.add_trace(
            go.Scatter(
                x=df_breach["Start"],
                y=df_breach["MAE"],
                mode="markers",
                marker=dict(color="#e53935", size=10, symbol="x"),
                name="Degradation Breach",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1d21",
        paper_bgcolor="#121417",
        font_color="#f0f2f5",
        xaxis_title="Monitoring Window Start",
        yaxis_title="Mean Absolute Error (Units)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# Performance Reports Table
st.subheader("📋 Chronological Window Performance Manifest")
st.dataframe(df, use_container_width=True)
