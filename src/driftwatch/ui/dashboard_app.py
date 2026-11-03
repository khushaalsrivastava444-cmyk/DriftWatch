"""DriftWatch — Production ML Monitoring & Governance Dashboard.

Dark charcoal theme with blue/green accents. Displays real stored telemetry only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftwatch.alerts.engine import AlertEngine
from driftwatch.ui.db_helper import (
    get_db_session,
    query_active_alerts,
    query_system_overview,
    render_evidence_badge,
    render_severity_badge,
)

st.set_page_config(
    page_title="DriftWatch — MLOps Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Dark Charcoal & Neon Accents
st.markdown(
    """
    <style>
    .main {
        background-color: #121417;
    }
    .metric-card {
        background-color: #1a1d21;
        border: 1px solid #2a2e35;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #00d26a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90a4ae;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .evidence-box {
        background-color: #1a1d21;
        border-left: 4px solid #1e88e5;
        padding: 12px 16px;
        margin-top: 10px;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DriftWatch Overview")
st.markdown(
    "Monitoring platform for demand forecasting models. "
    "Every alert and degradation signal carries an explicit evidence class."
)

db = get_db_session()

# System Metrics Bar
overview = query_system_overview(db)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    champ_label = f"v{overview['champion_version']}" if overview['champion_version'] else "None"
    st.markdown(f'<div class="metric-label">Active Champion</div><div class="metric-value">{champ_label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Total Inferences</div><div class="metric-value">{overview["total_predictions"]:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    cov_str = f"{overview['label_coverage']:.1%}"
    st.markdown(f'<div class="metric-label">Label Coverage</div><div class="metric-value">{cov_str}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    mae_str = f"{overview['recent_mae']:.2f}" if overview['recent_mae'] is not None else "N/A"
    st.markdown(f'<div class="metric-label">Latest MAE</div><div class="metric-value">{mae_str}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    alert_color = "#e53935" if overview["critical_alerts"] > 0 else ("#ffa000" if overview["active_alerts"] > 0 else "#00d26a")
    st.markdown(
        f'<div class="metric-label">Active Alerts</div>'
        f'<div class="metric-value" style="color: {alert_color};">{overview["active_alerts"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Evidence Classification Reference Banner
st.markdown("### 🏷️ Evidence Classification Principles")
e_cols = st.columns(4)
with e_cols[0]:
    st.markdown(
        '<div class="evidence-box" style="border-left-color: #1e88e5;">'
        '<strong>STATISTICAL_SHIFT</strong><br>'
        '<small style="color: #90a4ae;">P(X) distribution change via PSI/KS. Unlabeled data only. Max severity: WARNING.</small>'
        '</div>',
        unsafe_allow_html=True,
    )
with e_cols[1]:
    st.markdown(
        '<div class="evidence-box" style="border-left-color: #8e24aa;">'
        '<strong>PREDICTION_SHIFT</strong><br>'
        '<small style="color: #90a4ae;">P(Ŷ) output drift. Unlabeled model predictions. Max severity: WARNING.</small>'
        '</div>',
        unsafe_allow_html=True,
    )
with e_cols[2]:
    st.markdown(
        '<div class="evidence-box" style="border-left-color: #f57c00;">'
        '<strong>ESTIMATED_PERFORMANCE</strong><br>'
        '<small style="color: #90a4ae;">Incomplete labels (<80% coverage). Status: UNVERIFIED. Max severity: INFO.</small>'
        '</div>',
        unsafe_allow_html=True,
    )
with e_cols[3]:
    st.markdown(
        '<div class="evidence-box" style="border-left-color: #00d26a;">'
        '<strong>VERIFIED_PERFORMANCE</strong><br>'
        '<small style="color: #90a4ae;">Delayed true labels matched with 95% bootstrap CI breach. Required for CRITICAL.</small>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# Active Alerts Panel
st.subheader("Active Alerts")
active_alerts = query_active_alerts(db)

if not active_alerts:
    st.info("No active alerts in the database.")
else:
    alert_rows = []
    for a in active_alerts:
        alert_rows.append({
            "ID": a.id,
            "Severity": a.severity,
            "Evidence Class": a.evidence_class,
            "Type": a.alert_type,
            "Reason": a.reason,
            "Recommended Action": a.recommended_action,
            "Triggered At": a.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if a.created_at else "N/A",
        })

    df_alerts = pd.DataFrame(alert_rows)
    for _idx, row in df_alerts.iterrows():
        sev_html = render_severity_badge(row["Severity"])
        ev_html = render_evidence_badge(row["Evidence Class"])
        with st.expander(f"{row['Severity']}: {row['Type']} — {row['ID']} ({row['Triggered At']})", expanded=True):
            st.markdown(f"**Severity:** {sev_html} &nbsp;&nbsp; **Evidence Class:** {ev_html}", unsafe_allow_html=True)
            st.markdown(f"**Reason:** {row['Reason']}")
            st.markdown(f"**Recommended Action:** `{row['Recommended Action']}`")
            col_ack, col_res = st.columns([1, 1])
            with col_ack:
                if st.button("Acknowledge Alert", key=f"ack_{row['ID']}"):
                    engine = AlertEngine()
                    engine.acknowledge_alert(row["ID"], db)
                    st.rerun()
            with col_res:
                if st.button("Resolve Alert", key=f"res_{row['ID']}"):
                    engine = AlertEngine()
                    engine.resolve_alert(row["ID"], db)
                    st.rerun()

st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    **Pages:**
    1. **Overview**
    2. **Data Drift**
    3. **Performance**
    4. **Model Registry**
    5. **Retraining**
    6. **Alerts**
    7. **Data Quality**
    """
)
st.sidebar.caption("Connected to database.")
