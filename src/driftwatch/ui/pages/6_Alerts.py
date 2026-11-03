"""Page 6: Alert Management & Incident Resolution."""

from __future__ import annotations

import json

import streamlit as st

from driftwatch.alerts.engine import AlertEngine
from driftwatch.ui.db_helper import (
    get_db_session,
    query_all_alerts,
    render_evidence_badge,
    render_severity_badge,
    render_status_badge,
)

st.set_page_config(page_title="Alerts — DriftWatch", page_icon="🚨", layout="wide")

st.title("🚨 Alert & Incident Management")
st.markdown(
    "Tiered alerts categorized by severity (`INFO`, `WARNING`, `CRITICAL`) "
    "and certified with an **evidence classification**."
)

db = get_db_session()

# Filter Bar
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    sel_severity = st.selectbox("Filter by Severity:", ["ALL", "INFO", "WARNING", "CRITICAL"])
with f_col2:
    sel_status = st.selectbox("Filter by Status:", ["ALL", "ACTIVE", "ACKNOWLEDGED", "RESOLVED"])
with f_col3:
    st.write("")  # spacing
    if st.button("🔄 Refresh Alerts"):
        st.rerun()

alerts = query_all_alerts(db, severity=sel_severity, status=sel_status)

if not alerts:
    st.info("No alerts match the selected criteria.")
    st.stop()

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Alerts Listed", len(alerts))
with c2:
    st.metric("Active", len([a for a in alerts if a.status == "ACTIVE"]))
with c3:
    st.metric("Critical", len([a for a in alerts if a.severity == "CRITICAL"]))
with c4:
    st.metric("Resolved", len([a for a in alerts if a.status == "RESOLVED"]))

st.divider()

engine = AlertEngine()

for a in alerts:
    sev_badge = render_severity_badge(a.severity)
    ev_badge = render_evidence_badge(a.evidence_class)
    status_badge = render_status_badge(a.status)

    with st.container():
        st.markdown(
            f"### {a.alert_type} &nbsp; {sev_badge} &nbsp; {status_badge} &nbsp; {ev_badge}",
            unsafe_allow_html=True,
        )
        st.caption(f"Alert ID: `{a.id}` | Triggered: {a.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.markdown(f"**Reason:** {a.reason}")
        st.markdown(f"**Recommended Action:** `{a.recommended_action}`")

        if a.affected_features_json:
            try:
                feats = json.loads(a.affected_features_json)
                st.markdown(f"**Affected Feature(s):** {', '.join([f'`{f}`' for f in feats])}")
            except Exception:
                pass

        if a.status != "RESOLVED":
            col_ack, col_res, _ = st.columns([1, 1, 4])
            with col_ack:
                if a.status == "ACTIVE":
                    if st.button("Acknowledge", key=f"ack_{a.id}"):
                        engine.acknowledge_alert(a.id, db)
                        st.success(f"Alert {a.id} acknowledged.")
                        st.rerun()
            with col_res:
                if st.button("Resolve", key=f"res_{a.id}"):
                    engine.resolve_alert(a.id, db)
                    st.success(f"Alert {a.id} resolved.")
                    st.rerun()
        else:
            res_str = a.resolved_at.strftime("%Y-%m-%d %H:%M UTC") if a.resolved_at else "N/A"
            st.caption(f"✅ Resolved at: {res_str}")

        st.markdown("---")
