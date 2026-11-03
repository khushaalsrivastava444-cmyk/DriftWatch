"""Page 4: Model Registry & Champion-Challenger Lineage."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from driftwatch.api.service import serving_service
from driftwatch.ui.db_helper import get_db_session, query_model_registry

st.set_page_config(page_title="Model Registry — DriftWatch", page_icon="📦", layout="wide")

st.title("📦 Model Registry & Governance")
st.markdown(
    "Audited repository of demand forecasting model versions. "
    "Enforces explicit separation between candidate registration and active champion promotion."
)

db = get_db_session()
registry = query_model_registry(db)

if not registry:
    st.warning("⚠️ No models registered in database. Initialize or train a forecaster first.")
    st.stop()

for entry in registry:
    model = entry["model"]
    versions = entry["versions"]

    st.subheader(f"Model: `{model.name}` ({model.id})")
    st.caption(f"Task: {model.task_type} | Created: {model.created_at.strftime('%Y-%m-%d %H:%M UTC')}")

    if not versions:
        st.info("No versions registered for this model.")
        continue

    v_data = []
    active_version = None
    for v in versions:
        if v.is_active_champion:
            active_version = v

        metrics = json.loads(v.metrics_json) if v.metrics_json else {}
        v_data.append({
            "Version": f"v{v.version}",
            "Status": "ACTIVE CHAMPION" if v.is_active_champion else "CANDIDATE",
            "Artifact URI": v.artifact_uri,
            "Val MAE": f"{metrics.get('mae', 'N/A')}",
            "Val RMSE": f"{metrics.get('rmse', 'N/A')}",
            "Val WAPE": f"{metrics.get('wape', 'N/A')}",
            "Val MASE": f"{metrics.get('mase', 'N/A')}",
            "Registered At": v.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        })

    st.dataframe(pd.DataFrame(v_data), use_container_width=True)

    # Champion Card & Actions
    if active_version:
        st.success(f"🏆 Current Active Champion: **v{active_version.version}** (ID: `{active_version.id}`)")
        with st.expander("Inspect Champion Hyperparameters & Metrics"):
            p_col, m_col = st.columns(2)
            with p_col:
                st.markdown("**Parameters:**")
                st.json(json.loads(active_version.params_json) if active_version.params_json else {})
            with m_col:
                st.markdown("**Validation Metrics:**")
                st.json(json.loads(active_version.metrics_json) if active_version.metrics_json else {})

    # Actions: Promote Challenger or Rollback
    st.markdown("#### Governance Actions")
    act_col1, act_col2 = st.columns(2)

    with act_col1:
        other_versions = [v for v in versions if not v.is_active_champion]
        if other_versions:
            target_v = st.selectbox(
                "Select Candidate to Promote:",
                options=[v.version for v in other_versions],
                format_func=lambda x: f"Candidate v{x}",
                key=f"promote_sel_{model.id}",
            )
            if st.button("🚀 Promote to Champion", key=f"btn_promote_{model.id}"):
                try:
                    res = serving_service.promote_model_version(
                        model_id=model.id,
                        version=target_v,
                        actor="streamlit_admin",
                        db=db,
                    )
                    st.success(f"Successfully promoted v{target_v} to active champion!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Promotion failed: {e}")
        else:
            st.info("No alternate candidate versions available for promotion.")

    with act_col2:
        if st.button("⏮️ Rollback to Previous Champion", key=f"btn_rollback_{model.id}"):
            try:
                res = serving_service.rollback_champion(
                    model_id=model.id,
                    actor="streamlit_admin",
                    db=db,
                )
                st.success(f"Rollback successful: Active champion is now v{res.version}")
                st.rerun()
            except Exception as e:
                st.error(f"Rollback rejected: {e}")

    st.divider()
