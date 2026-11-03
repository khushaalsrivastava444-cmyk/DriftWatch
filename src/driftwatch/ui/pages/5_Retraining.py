"""Page 5: Retraining Pipelines & Champion-Challenger Evaluations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftwatch.ui.db_helper import (
    get_db_session,
    query_model_evaluations,
    query_retraining_jobs,
)

st.set_page_config(page_title="Retraining — DriftWatch", page_icon="🔄", layout="wide")

st.title("🔄 Retraining Pipeline & Model Comparison")
st.markdown(
    "Orchestrates model retraining on validated data. Evaluates candidates using "
    "**walk-forward temporal validation** and paired statistical hypothesis testing (Wilcoxon / paired t-test)."
)

db = get_db_session()

jobs = query_retraining_jobs(db, limit=30)
evaluations = query_model_evaluations(db, limit=30)

c1, c2 = st.columns(2)
with c1:
    st.subheader("📋 Retraining Job History")
    if not jobs:
        st.info("No retraining jobs recorded yet. Triggered automatically on sustained verified degradation.")
    else:
        j_data = []
        for j in jobs:
            j_data.append({
                "Job ID": j.id,
                "Policy": j.trigger_policy,
                "Status": j.status,
                "Champion": j.champion_version_id,
                "Candidate": j.candidate_version_id or "Pending",
                "Created At": j.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            })
        st.dataframe(pd.DataFrame(j_data), use_container_width=True)

with c2:
    st.subheader("⚖️ Champion vs Challenger Statistical Evaluations")
    if not evaluations:
        st.info("No comparative evaluations found.")
    else:
        e_data = []
        for e in evaluations:
            e_data.append({
                "Evaluation ID": e.id,
                "Metric": e.metric_name,
                "Champion": f"{e.champion_metric:.4f}",
                "Candidate": f"{e.candidate_metric:.4f}",
                "Paired p-value": f"{e.p_value_paired:.4f}",
                "Stat Significant": "YES" if e.is_stat_significant else "NO",
                "Decision": e.decision,
            })
        st.dataframe(pd.DataFrame(e_data), use_container_width=True)

st.divider()

st.markdown("### 🛡️ Autonomous Governance Rules")
st.markdown(
    """
    - **Drift alone NEVER triggers retraining**: Input distribution shift ($P(X)$) does not imply performance decay.
    - **Verified degradation required**: Retraining triggers only when ground-truth error breaches the 95% bootstrap CI band across consecutive windows.
    - **Challenger evaluation**: A newly trained model is evaluated against the Champion on identical unseen temporal holdout test folds.
    - **Auditability**: All promotions require statistically significant improvement ($p < 0.05$) and are logged to `audit_logs`.
    """
)
