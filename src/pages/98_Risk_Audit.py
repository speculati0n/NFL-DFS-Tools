import _bootstrap_path  # ensure project root is on sys.path

import os
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk & Jitter Audit", layout="wide")

st.title("Risk Audit — Consistency / Upside / Duds")
st.caption("Confirms risk inputs and derived σ & skew used by the Optimizer/Simulator.")

st.markdown("---")

st.header("Risk Table")

def _find_existing(paths):
    return [p for p in paths if p and os.path.exists(p)]

def _latest(paths):
    xs = _find_existing(paths)
    return max(xs, key=lambda p: os.path.getmtime(p)) if xs else None

# Candidate default locations
CAND_RISK = [
    os.path.join("outputs", "risk_table_optimizer.csv"),
    os.path.join("outputs", "risk_table_simulator.csv"),
    os.path.join("risk_table_optimizer.csv"),
    os.path.join("risk_table_simulator.csv"),
]
CAND_JITTER = [
    os.path.join("outputs", "risk_jitter_optimizer.csv"),
    os.path.join("risk_jitter_optimizer.csv"),

]

with st.expander("Data sources", expanded=True):
    # Risk
    risk_auto = _latest(CAND_RISK)
    risk_choice = st.radio("Risk table", ["Auto-detect latest", "Pick file..."], horizontal=True, key="risk_src")
    risk_path = risk_auto if risk_choice == "Auto-detect latest" else st.text_input("Path to risk_table CSV", value=risk_auto or "", key="risk_path")

    # Jitter
    jitter_auto = _latest(CAND_JITTER)
    jitter_choice = st.radio("Jitter table", ["Auto-detect latest", "Pick file..."], horizontal=True, key="jitter_src")
    jitter_path = jitter_auto if jitter_choice == "Auto-detect latest" else st.text_input("Path to risk_jitter CSV", value=jitter_auto or "", key="jitter_path")

# ----- Risk table -----
if risk_path and os.path.exists(risk_path):
    rdf = pd.read_csv(risk_path)

    cols = st.columns([1, 1, 2, 2])
    with cols[0]:
        pos_filter = st.multiselect("Position", sorted([p for p in rdf["pos"].dropna().unique() if isinstance(p, str)]))
    with cols[1]:
        team_filter = st.multiselect("Team", sorted([t for t in rdf["team"].dropna().unique() if isinstance(t, str)]))
    with cols[2]:
        name_search = st.text_input("Name contains", "", key="risk_name")
    with cols[3]:
        sort_by = st.selectbox("Sort by", ["pos","team","name","proj","sigma_eff","r_plus","r_minus"], key="risk_sort")

    q = rdf.copy()
    if pos_filter: q = q[q["pos"].isin(pos_filter)]
    if team_filter: q = q[q["team"].isin(team_filter)]
    if name_search.strip():
        s = name_search.strip().lower()
        q = q[q["name"].str.lower().str.contains(s, na=False)]
    q = q.sort_values(sort_by, kind="stable").reset_index(drop=True)

    def _fmt_percent_auto(x):
        try:
            x = float(x)
        except Exception:
            return x
        if 0.0 <= x <= 1.0:
            return f"{x*100:.1f}%"
        return f"{x:.1f}"

    show_cols = ["name","pos","team","proj","floor","ceiling","consistency","upside","duds","sigma_base","sigma_eff","sigma_shrink_factor","r_plus","r_minus"]
    show_cols = [c for c in show_cols if c in q.columns]

    st.dataframe(
        q[show_cols].style.format({
            "proj": "{:.2f}",
            "floor": "{:.2f}",
            "ceiling": "{:.2f}",
            "sigma_base": "{:.3f}",
            "sigma_eff": "{:.3f}",
            "sigma_shrink_factor": "{:.3f}",
            "r_plus": "{:.3f}",
            "r_minus": "{:.3f}",
            "consistency": _fmt_percent_auto,
            "upside": _fmt_percent_auto,
            "duds": _fmt_percent_auto,
        }),
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "Download Risk CSV",
        data=q.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(risk_path).replace(".csv", "_filtered.csv"),
        mime="text/csv",
        key="dl_risk",
    )
else:
    st.info("No risk table found yet. Run the optimizer or simulator first.")

st.markdown("---")


if jitter_path and os.path.exists(jitter_path):
    jdf = pd.read_csv(jitter_path)

    cols2 = st.columns([1, 1, 2, 2])
    with cols2[0]:
        pos_filter_j = st.multiselect("Position", sorted([p for p in jdf["pos"].dropna().unique() if isinstance(p, str)]), key="jit_pos")
    with cols2[1]:
        team_filter_j = st.multiselect("Team", sorted([t for t in jdf["team"].dropna().unique() if isinstance(t, str)]), key="jit_team")
    with cols2[2]:
        name_search_j = st.text_input("Name contains", "", key="jit_name")
    with cols2[3]:
        sort_by_j = st.selectbox(
            "Sort by",
            ["pos","team","name","nudged_total","selection_rate","selected_when_up_rate","selected_when_down_rate","avg_nudge_pts","avg_abs_nudge_pts"],
            key="jit_sort"
        )

    qj = jdf.copy()
    if pos_filter_j: qj = qj[qj["pos"].isin(pos_filter_j)]
    if team_filter_j: qj = qj[qj["team"].isin(team_filter_j)]
    if name_search_j.strip():
        s = name_search_j.strip().lower()
        qj = qj[qj["name"].str.lower().str.contains(s, na=False)]
    qj = qj.sort_values(sort_by_j, kind="stable").reset_index(drop=True)

    show_cols_j = [
        "name","pos","team","proj",
        "sigma_base","sigma_eff","r_plus","r_minus",
        "nudged_up","nudged_down","nudged_total",
        "nudged_up_selected","nudged_up_not_selected",
        "nudged_down_selected","nudged_down_not_selected",
        "selected_total","selection_rate",
        "selected_when_up_rate","selected_when_down_rate",
        "avg_nudge_pts","avg_abs_nudge_pts",
        "avg_nudge_pts_selected","avg_abs_nudge_pts_selected",
    ]
    show_cols_j = [c for c in show_cols_j if c in qj.columns]

    st.dataframe(
        qj[show_cols_j].style.format({
            "proj": "{:.2f}",
            "sigma_base": "{:.3f}",
            "sigma_eff": "{:.3f}",
            "r_plus": "{:.3f}",
            "r_minus": "{:.3f}",
            "selection_rate": "{:.1%}",
            "selected_when_up_rate": "{:.1%}",
            "selected_when_down_rate": "{:.1%}",
            "avg_nudge_pts": "{:.4f}",
            "avg_abs_nudge_pts": "{:.4f}",
            "avg_nudge_pts_selected": "{:.4f}",
            "avg_abs_nudge_pts_selected": "{:.4f}",
        }),
        use_container_width=True,
        height=480,
    )

    st.download_button(
        "Download Jitter+Selection CSV",
        data=qj.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(jitter_path).replace(".csv", "_filtered.csv"),
        mime="text/csv",
        key="dl_jitter",
    )
else:


