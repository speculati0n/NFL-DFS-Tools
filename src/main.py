import os
import shutil
import pandas as pd
import streamlit as st

from nfl_optimizer import NFL_Optimizer
from nfl_showdown_optimizer import NFL_Showdown_Optimizer
from nfl_gpp_simulator import NFL_GPP_Simulator
from nfl_showdown_simulator import NFL_Showdown_Simulator

st.title("NFL DFS Tools")

# Sidebar for uploading files
st.sidebar.header("Upload Required Files")
site_upload = st.sidebar.text_input("Site (dk or fd)")
# Normalize the site input to avoid path issues
site_upload = site_upload.strip().lower()
projections_file = st.sidebar.file_uploader("Projections CSV", type="csv")
players_file = st.sidebar.file_uploader("Player IDs CSV", type="csv")
contest_file = st.sidebar.file_uploader(
    "Contest Structure CSV (optional)", type="csv"
)
config_file = st.sidebar.file_uploader(
    "Config JSON (optional)", type="json"
)

# Determine repository root so uploaded files are saved where the
# optimizer expects them (one level above ``src``).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")

if st.sidebar.button("Save Files"):
    if site_upload:
        data_dir = os.path.join(UPLOAD_DIR, site_upload)
        os.makedirs(data_dir, exist_ok=True)
        if projections_file:
            with open(os.path.join(data_dir, "projections.csv"), "wb") as f:
                f.write(projections_file.getbuffer())
        if players_file:
            with open(os.path.join(data_dir, "player_ids.csv"), "wb") as f:
                f.write(players_file.getbuffer())
        if contest_file and contest_file.name:
            with open(os.path.join(data_dir, "contest_structure.csv"), "wb") as f:
                f.write(contest_file.getbuffer())
        if config_file and config_file.name:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            with open(os.path.join(UPLOAD_DIR, "config.json"), "wb") as f:
                f.write(config_file.getbuffer())
        st.sidebar.success("Files saved.")
    else:
        st.sidebar.error("Please specify a site before saving.")

# Optimizer section
st.header("Optimize Lineups")
with st.form("optimize"):
    site_opt = st.text_input("Site", key="site_opt")
    site_opt = site_opt.strip().lower()
    num_lineups = st.number_input("Number of Lineups", min_value=1, value=1, step=1)
    num_uniques = st.number_input("Number of Uniques", min_value=1, value=1, step=1)
    mode_opt = st.selectbox("Mode", ["classic", "showdown"], key="mode_opt")
    save_lineups = st.checkbox("Save lineups for simulator")
    submitted_opt = st.form_submit_button("Run Optimizer")
    if submitted_opt:
        if mode_opt == "showdown":
            optimizer = NFL_Showdown_Optimizer(site_opt, num_lineups, num_uniques)
        else:
            optimizer = NFL_Optimizer(site_opt, num_lineups, num_uniques)
        optimizer.optimize()
        lineup_path, stack_path = optimizer.output()
        if save_lineups:
            dest_dir = os.path.join(UPLOAD_DIR, site_opt)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(lineup_path, os.path.join(dest_dir, "tournament_lineups.csv"))
        df = pd.read_csv(lineup_path)
        st.subheader("Lineups")
        st.dataframe(df)
        if stack_path:
            stack_df = pd.read_csv(stack_path)
            st.subheader("Stack Exposure")
            st.dataframe(stack_df)
        try:
            if hasattr(optimizer, "risk_table_df") and optimizer.risk_table_df is not None and not optimizer.risk_table_df.empty:
                st.subheader("Risk Audit (Optimizer)")
                st.dataframe(optimizer.risk_table_df, use_container_width=True, height=350)
            if hasattr(optimizer, "jitter_table_df") and optimizer.jitter_table_df is not None and not optimizer.jitter_table_df.empty:
                st.subheader("Jitter + Selection Audit (Optimizer)")
                st.dataframe(optimizer.jitter_table_df, use_container_width=True, height=350)
        except Exception:
            pass

# Simulation section
st.header("Simulate Tournament")
with st.form("simulate"):
    site_sim = st.text_input("Site", key="site_sim")
    site_sim = site_sim.strip().lower()
    field_size = st.number_input("Field Size", min_value=1, value=10, step=1)
    num_iterations = st.number_input("Iterations", min_value=1, value=10, step=1)
    mode_sim = st.selectbox("Mode", ["classic", "showdown"], key="mode_sim")
    use_contest_data = st.checkbox("Use contest structure")
    use_lineup_input = st.checkbox("Use saved lineups")
    submitted_sim = st.form_submit_button("Run Simulation")
    if submitted_sim:
        if mode_sim == "showdown":
            sim = NFL_Showdown_Simulator(
                site_sim, field_size, num_iterations, use_contest_data, use_lineup_input
            )
            sim.generate_field_lineups()
            sim.run_tournament_simulation()
            lineup_path, exposure_path = sim.save_results()
        else:
            sim = NFL_GPP_Simulator(
                site_sim, field_size, num_iterations, use_contest_data, use_lineup_input
            )
            sim.generate_field_lineups()
            sim.run_tournament_simulation()
            lineup_path, exposure_path, stack_path = sim.output()
        lineup_df = pd.read_csv(lineup_path, nrows=1000)
        exposure_df = pd.read_csv(exposure_path)
        st.subheader("Lineups (showing first 1,000 entries)")
        st.dataframe(lineup_df)
        if lineup_path and os.path.exists(lineup_path):
            with open(lineup_path, "rb") as f:
                lineup_csv = f.read()
            st.download_button(
                "Download full lineup CSV",
                lineup_csv,
                file_name=os.path.basename(lineup_path),
                mime="text/csv",
            )
        st.subheader("Exposure")
        st.dataframe(exposure_df)
        if stack_path:
            stack_df = pd.read_csv(stack_path)
            st.subheader("Stack Exposure")
            st.dataframe(stack_df)
