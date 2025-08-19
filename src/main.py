import os
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
projections_file = st.sidebar.file_uploader("Projections CSV", type="csv")
players_file = st.sidebar.file_uploader("Player IDs CSV", type="csv")
contest_file = st.sidebar.file_uploader(
    "Contest Structure CSV (optional)", type="csv"
)
config_file = st.sidebar.file_uploader(
    "Config JSON (optional)", type="json"
)

if st.sidebar.button("Save Files"):
    if site_upload:
        data_dir = f"{site_upload}_data"
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
            with open("config.json", "wb") as f:
                f.write(config_file.getbuffer())
        st.sidebar.success("Files saved.")
    else:
        st.sidebar.error("Please specify a site before saving.")

# Optimizer section
st.header("Optimize Lineups")
with st.form("optimize"):
    site_opt = st.text_input("Site", key="site_opt")
    num_lineups = st.number_input("Number of Lineups", min_value=1, value=1, step=1)
    num_uniques = st.number_input("Number of Uniques", min_value=1, value=1, step=1)
    mode_opt = st.selectbox("Mode", ["classic", "showdown"], key="mode_opt")
    submitted_opt = st.form_submit_button("Run Optimizer")
    if submitted_opt:
        if mode_opt == "showdown":
            opto = NFL_Showdown_Optimizer(site_opt, num_lineups, num_uniques)
        else:
            opto = NFL_Optimizer(site_opt, num_lineups, num_uniques)
        opto.optimize()
        output_path = opto.output()
        df = pd.read_csv(output_path)
        st.subheader("Lineups")
        st.dataframe(df)

# Simulation section
st.header("Simulate Tournament")
with st.form("simulate"):
    site_sim = st.text_input("Site", key="site_sim")
    field_size = st.number_input("Field Size", min_value=1, value=10, step=1)
    num_iterations = st.number_input("Iterations", min_value=1, value=10, step=1)
    mode_sim = st.selectbox("Mode", ["classic", "showdown"], key="mode_sim")
    submitted_sim = st.form_submit_button("Run Simulation")
    if submitted_sim:
        if mode_sim == "showdown":
            sim = NFL_Showdown_Simulator(site_sim, field_size, num_iterations, False, False)
            sim.generate_field_lineups()
            sim.run_tournament_simulation()
            lineup_path, exposure_path = sim.save_results()
        else:
            sim = NFL_GPP_Simulator(site_sim, field_size, num_iterations, False, False)
            sim.generate_field_lineups()
            sim.run_tournament_simulation()
            lineup_path, exposure_path = sim.output()
        lineup_df = pd.read_csv(lineup_path)
        exposure_df = pd.read_csv(exposure_path)
        st.subheader("Lineups")
        st.dataframe(lineup_df)
        st.subheader("Exposure")
        st.dataframe(exposure_df)
