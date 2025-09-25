import os
import json
import csv
import re


import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify, send_file, abort

# Existing repo classes
from src.nfl_optimizer import NFL_Optimizer
from src.nfl_showdown_optimizer import NFL_Showdown_Optimizer
from src.nfl_gpp_simulator import NFL_GPP_Simulator
from src.nfl_showdown_simulator import NFL_Showdown_Simulator

# Diversity engine (exists in repo)
from src.anti_cannibalizer import DiversityRules, Candidate, diversify_portfolio

# Utility path resolver used by the simulator
from src.utils import get_data_path  # only for reference; we write to uploads/{site}/tournament_lineups.csv

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    site = request.form["site"].strip().lower()
    site_dir = os.path.join(UPLOAD_DIR, site)
    os.makedirs(site_dir, exist_ok=True)
    # Persist uploads (optional projections/players/contest)
    for field in ["projections", "players", "contest"]:
        f = request.files.get(field)
        if f and f.filename:
            f.save(os.path.join(site_dir, f.filename))
    return redirect("/")

@app.route("/optimize", methods=["POST"])
def optimize():
    _reset_progress()
    site = request.form["site"].strip().lower()
    num_lineups = request.form["num_lineups"]
    num_uniques = request.form["num_uniques"]
    mode = request.form.get("mode", "classic")
    save_lineups = "save_lineups" in request.form

    apply_diversity = "apply_diversity" in request.form
    # Diversity guard fields (safe defaults)
    max_shared_players = int(request.form.get("max_shared_players", 6) or 6)
    min_jaccard_distance = float(request.form.get("min_jaccard_distance", 0.20) or 0.20)
    _pp = request.form.get("per_player_cap", "").strip()
    _pt = request.form.get("per_team_cap", "").strip()
    per_player_cap = float(_pp) if _pp else None
    per_team_cap = float(_pt) if _pt else None
    stack_mix_text = request.form.get("stack_mix", "").strip()
    min_stack_mix = None
    if stack_mix_text:
        tmp = {}
        for kv in stack_mix_text.split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                try:
                    tmp[k.strip()] = float(v.strip())
                except Exception:
                    pass
        min_stack_mix = tmp or None

    def _run():
        progress_data["status"] = "running"
        try:
            if mode == "showdown":
                opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
                total = opto.num_lineups
            else:
                opto = NFL_Optimizer(site, num_lineups, num_uniques)
                total = max(int(opto.num_lineups * getattr(opto, "pool_factor", 1)), opto.num_lineups)

            progress_data["total"] = total

@app.route("/simulate", methods=["POST"])
def simulate():
    _reset_progress()
    site = request.form["site"].strip().lower()
    field_size = request.form["field_size"]
    num_iterations = request.form["num_iterations"]
    mode = request.form.get("mode", "classic")
    use_contest_data = "use_contest_data" in request.form
    use_lineup_input = "use_lineup_input" in request.form

    apply_diversity_sim = "apply_diversity_sim" in request.form
    max_shared_players = int(request.form.get("max_shared_players", 6) or 6)
    min_jaccard_distance = float(request.form.get("min_jaccard_distance", 0.20) or 0.20)
    _pp = request.form.get("per_player_cap", "").strip()
    _pt = request.form.get("per_team_cap", "").strip()
    per_player_cap = float(_pp) if _pp else None
    per_team_cap = float(_pt) if _pt else None
    stack_mix_text = request.form.get("stack_mix", "").strip()
    min_stack_mix = None
    if stack_mix_text:
        tmp = {}
        for kv in stack_mix_text.split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                try:
                    tmp[k.strip()] = float(v.strip())
                except Exception:
                    pass
        min_stack_mix = tmp or None

@app.route("/progress")
def progress():
    # Return JSON used by progress.html polling
    # percent can be derived if current/total updated by optimizer/sim; we keep it simple here.
    return jsonify(progress_data)

@app.route("/results")
def results():
    if not any([progress_data.get("output_path"), progress_data.get("sim_lineup_path")]):
        return redirect("/")

    tables = []
    title = "Results"
    lineup_url = None
    stack_url = None
    diverse_url = None
    sim_diverse_input_url = None
    diverse_audit = None
    sim_diversity_audit = None

    def _normalize_indices(values):
        normalized = set()
        if not values:
            return normalized
        for raw in values:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                try:
                    idx = int(float(raw))
                except (TypeError, ValueError):
                    continue
            # taken_indices are zero-based; display is one-based
            normalized.add(idx + 1)
        return normalized

    def _coerce_lineup_index(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and pd.isna(value):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            if token.isdigit():
                return int(token)
            match = re.search(r"(\d+)", token)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    return None
            try:
                return int(float(token))
            except (TypeError, ValueError):
                return None
        return None

    def _annotate_diversified(df, diversified_indices=None, default_is_diversified=False):
        diversified_indices = diversified_indices or set()
        annotated = df.copy()
        column_name = "Diversified"

        if column_name in annotated.columns:
            annotated = annotated.drop(columns=[column_name])

        if "Lineup" in annotated.columns:
            flags = []
            for value in annotated["Lineup"]:
                idx = _coerce_lineup_index(value)
                is_diverse = default_is_diversified
                if idx is not None:
                    is_diverse = (idx in diversified_indices) or default_is_diversified
                flags.append(is_diverse)
            insert_at = annotated.columns.get_loc("Lineup") + 1
        else:
            flags = [default_is_diversified] * len(annotated.index)
            insert_at = 0

        annotated.insert(insert_at, column_name, flags)
        return annotated

    optimized_diverse_indices = set()
    if progress_data.get("diversity_audit_path") and os.path.exists(progress_data["diversity_audit_path"]):
        with open(progress_data["diversity_audit_path"], "r", encoding="utf-8") as f:
            diverse_audit = json.load(f)
        optimized_diverse_indices = _normalize_indices(diverse_audit.get("taken_indices"))

    # Optimizer lineups (first 1000 rows)
    if progress_data.get("output_path") and os.path.exists(progress_data["output_path"]):
        df = pd.read_csv(progress_data["output_path"], nrows=1000)
        df = _annotate_diversified(df, optimized_diverse_indices, default_is_diversified=False)
        tables.append(("Lineups (first 1000)", df.to_html(index=False)))
        lineup_url = "/download/lineups"

    # Optimizer stack exposure
    if progress_data.get("stack_path") and os.path.exists(progress_data["stack_path"]):
        stack_df = pd.read_csv(progress_data["stack_path"])
        tables.append(("Stack Exposure", stack_df.to_html(index=False)))
        stack_url = "/download/stacks"


    # Optimizer diversified portfolio & audit
    if progress_data.get("diverse_path") and os.path.exists(progress_data["diverse_path"]):
        df2 = pd.read_csv(progress_data["diverse_path"], nrows=1000)
        df2 = _annotate_diversified(df2, optimized_diverse_indices, default_is_diversified=True)
        tables.append(("Diversified Lineups (first 1000)", df2.to_html(index=False)))
        diverse_url = "/download/diverse_lineups"

    sim_diverse_indices = set()
    if progress_data.get("sim_diversity_audit_path") and os.path.exists(progress_data["sim_diversity_audit_path"]):
        with open(progress_data["sim_diversity_audit_path"], "r", encoding="utf-8") as f:
            sim_diversity_audit = json.load(f)
        sim_diverse_indices = _normalize_indices(sim_diversity_audit.get("taken_indices"))

    # Simulator diversified inputs & audit
    if progress_data.get("sim_diverse_input_path") and os.path.exists(progress_data["sim_diverse_input_path"]):
        df3 = pd.read_csv(progress_data["sim_diverse_input_path"], nrows=1000)
        df3 = _annotate_diversified(
            df3,
            sim_diverse_indices,
            default_is_diversified=not bool(sim_diverse_indices),
        )
        tables.append(("Simulator Diversified Inputs (first 1000)", df3.to_html(index=False)))
        sim_diverse_input_url = "/download/sim_diverse_inputs"

    return render_template(
        "results.html",
        title=title,
        tables=tables,
        lineup_url=lineup_url,
        stack_url=stack_url,
        diverse_url=diverse_url,
        sim_diverse_input_url=sim_diverse_input_url,
        diverse_audit=diverse_audit,
        sim_diversity_audit=sim_diversity_audit,
    )

@app.route("/download/<file_type>")
def download(file_type: str):
    mapping = {
        "lineups": os.path.join(OUTPUT_DIR, "optimized_lineups.csv"),
        "stacks": os.path.join(OUTPUT_DIR, "stack_exposure.csv"),
        "diverse_lineups": os.path.join(OUTPUT_DIR, "optimized_lineups_diverse.csv"),
        "sim_diverse_inputs": os.path.join(OUTPUT_DIR, "simulator_diverse_lineups.csv"),
    }
    path = mapping.get(file_type)
    if not path or not os.path.exists(path):
        return abort(404)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
