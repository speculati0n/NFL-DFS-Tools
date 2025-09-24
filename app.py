import os
import json
import csv
import re
import threading
from collections import Counter
from typing import Dict, List, Optional, Sequence

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

# progress state for both optimizer and simulator flows
progress_data: Dict[str, Optional[str]] = {
    "current": 0,
    "total": 0,
    "percent": 0,
    "status": "idle",
    "output_path": None,
    "stack_path": None,
    "diverse_path": None,
    "diversity_audit_path": None,
    "sim_lineup_path": None,
    "sim_diverse_input_path": None,
    "sim_diversity_audit_path": None,
    "player_exposure_path": None,
    "player_exposure_diverse_path": None,
    "sim_diverse_lineup_signatures": None,
}

def _reset_progress():
    progress_data.update({
        "current": 0, "total": 0, "percent": 0, "status": "idle",
        "output_path": None, "stack_path": None,
        "diverse_path": None, "diversity_audit_path": None,
        "sim_lineup_path": None, "sim_diverse_input_path": None, "sim_diversity_audit_path": None,
        "player_exposure_path": None, "player_exposure_diverse_path": None,
        "sim_diverse_lineup_signatures": None,
    })


def _normalize_player_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _lineup_signature(players: Sequence[str]) -> str:
    normalized = [_normalize_player_name(p) for p in players if _normalize_player_name(p)]
    if not normalized:
        return ""
    return "|".join(sorted(normalized))


def _player_columns_from_df(df: pd.DataFrame) -> List[str]:
    player_cols: List[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if lower == "lineup":
            continue
        if any(keyword in lower for keyword in [
            "fpts",
            "win",
            "top",
            "stack",
            "salary",
            "roi",
            "own",
            "players vs",
            "lineup type",
            "sim dupes",
            "field",
            "avg",
        ]):
            break
        player_cols.append(col)
    if player_cols:
        return player_cols
    digit_cols = [col for col in df.columns if str(col).isdigit()]
    if digit_cols:
        return digit_cols
    return [col for col in df.columns if df[col].dtype == object and str(col).lower() != "lineup"]


def _write_player_exposure_csv(source_csv: str, dest_csv: str) -> Optional[str]:
    if not source_csv or not os.path.exists(source_csv):
        return None
    df = pd.read_csv(source_csv)
    if df.empty:
        exposure_df = pd.DataFrame(columns=["Player", "Lineups", "Exposure%"])
    else:
        player_cols = _player_columns_from_df(df)
        if not player_cols:
            return None
        counts: Counter[str] = Counter()
        for _, row in df[player_cols].iterrows():
            for val in row.values:
                if isinstance(val, str):
                    name = val.strip()
                    if not name:
                        continue
                    counts[name] += 1
        total = max(len(df.index), 1)
        rows = [
            {"Player": player, "Lineups": count, "Exposure%": round((count / total) * 100, 2)}
            for player, count in counts.most_common()
        ]
        exposure_df = pd.DataFrame(rows)
    exposure_df.to_csv(dest_csv, index=False)
    return dest_csv


def _annotate_diversified_column(df: pd.DataFrame, signatures: Optional[List[str]]) -> pd.DataFrame:
    sig_set = {s for s in (signatures or []) if s}
    player_cols = _player_columns_from_df(df)
    if "Diversified Lineup" in df.columns:
        df = df.drop(columns=["Diversified Lineup"])
    if df.empty:
        series = pd.Series([], dtype=bool, index=df.index)
    elif sig_set and player_cols:
        flags: List[bool] = []
        for _, row in df.iterrows():
            players: List[str] = []
            for col in player_cols:
                val = row.get(col)
                if isinstance(val, str):
                    cleaned = val.strip()
                    if not cleaned:
                        continue
                    cleaned = cleaned.split(":")[0].strip()
                    if "(" in cleaned:
                        cleaned = cleaned.split("(")[0].strip()
                    players.append(cleaned)
            sig = _lineup_signature(players)
            flags.append(sig in sig_set if sig else False)
        series = pd.Series(flags, index=df.index)
    else:
        series = pd.Series([bool(sig_set)] * len(df.index), index=df.index)
    insert_idx = len(player_cols) if player_cols else len(df.columns)
    df.insert(insert_idx, "Diversified Lineup", series)
    return df

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

            # Run optimizer (existing behavior); optimizer updates its own output paths
            opto.optimize(save_lineups=save_lineups)

            # Collect outputs
            output_path = getattr(opto, "output_path", None) or os.path.join(OUTPUT_DIR, "optimized_lineups.csv")
            stack_path = getattr(opto, "stack_path", None)
            progress_data["output_path"] = output_path if os.path.exists(output_path) else None
            progress_data["stack_path"] = stack_path if stack_path and os.path.exists(stack_path) else None
            if progress_data["output_path"]:
                exposure_csv = os.path.join(OUTPUT_DIR, "optimized_player_exposure.csv")
                progress_data["player_exposure_path"] = _write_player_exposure_csv(progress_data["output_path"], exposure_csv)

            # Optional portfolio diversity on optimizer outputs
            if apply_diversity and progress_data["output_path"]:
                diverse_csv, audit_json = _apply_diversity_to_optimizer(
                    opto=opto,
                    output_csv=progress_data["output_path"],
                    desired_count=opto.num_lineups,
                    max_shared_players=max_shared_players,
                    min_jaccard_distance=min_jaccard_distance,
                    per_player_cap=per_player_cap,
                    per_team_cap=per_team_cap,
                    min_stack_mix=min_stack_mix,
                )
                progress_data["diverse_path"] = diverse_csv
                progress_data["diversity_audit_path"] = audit_json
                if diverse_csv:
                    diverse_exposure_csv = os.path.join(OUTPUT_DIR, "optimized_diverse_player_exposure.csv")
                    progress_data["player_exposure_diverse_path"] = _write_player_exposure_csv(diverse_csv, diverse_exposure_csv)

            progress_data["status"] = "done"
        except Exception as e:
            progress_data["status"] = f"error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return redirect("/progress")

def _apply_diversity_to_optimizer(
    *,
    opto,
    output_csv: str,
    desired_count: int,
    max_shared_players: int,
    min_jaccard_distance: float,
    per_player_cap: Optional[float],
    per_team_cap: Optional[float],
    min_stack_mix: Optional[Dict[str, float]],
):
    # Build Candidate[] from optimizer outputs using optimizer.player_dict for Own% and TeamAbbrev
    player_dict: Dict = getattr(opto, "player_dict", {}) or {}
    candidates: List[Candidate] = []
    with open(output_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names = [row.get(str(i), "") for i in range(1, 10)]
            names = [n for n in names if n]
            owners, teams = [], []
            for n in names:
                rec = player_dict.get(n) or {}
                o = rec.get("Own%", rec.get("Own", 0.0)) or 0.0
                t = rec.get("TeamAbbrev", rec.get("Team", "")) or ""
                try:
                    owners.append(float(o))
                except Exception:
                    owners.append(0.0)
                teams.append(str(t).upper())
            candidates.append(Candidate(players=names, owners=owners, teams=teams))

    rules = DiversityRules(
        max_shared_players=max_shared_players,
        min_jaccard_distance=min_jaccard_distance,
        per_player_cap=per_player_cap,
        per_team_cap=per_team_cap,
        min_stack_mix=min_stack_mix,
        max_stack_mix=None,
        lineup_count=desired_count,
    )
    result = diversify_portfolio(candidates, rules)

    diverse_csv = os.path.join(OUTPUT_DIR, "optimized_lineups_diverse.csv")
    with open(diverse_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["Lineup"] + [str(i) for i in range(1, 10)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for idx, names in enumerate(result.lineups, start=1):
            row = {"Lineup": idx}
            for i, nm in enumerate(names, start=1):
                row[str(i)] = nm
            w.writerow(row)

    audit_json = os.path.join(OUTPUT_DIR, "diversity_audit.json")
    with open(audit_json, "w", encoding="utf-8") as fo:
        json.dump({"rules": result.metrics, "rejections": result.reasons_rejected}, fo, indent=2)

    return diverse_csv, audit_json

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

    def _run():
        progress_data["status"] = "running"
        try:
            # If diversity is requested before sim, prepare uploads/{site}/tournament_lineups.csv
            # Prefer diversified optimizer output if present; else use last optimized_lineups.csv if available; else no-op.
            if apply_diversity_sim:
                _prep_diverse_sim_inputs(
                    site=site,
                    preferred_csv=os.path.join(OUTPUT_DIR, "optimized_lineups_diverse.csv"),
                    fallback_csv=os.path.join(OUTPUT_DIR, "optimized_lineups.csv"),
                    max_shared_players=max_shared_players,
                    min_jaccard_distance=min_jaccard_distance,
                    per_player_cap=per_player_cap,
                    per_team_cap=per_team_cap,
                    min_stack_mix=min_stack_mix,
                )
                use_lineup_input = True  # ensure sim reads the prepared file

            if mode == "showdown":
                sim = NFL_Showdown_Simulator(site, field_size, num_iterations, use_contest_data, use_lineup_input)
                sim.generate_field_lineups()
                sim.run_tournament_simulation()
                lineup_path, exposure_path = sim.save_results()
                stack_path = None
            else:
                sim = NFL_GPP_Simulator(site, field_size, num_iterations, use_contest_data, use_lineup_input)
                sim.generate_field_lineups()
                sim.run_tournament_simulation()
                lineup_path, exposure_path, stack_path = sim.save_results()

            # Prepare tables for /results
            lineup_df = pd.read_csv(lineup_path)
            lineup_df = _annotate_diversified_column(lineup_df, progress_data.get("sim_diverse_lineup_signatures"))
            lineup_df.to_csv(lineup_path, index=False)
            exposure_df = pd.read_csv(exposure_path)
            tables = [
                ("Lineups (first 1000)", lineup_df.head(1000).to_html(index=False)),
                ("Exposure", exposure_df.to_html(index=False)),
            ]
            if stack_path and os.path.exists(stack_path):
                stack_df = pd.read_csv(stack_path)
                tables.append(("Stack Exposure", stack_df.to_html(index=False)))

            # Stash for results rendering
            progress_data.update({
                "output_path": lineup_path,
                "stack_path": stack_path if stack_path and os.path.exists(stack_path) else None,
                "status": "done",
            })
        except Exception as e:
            progress_data["status"] = f"error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return redirect("/progress")

def _prep_diverse_sim_inputs(
    *,
    site: str,
    preferred_csv: str,
    fallback_csv: str,
    max_shared_players: int,
    min_jaccard_distance: float,
    per_player_cap: Optional[float],
    per_team_cap: Optional[float],
    min_stack_mix: Optional[Dict[str, float]],
):
    base_csv = preferred_csv if os.path.exists(preferred_csv) else (fallback_csv if os.path.exists(fallback_csv) else None)
    if not base_csv:
        return  # nothing to diversify

    # Read base CSV -> Candidate[] (owners/teams not strictly required for pre-sim filters)
    rows: List[List[str]] = []
    with open(base_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names = [row.get(str(i), "") for i in range(1, 10)]
            names = [n for n in names if n]
            if names:
                rows.append(names)

    candidates = [Candidate(players=ln, owners=None, teams=None) for ln in rows]
    rules = DiversityRules(
        max_shared_players=max_shared_players,
        min_jaccard_distance=min_jaccard_distance,
        per_player_cap=per_player_cap,
        per_team_cap=per_team_cap,
        min_stack_mix=min_stack_mix,
        max_stack_mix=None,
        lineup_count=len(candidates),
    )
    result = diversify_portfolio(candidates, rules)

    signatures: List[str] = []
    for names in result.lineups:
        sig = _lineup_signature(names)
        if sig:
            signatures.append(sig)
    progress_data["sim_diverse_lineup_signatures"] = signatures

    # Write to OUTPUT (for download) and to uploads/{site}/tournament_lineups.csv (for simulator to read)
    out_csv = os.path.join(OUTPUT_DIR, "simulator_diverse_lineups.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["Lineup"] + [str(i) for i in range(1, 10)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for idx, names in enumerate(result.lineups, start=1):
            row = {"Lineup": idx}
            for i, nm in enumerate(names, start=1):
                row[str(i)] = nm
            w.writerow(row)

    audit_json = os.path.join(OUTPUT_DIR, "simulator_diversity_audit.json")
    with open(audit_json, "w", encoding="utf-8") as fo:
        json.dump({"rules": result.metrics, "rejections": result.reasons_rejected}, fo, indent=2)

    site_dir = os.path.join(UPLOAD_DIR, site)
    os.makedirs(site_dir, exist_ok=True)
    uploads_tourney = os.path.join(site_dir, "tournament_lineups.csv")
    # Write the exact format the simulator expects (9 columns, no Lineup index needed)
    with open(uploads_tourney, "w", newline="", encoding="utf-8") as f:
        cols = [str(i) for i in range(9)]
        w = csv.writer(f)
        for names in result.lineups:
            # if fewer than 9 (shouldn't happen), pad with blanks
            row = names[:9] + [""] * max(0, 9 - len(names))
            w.writerow(row)

    progress_data["sim_diverse_input_path"] = out_csv
    progress_data["sim_diversity_audit_path"] = audit_json

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

    # Optimizer lineups (first 1000 rows)
    if progress_data.get("output_path") and os.path.exists(progress_data["output_path"]):
        df = pd.read_csv(progress_data["output_path"], nrows=1000)
        tables.append(("Lineups (first 1000)", df.to_html(index=False)))
        lineup_url = "/download/lineups"

    # Optimizer stack exposure
    if progress_data.get("stack_path") and os.path.exists(progress_data["stack_path"]):
        stack_df = pd.read_csv(progress_data["stack_path"])
        tables.append(("Stack Exposure", stack_df.to_html(index=False)))
        stack_url = "/download/stacks"

    # Player exposure tables
    if progress_data.get("player_exposure_path") and os.path.exists(progress_data["player_exposure_path"]):
        exp_df = pd.read_csv(progress_data["player_exposure_path"])
        tables.append(("Player Exposure (All Lineups)", exp_df.to_html(index=False)))

    if progress_data.get("player_exposure_diverse_path") and os.path.exists(progress_data["player_exposure_diverse_path"]):
        exp_div_df = pd.read_csv(progress_data["player_exposure_diverse_path"])
        tables.append(("Player Exposure (Diversified Lineups)", exp_div_df.to_html(index=False)))

    # Optimizer diversified portfolio & audit
    if progress_data.get("diverse_path") and os.path.exists(progress_data["diverse_path"]):
        df2 = pd.read_csv(progress_data["diverse_path"], nrows=1000)
        tables.append(("Diversified Lineups (first 1000)", df2.to_html(index=False)))
        diverse_url = "/download/diverse_lineups"

    if progress_data.get("diversity_audit_path") and os.path.exists(progress_data["diversity_audit_path"]):
        with open(progress_data["diversity_audit_path"], "r", encoding="utf-8") as f:
            diverse_audit = json.load(f)

    # Simulator diversified inputs & audit
    if progress_data.get("sim_diverse_input_path") and os.path.exists(progress_data["sim_diverse_input_path"]):
        df3 = pd.read_csv(progress_data["sim_diverse_input_path"], nrows=1000)
        tables.append(("Simulator Diversified Inputs (first 1000)", df3.to_html(index=False)))
        sim_diverse_input_url = "/download/sim_diverse_inputs"

    if progress_data.get("sim_diversity_audit_path") and os.path.exists(progress_data["sim_diversity_audit_path"]):
        with open(progress_data["sim_diversity_audit_path"], "r", encoding="utf-8") as f:
            sim_diversity_audit = json.load(f)

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
