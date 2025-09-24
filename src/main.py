import csv
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from anti_cannibalizer import Candidate, DiversityRules, diversify_portfolio
from nfl_optimizer import NFL_Optimizer
from nfl_showdown_optimizer import NFL_Showdown_Optimizer
from nfl_gpp_simulator import NFL_GPP_Simulator
from nfl_showdown_simulator import NFL_Showdown_Simulator

PLAYER_COLS = [
    "QB",
    "RB1",
    "RB2",
    "WR1",
    "WR2",
    "WR3",
    "TE",
    "FLEX",
    "DST",
]

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
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
METADATA_PATH = os.path.join(OUTPUT_DIR, "lineup_player_metadata.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _coerce_int(value: str, default: int) -> Tuple[int, Optional[str]]:
    raw = (value or "").strip()
    if not raw:
        return default, None
    try:
        return int(float(raw)), None
    except (TypeError, ValueError):
        return default, f"Invalid integer value '{value}'. Using default {default}."


def _coerce_float(value: str, default: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    raw = (value or "").strip()
    if not raw:
        return default, None
    try:
        return float(raw), None
    except (TypeError, ValueError):
        if default is None:
            return None, f"Invalid number '{value}'. Ignoring field."
        return default, f"Invalid number '{value}'. Using default {default}."


def _parse_stack_mix(value: str) -> Tuple[Optional[Dict[str, float]], List[str]]:
    raw = (value or "").strip()
    if not raw:
        return None, []
    result: Dict[str, float] = {}
    warnings: List[str] = []
    for piece in raw.split(","):
        if not piece.strip():
            continue
        if ":" not in piece:
            warnings.append(f"Ignoring stack mix entry '{piece.strip()}'. Expected key:value format.")
            continue
        key, val = piece.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            result[key] = float(val)
        except ValueError:
            warnings.append(f"Invalid stack mix weight '{val}' for key '{key}'.")
    return (result or None), warnings


def _build_player_lookup(player_dict: Optional[Dict]) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    if not isinstance(player_dict, dict):
        return lookup
    for rec in player_dict.values():
        if not isinstance(rec, dict):
            continue
        name = str(
            rec.get("Name")
            or rec.get("PLAYER_NAME")
            or rec.get("Player")
            or rec.get("player_name")
            or rec.get("name")
            or ""
        ).strip()
        player_id = str(
            rec.get("ID")
            or rec.get("Id")
            or rec.get("PlayerID")
            or rec.get("player_id")
            or rec.get("id")
            or ""
        ).strip()
        alt_key = str(rec.get("key") or rec.get("Key") or rec.get("PLAYERID") or "").strip()
        keys = {name, player_id, alt_key}
        if name and player_id:
            keys.add(f"{name} ({player_id})")
            keys.add(f"{player_id}:{name}")
        for key in list(keys):
            if not key:
                continue
            lookup[key.lower()] = rec
    return lookup


def _lookup_record(lookup: Dict[str, Dict], display: str) -> Optional[Dict]:
    if not display:
        return None
    key = display.strip().lower()
    if key in lookup:
        return lookup[key]
    if "(" in display and ")" in display:
        inner = display.split("(")[-1].split(")")[0].strip().lower()
        if inner in lookup:
            return lookup[inner]
    if ":" in display:
        left, right = display.split(":", 1)
        left_key = left.strip().lower()
        right_key = right.strip().lower()
        if left_key in lookup:
            return lookup[left_key]
        if right_key in lookup:
            return lookup[right_key]
    return None


def _extract_owner(rec: Dict) -> float:
    for key in ("Own%", "Ownership", "Ownership%", "ProjOwn", "Own", "Proj Ownership"):
        if key in rec and rec[key] is not None:
            try:
                return float(rec[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _extract_team(rec: Dict) -> str:
    return str(
        rec.get("TeamAbbrev")
        or rec.get("Team")
        or rec.get("team")
        or rec.get("TeamAbbreviation")
        or ""
    ).upper()


def _build_candidates_from_csv(
    csv_path: str,
    *,
    player_dict: Optional[Dict] = None,
    metadata: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[List[Candidate], Dict[str, Dict[str, object]]]:
    candidates: List[Candidate] = []
    metadata_out: Dict[str, Dict[str, object]] = dict(metadata or {})
    lookup = _build_player_lookup(player_dict)
    if not os.path.exists(csv_path):
        return candidates, metadata_out

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return candidates, metadata_out
        cols = [c for c in reader.fieldnames if c.strip() in PLAYER_COLS]
        if not cols:
            cols = reader.fieldnames[:9]
        for row in reader:
            players: List[str] = []
            owners: List[float] = []
            teams: List[str] = []
            for col in cols:
                raw = row.get(col, "")
                name = str(raw).strip()
                if not name:
                    continue
                players.append(name)
                record = _lookup_record(lookup, name)
                owner_val: float = 0.0
                team_val: str = ""
                if record:
                    owner_val = _extract_owner(record)
                    team_val = _extract_team(record)
                elif name in metadata_out:
                    meta = metadata_out[name]
                    owner_val = float(meta.get("ownership", 0.0) or 0.0)
                    team_val = str(meta.get("team", "") or "").upper()
                owners.append(owner_val)
                teams.append(team_val)
                if name not in metadata_out:
                    metadata_out[name] = {"ownership": owner_val, "team": team_val}
            if players:
                candidates.append(Candidate(players=players, owners=owners, teams=teams))
    return candidates, metadata_out


def _write_diverse_csv(path: str, lineups: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["Lineup"] + [f"Player{i}" for i in range(1, 10)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, lineup in enumerate(lineups, start=1):
            row = [idx]
            slots = list(lineup)[:9]
            if len(slots) < 9:
                slots.extend([""] * (9 - len(slots)))
            row.extend(slots)
            writer.writerow(row)


def _write_sim_upload(path: str, lineups: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(i) for i in range(9)])
        for lineup in lineups:
            slots = list(lineup)[:9]
            if len(slots) < 9:
                slots.extend([""] * (9 - len(slots)))
            writer.writerow(slots)


def _write_audit(path: str, result) -> Dict[str, object]:
    payload = {"rules": result.metrics, "rejections": result.reasons_rejected}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _prepare_sim_diversity(
    site: str,
    *,
    max_shared_players: int,
    min_jaccard_distance: float,
    per_player_cap: Optional[float],
    per_team_cap: Optional[float],
    min_stack_mix: Optional[Dict[str, float]],
) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    preferred = os.path.join(OUTPUT_DIR, "optimized_lineups_diverse.csv")
    fallback = os.path.join(OUTPUT_DIR, "optimized_lineups.csv")
    base_csv = preferred if os.path.exists(preferred) else fallback if os.path.exists(fallback) else None
    if not base_csv:
        return None, None

    metadata: Dict[str, Dict[str, object]] = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    candidates, metadata = _build_candidates_from_csv(base_csv, metadata=metadata)
    if not candidates:
        return None, None

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

    diverse_csv = os.path.join(OUTPUT_DIR, "simulator_diverse_lineups.csv")
    audit_path = os.path.join(OUTPUT_DIR, "simulator_diversity_audit.json")
    _write_diverse_csv(diverse_csv, result.lineups)
    audit_payload = _write_audit(audit_path, result)

    upload_dir = os.path.join(UPLOAD_DIR, site)
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, "tournament_lineups.csv")
    _write_sim_upload(upload_path, result.lineups)

    # Persist metadata for downstream simulator runs
    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass

    return diverse_csv, audit_payload

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
    st.markdown("### Portfolio Diversity (Anti-Cannibalization)")
    apply_diversity = st.checkbox("Apply diversity guards", key="opt_apply_diversity")
    st.caption("Leave fields blank to use defaults.")
    max_shared_players_input = st.text_input("Max shared players", value="", key="opt_max_shared")
    min_jaccard_input = st.text_input("Min Jaccard distance", value="", key="opt_min_jaccard")
    per_player_cap_input = st.text_input("Per-player cap (0..1)", value="", key="opt_per_player")
    per_team_cap_input = st.text_input("Per-team cap (0..1)", value="", key="opt_per_team")
    stack_mix_input = st.text_input(
        "Min stack mix (k:v,k:v)",
        value="",
        key="opt_stack_mix",
        placeholder="QB+WR:0.30,QB+WR+TE:0.15,No Stack:0.10",
    )
    submitted_opt = st.form_submit_button("Run Optimizer")
    if submitted_opt:
        warnings: List[str] = []
        max_shared_val = 6
        min_jaccard_val = 0.20
        per_player_cap_val: Optional[float] = 0.45
        per_team_cap_val: Optional[float] = 0.40
        stack_mix_val: Optional[Dict[str, float]] = None
        if apply_diversity:
            max_shared_val, warn = _coerce_int(max_shared_players_input, 6)
            if warn:
                warnings.append(warn)
            min_jaccard_val, warn = _coerce_float(min_jaccard_input, 0.20)
            if warn:
                warnings.append(warn)
            per_player_cap_val, warn = _coerce_float(per_player_cap_input, 0.45)
            if warn:
                warnings.append(warn)
            per_team_cap_val, warn = _coerce_float(per_team_cap_input, 0.40)
            if warn:
                warnings.append(warn)
            stack_mix_val, stack_warnings = _parse_stack_mix(stack_mix_input)
            warnings.extend(stack_warnings)

        if mode_opt == "showdown":
            optimizer = NFL_Showdown_Optimizer(site_opt, num_lineups, num_uniques)
        else:
            optimizer = NFL_Optimizer(site_opt, num_lineups, num_uniques)
        optimizer.optimize()
        lineup_path, stack_path = optimizer.output()
        base_lineup_path = lineup_path
        canonical_lineup_path = os.path.join(OUTPUT_DIR, "optimized_lineups.csv")
        try:
            shutil.copy(lineup_path, canonical_lineup_path)
            base_lineup_path = canonical_lineup_path
        except Exception:
            canonical_lineup_path = lineup_path

        canonical_stack_path = None
        if stack_path:
            canonical_stack_path = os.path.join(OUTPUT_DIR, "stack_exposure.csv")
            try:
                shutil.copy(stack_path, canonical_stack_path)
            except Exception:
                canonical_stack_path = stack_path

        player_dict = getattr(optimizer, "player_dict", {})
        candidates, metadata = _build_candidates_from_csv(
            base_lineup_path, player_dict=player_dict
        )
        try:
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass

        if save_lineups:
            dest_dir = os.path.join(UPLOAD_DIR, site_opt)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(base_lineup_path, os.path.join(dest_dir, "tournament_lineups.csv"))

        if warnings:
            for message in warnings:
                st.warning(message)

        df = pd.read_csv(base_lineup_path)
        st.subheader("Lineups")
        st.dataframe(df)
        try:
            with open(base_lineup_path, "rb") as f:
                st.download_button(
                    "Download Lineups CSV",
                    f.read(),
                    file_name="optimized_lineups.csv",
                    mime="text/csv",
                )
        except Exception:
            pass
        if stack_path:
            stack_source = canonical_stack_path or stack_path
            stack_df = pd.read_csv(stack_source)
            st.subheader("Stack Exposure")
            st.dataframe(stack_df)
            try:
                with open(stack_source, "rb") as f:
                    st.download_button(
                        "Download Stack Exposure CSV",
                        f.read(),
                        file_name="stack_exposure.csv",
                        mime="text/csv",
                    )
            except Exception:
                pass

        diverse_audit: Optional[Dict[str, object]] = None
        diverse_csv_path: Optional[str] = None
        if apply_diversity:
            if not candidates:
                st.warning("No lineups available to diversify.")
            else:
                rules = DiversityRules(
                    max_shared_players=max_shared_val,
                    min_jaccard_distance=min_jaccard_val,
                    per_player_cap=per_player_cap_val,
                    per_team_cap=per_team_cap_val,
                    min_stack_mix=stack_mix_val,
                    max_stack_mix=None,
                    lineup_count=min(len(candidates), int(num_lineups)),
                )
                result = diversify_portfolio(candidates, rules)
                diverse_csv_path = os.path.join(OUTPUT_DIR, "optimized_lineups_diverse.csv")
                audit_path = os.path.join(OUTPUT_DIR, "diversity_audit.json")
                _write_diverse_csv(diverse_csv_path, result.lineups)
                diverse_audit = _write_audit(audit_path, result)

                diverse_df = pd.read_csv(diverse_csv_path)
                st.subheader("Diversified Lineups")
                st.dataframe(diverse_df)
                try:
                    with open(diverse_csv_path, "rb") as f:
                        st.download_button(
                            "Download Diversified Lineups CSV",
                            f.read(),
                            file_name="optimized_lineups_diverse.csv",
                            mime="text/csv",
                        )
                except Exception:
                    pass

                st.subheader("Diversity Audit (Optimizer)")
                metrics = diverse_audit.get("rules", {}) if diverse_audit else {}
                for key in [
                    "pairwise_jaccard_mean",
                    "player_HHI",
                    "max_shared_players",
                    "min_jaccard_distance",
                    "per_player_cap",
                    "per_team_cap",
                ]:
                    if key in metrics:
                        st.write(f"**{key}:** {metrics[key]}")
                rejections = (diverse_audit or {}).get("rejections", {})
                if rejections:
                    rej_df = pd.DataFrame(
                        [(k, v) for k, v in rejections.items()],
                        columns=["Reason", "Count"],
                    )
                    st.dataframe(rej_df)

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
    st.markdown("### Portfolio Diversity Before Simulation")
    apply_diversity_sim = st.checkbox(
        "Apply diversity guards before sim", key="sim_apply_diversity"
    )
    st.caption("Uses the guard fields below; leave blank for defaults.")
    sim_max_shared_input = st.text_input("Max shared players", value="", key="sim_max_shared")
    sim_min_jaccard_input = st.text_input(
        "Min Jaccard distance", value="", key="sim_min_jaccard"
    )
    sim_per_player_input = st.text_input(
        "Per-player cap (0..1)", value="", key="sim_per_player"
    )
    sim_per_team_input = st.text_input(
        "Per-team cap (0..1)", value="", key="sim_per_team"
    )
    sim_stack_mix_input = st.text_input(
        "Min stack mix (k:v,k:v)",
        value="",
        key="sim_stack_mix",
        placeholder="QB+WR:0.30,QB+WR+TE:0.15,No Stack:0.10",
    )
    submitted_sim = st.form_submit_button("Run Simulation")
    if submitted_sim:
        sim_warnings: List[str] = []
        sim_max_shared = 6
        sim_min_jaccard = 0.20
        sim_per_player_cap: Optional[float] = 0.60
        sim_per_team_cap: Optional[float] = 0.60
        sim_stack_mix: Optional[Dict[str, float]] = None
        if apply_diversity_sim:
            sim_max_shared, warn = _coerce_int(sim_max_shared_input, 6)
            if warn:
                sim_warnings.append(warn)
            sim_min_jaccard, warn = _coerce_float(sim_min_jaccard_input, 0.20)
            if warn:
                sim_warnings.append(warn)
            sim_per_player_cap, warn = _coerce_float(sim_per_player_input, 0.60)
            if warn:
                sim_warnings.append(warn)
            sim_per_team_cap, warn = _coerce_float(sim_per_team_input, 0.60)
            if warn:
                sim_warnings.append(warn)
            sim_stack_mix, stack_warnings = _parse_stack_mix(sim_stack_mix_input)
            sim_warnings.extend(stack_warnings)

        sim_diverse_payload: Optional[Dict[str, object]] = None
        sim_diverse_path: Optional[str] = None
        if apply_diversity_sim:
            result = _prepare_sim_diversity(
                site_sim,
                max_shared_players=sim_max_shared,
                min_jaccard_distance=sim_min_jaccard,
                per_player_cap=sim_per_player_cap,
                per_team_cap=sim_per_team_cap,
                min_stack_mix=sim_stack_mix,
            )
            sim_diverse_path, sim_diverse_payload = result
            if sim_diverse_path:
                use_lineup_input = True
            else:
                st.warning(
                    "No optimizer lineups were found to diversify before simulation."
                )

        if sim_warnings:
            for message in sim_warnings:
                st.warning(message)

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
        lineup_df = pd.read_csv(lineup_path)
        exposure_df = pd.read_csv(exposure_path)
        st.subheader("Lineups")
        st.dataframe(lineup_df)
        try:
            with open(lineup_path, "rb") as f:
                st.download_button(
                    "Download Simulation Lineups CSV",
                    f.read(),
                    file_name=os.path.basename(lineup_path),
                    mime="text/csv",
                )
        except Exception:
            pass
        st.subheader("Exposure")
        st.dataframe(exposure_df)
        try:
            with open(exposure_path, "rb") as f:
                st.download_button(
                    "Download Exposure CSV",
                    f.read(),
                    file_name=os.path.basename(exposure_path),
                    mime="text/csv",
                )
        except Exception:
            pass
        if stack_path:
            stack_df = pd.read_csv(stack_path)
            st.subheader("Stack Exposure")
            st.dataframe(stack_df)
            try:
                with open(stack_path, "rb") as f:
                    st.download_button(
                        "Download Simulation Stack Exposure CSV",
                        f.read(),
                        file_name=os.path.basename(stack_path),
                        mime="text/csv",
                    )
            except Exception:
                pass

        if sim_diverse_path and os.path.exists(sim_diverse_path):
            diverse_inputs_df = pd.read_csv(sim_diverse_path)
            st.subheader("Simulator Diversified Inputs")
            st.dataframe(diverse_inputs_df)
            try:
                with open(sim_diverse_path, "rb") as f:
                    st.download_button(
                        "Download Diversified Simulator Inputs CSV",
                        f.read(),
                        file_name="simulator_diverse_lineups.csv",
                        mime="text/csv",
                    )
            except Exception:
                pass

        if sim_diverse_payload:
            st.subheader("Diversity Audit (Simulator)")
            metrics = sim_diverse_payload.get("rules", {}) if sim_diverse_payload else {}
            for key in [
                "pairwise_jaccard_mean",
                "player_HHI",
                "max_shared_players",
                "min_jaccard_distance",
                "per_player_cap",
                "per_team_cap",
            ]:
                if key in metrics:
                    st.write(f"**{key}:** {metrics[key]}")
            rejections = (sim_diverse_payload or {}).get("rejections", {})
            if rejections:
                rej_df = pd.DataFrame(
                    [(k, v) for k, v in rejections.items()],
                    columns=["Reason", "Count"],
                )
                st.dataframe(rej_df)
