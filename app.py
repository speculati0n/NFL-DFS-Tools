import os
import shutil
import threading
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify
from src.nfl_optimizer import NFL_Optimizer
from src.nfl_showdown_optimizer import NFL_Showdown_Optimizer
from src.nfl_gpp_simulator import NFL_GPP_Simulator
from src.nfl_showdown_simulator import NFL_Showdown_Simulator

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

progress_data = {"current": 0, "total": 0, "percent": 0, "status": "idle", "output_path": None}


def update_progress(current, total):
    progress_data["current"] = current
    progress_data["total"] = total
    progress_data["percent"] = int((current / total) * 100)


def run_optimizer(opto, site, save_lineups):
    opto.optimize(progress_callback=update_progress)
    output_path = opto.output()
    if save_lineups:
        dest_dir = os.path.join(UPLOAD_DIR, site)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(output_path, os.path.join(dest_dir, "tournament_lineups.csv"))
    progress_data["output_path"] = output_path
    progress_data["status"] = "done"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    site = request.form['site'].strip().lower()
    data_dir = os.path.join(UPLOAD_DIR, site)
    os.makedirs(data_dir, exist_ok=True)
    projections = request.files.get('projections')
    players = request.files.get('players')
    contest = request.files.get('contest')
    config = request.files.get('config')
    if projections and projections.filename:
        projections.save(os.path.join(data_dir, 'projections.csv'))
    if players and players.filename:
        players.save(os.path.join(data_dir, 'player_ids.csv'))
    if contest and contest.filename:
        contest.save(os.path.join(data_dir, 'contest_structure.csv'))
    if config and config.filename:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        config.save(os.path.join(UPLOAD_DIR, 'config.json'))
    return redirect('/')

@app.route('/optimize', methods=['POST'])
def optimize():
    site = request.form['site'].strip().lower()
    num_lineups = request.form['num_lineups']
    num_uniques = request.form['num_uniques']
    mode = request.form.get('mode', 'classic')
    save_lineups = 'save_lineups' in request.form

    if mode == 'showdown':
        opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
        total = opto.num_lineups
    else:
        opto = NFL_Optimizer(site, num_lineups, num_uniques)


    progress_data.update({'current': 0, 'total': total, 'percent': 0, 'status': 'running', 'output_path': None})

    thread = threading.Thread(target=run_optimizer, args=(opto, site, save_lineups))
    thread.start()

    return render_template('progress.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    site = request.form['site'].strip().lower()
    field_size = request.form['field_size']
    num_iterations = request.form['num_iterations']
    mode = request.form.get('mode', 'classic')
    use_contest_data = 'use_contest_data' in request.form
    use_lineup_input = 'use_lineup_input' in request.form

    if mode == 'showdown':
        sim = NFL_Showdown_Simulator(site, field_size, num_iterations, use_contest_data, use_lineup_input)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        lineup_path, exposure_path = sim.save_results()
    else:
        sim = NFL_GPP_Simulator(site, field_size, num_iterations, use_contest_data, use_lineup_input)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        lineup_path, exposure_path = sim.output()

    # Limit displayed lineups to the first 1000 while keeping full export files
    lineup_df = pd.read_csv(lineup_path, nrows=1000)
    exposure_df = pd.read_csv(exposure_path)
    tables = [
        ("Lineups (first 1000)", lineup_df.to_html(index=False)),
        ("Exposure", exposure_df.to_html(index=False)),
    ]
    return render_template('results.html', title='Simulation Results', tables=tables)


@app.route('/reset', methods=['POST'])
def reset():
    site = request.form['site'].strip().lower()
    shutil.rmtree(os.path.join(UPLOAD_DIR, site), ignore_errors=True)
    config_path = os.path.join(UPLOAD_DIR, 'config.json')
    if os.path.exists(config_path):
        os.remove(config_path)
    return redirect('/')


@app.route('/progress')
def progress():
    return jsonify(progress_data)


@app.route('/results')
def results():
    if not progress_data.get('output_path'):
        return redirect('/')
    df = pd.read_csv(progress_data['output_path'], nrows=1000)
    tables = [("Lineups (first 1000)", df.to_html(index=False))]
    return render_template('results.html', title='Optimization Results', tables=tables)

if __name__ == '__main__':
    app.run(debug=True)
