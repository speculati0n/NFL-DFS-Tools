import os
import pandas as pd
from flask import Flask, render_template, request, redirect
from src.nfl_optimizer import NFL_Optimizer
from src.nfl_showdown_optimizer import NFL_Showdown_Optimizer
from src.nfl_gpp_simulator import NFL_GPP_Simulator
from src.nfl_showdown_simulator import NFL_Showdown_Simulator

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    site = request.form['site'].strip().lower()
    data_dir = os.path.join(BASE_DIR, f"{site}_data")
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
        config.save(os.path.join(BASE_DIR, 'config.json'))
    return redirect('/')

@app.route('/optimize', methods=['POST'])
def optimize():
    site = request.form['site'].strip().lower()
    num_lineups = request.form['num_lineups']
    num_uniques = request.form['num_uniques']
    mode = request.form.get('mode', 'classic')

    if mode == 'showdown':
        opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
    else:
        opto = NFL_Optimizer(site, num_lineups, num_uniques)
    opto.optimize()
    output_path = opto.output()
    df = pd.read_csv(output_path)
    tables = [("Lineups", df.to_html(index=False))]
    return render_template('results.html', title='Optimization Results', tables=tables)

@app.route('/simulate', methods=['POST'])
def simulate():
    site = request.form['site'].strip().lower()
    field_size = request.form['field_size']
    num_iterations = request.form['num_iterations']
    mode = request.form.get('mode', 'classic')

    if mode == 'showdown':
        sim = NFL_Showdown_Simulator(site, field_size, num_iterations, False, False)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        lineup_path, exposure_path = sim.save_results()
    else:
        sim = NFL_GPP_Simulator(site, field_size, num_iterations, False, False)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        lineup_path, exposure_path = sim.output()

    lineup_df = pd.read_csv(lineup_path)
    exposure_df = pd.read_csv(exposure_path)
    tables = [
        ("Lineups", lineup_df.to_html(index=False)),
        ("Exposure", exposure_df.to_html(index=False)),
    ]
    return render_template('results.html', title='Simulation Results', tables=tables)

if __name__ == '__main__':
    app.run(debug=True)
