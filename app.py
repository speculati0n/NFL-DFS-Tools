import os
import tempfile
import zipfile
from flask import Flask, render_template, request, send_file

# Attempt to import core modules but capture missing dependencies so the
# application can surface a helpful message instead of crashing outright.
missing_dependency = None
try:
    from src.nfl_optimizer import NFL_Optimizer
    from src.nfl_showdown_optimizer import NFL_Showdown_Optimizer
    from src.nfl_gpp_simulator import NFL_GPP_Simulator
    from src.nfl_showdown_simulator import NFL_Showdown_Simulator
except ModuleNotFoundError as e:
    # Store the error message for later use. Individual routes will report it
    # back to the user if they attempt to run an action that requires the
    # missing package.
    missing_dependency = str(e)
    NFL_Optimizer = NFL_Showdown_Optimizer = None
    NFL_GPP_Simulator = NFL_Showdown_Simulator = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    if missing_dependency:
        return missing_dependency, 500

    site = request.form['site']
    num_lineups = request.form['num_lineups']
    num_uniques = request.form['num_uniques']
    mode = request.form.get('mode', 'classic')

    try:
        if mode == 'showdown':
            opto = NFL_Showdown_Optimizer(site, num_lineups, num_uniques)
        else:
            opto = NFL_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        output_path = opto.output()
        return send_file(output_path, as_attachment=True)
    except ModuleNotFoundError as e:
        # Surface missing dependency errors directly to the requester.
        return str(e), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    if missing_dependency:
        return missing_dependency, 500

    site = request.form['site']
    field_size = request.form['field_size']
    num_iterations = request.form['num_iterations']
    mode = request.form.get('mode', 'classic')

    try:
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

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(tmp.name, 'w') as zf:
            zf.write(lineup_path, os.path.basename(lineup_path))
            zf.write(exposure_path, os.path.basename(exposure_path))
        return send_file(tmp.name, as_attachment=True, download_name='results.zip')
    except ModuleNotFoundError as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
