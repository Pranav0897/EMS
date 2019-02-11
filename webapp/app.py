from flask import Flask, render_template, flash, redirect, request, url_for, session
from var import buses
from config import Config
# from forms import DataForm
from functions import *
import numpy as np

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def base():
    return render_template('base.html', buses=buses)

@app.route('/operations')
def operations():
    # paths = [session['mat_path'], session['pq_path']]
    return render_template('operations.html', buses=buses)

@app.route('/state_estimation')
def state_estimation():
    return render_template('state_estimation.html', buses=buses)

@app.route('/readme')
def readme():
    return render_template('readme.html', buses=buses)

@app.route('/data', methods=['GET', 'POST'])
def data():
    # form = DataForm()
    if request.method=='POST':
        flash("Data paths have been recorded")
        paths = []
        paths.append(request.form.get('p_path_input'))
        paths.append(request.form.get('q_path_input'))
        paths.append(request.form.get('pij_path_input'))
        paths.append(request.form.get('qij_path_input'))
        session['paths'] = paths
        return redirect(url_for('grid'))

    return render_template('data.html', buses=buses)

@app.route('/grid')
def grid():
    # paths are extracted from session. these were collected in data.html
    paths = session['paths']
    p_path_input = paths[0]
    q_path_input = paths[1]
    pij_path_input = paths[2]
    qij_path_input = paths[3]

    # read data
    p = np.load(p_path_input)
    q = np.load(q_path_input)
    pij = np.load(pij_path_input)
    qij = np.load(qij_path_input)
    
    # perform state estimation
    vse, phise = state_estimate_alt("case14.mat", p, q, pij, qij, 1.1, 0.9)
    
    # generate graph using vse, phise pass the path of the graph created to render_template
    graph_path = "/home/akash/github/EMS/webapp/bus_grid.jpg"
    return render_template('grid.html', buses=buses, graph_path = graph_path)


if(__name__ == '__main__'):
    app.run(debug=True)


# create a class which stores all relevant fields. paths, bus datas. write all updates to this class. 
# each class can be stored as a session. 
