from flask import Flask, render_template, flash, redirect, request, url_for, session
from var import buses
from config import Config
# from forms import DataForm
from functions import *
import numpy as np
import time
app = Flask(__name__)
app.config.from_object(Config)
init_time=0

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)


@app.route('/')
@nocache
def base():
    return render_template('base.html', buses=buses)

@app.route('/operations')
@nocache
def operations():
    # paths = [session['mat_path'], session['pq_path']]
    return render_template('operations.html', buses=buses)

@app.route('/state_estimation')
@nocache
def state_estimation():
    global init_time
    graph_path = "/Users/pranav97/Downloads/EMS-master/webapp/static/bus_grid.png".rstrip()
    if init_time==0 or (time.time()-init_time)>=300.0:
        init_time=time.time()
        graph_path=se()
        init_time=time.time()
        return render_template('grid.html', buses=buses, graph_path = graph_path)
    else:
        return render_template('grid.html', buses=buses, graph_path = graph_path)
    # return render_template('state_estimation.html', buses=buses)

@app.route('/readme')
@nocache
def readme():
    return render_template('readme.html', buses=buses)

@app.route('/data', methods=['GET', 'POST'])
@nocache
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
@nocache
def grid():
    global init_time
    # paths are extracted from session. these were collected in data.html
    if('paths' in session.keys()):
        if init_time==0 or (time.time()-init_time)>=300.0:
            init_time=time.time()
            #do state estimation again
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
            print(p,q)
            vse, phise = state_estimate_alt("case14.mat", p, q, pij, qij, 1.1, 0.9)
            
            # generate graph using vse, phise pass the path of the graph created to render_template
            graph_path = "/Users/pranav97/Downloads/EMS-master/webapp/static/bus_grid.png".rstrip()
            draw_graph(pij,qij,vse,phise,graph_path)
            init_time=time.time()
            return render_template('grid.html', buses=buses, graph_path = graph_path)
        else:
            #SE done <5 mins ago, just render the old image
            print("Last SE less than 5 mins ago. Rendering old image.")
            graph_path = "/Users/pranav97/Downloads/EMS-master/webapp/static/bus_grid.jpg".rstrip()
            return render_template('grid.html', buses=buses, graph_path = graph_path)

    else:
        return "please enter paths first"


if(__name__ == '__main__'):
    app.run(debug=True)


# create a class which stores all relevant fields. paths, bus datas. write all updates to this class. 
# each class can be stored as a session. 
