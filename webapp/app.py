from flask import Flask, render_template, flash, redirect
from var import buses
from config import Config
from forms import LoginForm

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def base():
    return render_template('base.html', buses=buses)

@app.route('/operations')
def operations():
    return render_template('operations.html', buses=buses)

@app.route('/readme')
def readme():
    return render_template('readme.html', buses=buses)

@app.route('/data', methods=['GET', 'POST'])
def data():
    form = LoginForm()
    if form.validate_on_submit():
        flash("Data paths have been recorded")
        return redirect('/operations')
    return render_template('data.html', buses=buses, form=form)


if(__name__ == '__main__'):
    app.run(debug=True)