from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class DataForm(FlaskForm):
    mat_path_input = StringField('path', validators=[DataRequired()])
    pqij_path_input = StringField('path', validators=[DataRequired()])
    submit = SubmitField('submit')
