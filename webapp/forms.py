from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    path_input = StringField('path', validators=[DataRequired()])
    submit = SubmitField('submit')
