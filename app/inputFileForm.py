from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField


class InputFileForm(FlaskForm):
    searchbox = StringField('Что ищем?')
    file = FileField()
    submit = SubmitField('Submit')
