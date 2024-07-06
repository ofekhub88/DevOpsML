
from flask import Flask, request
import pandas as pd

app = Flask(__name__)
def readpandas(file_path):
    return pd.read_csv(file_path)

@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user + '\n'

@app.route('/size')
def size():
    file_path = request.args.get('filename')
    df = readpandas(file_path)
    return str(len(df.index)) + '\n'

@app.route('/summary')
def summary():
    file_path = request.args.get('filename')
    df = readpandas(file_path)
    return str(df.mean()) + '\n'


app.run(host='0.0.0.0', port=8000)




