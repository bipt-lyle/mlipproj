from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import plotly
from plotly.offline import plot
import plotly.graph_objs as go
import json
import matplotlib.pyplot as plt
import seaborn as sns
from api import build_and_train_model,load_and_preprocess_data,evaluate_model,make_predictions,train_test_data,standard,evaluate_predict_plots
from io import BytesIO
import base64
app = Flask(__name__)

@app.route('/')
def home():
    plots_divs, metrics,predictions_html = evaluate_predict_plots()
    return render_template('index.html',predictions_html=predictions_html)

@app.route('/EDA')
def EDA():
    with open('statics/EDA_plots.html', 'r') as file:
        eda_plots_html = file.read()
    return render_template('eda.html', eda_plots_html=eda_plots_html)

@app.route('/eva')
def eva():
    # with open('statics/evaluate_predict.html', 'r') as file:
    #     eva_plots_html = file.read()
    plots_divs, metrics,predictions_html = evaluate_predict_plots()
    return render_template('evaluate_predict_page.html', metrics=metrics,plots_divs=plots_divs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
