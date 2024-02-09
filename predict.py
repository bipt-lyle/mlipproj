from api import evaluate_predict_plots, make_predictions, configure_mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import load_model
import os
import pickle
import mlflow
from datetime import datetime



if __name__ == "__main__":
    evaluate_predict_plots()
