import requests
import os
import zipfile
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from api import load_and_preprocess_data,standard,save_plots_divs_to_file,evaluate_predict_plots,EDA_plots




if __name__ == "__main__":
    df = load_and_preprocess_data()
  

    EDA_plots(df)

    df,transformed_dataset,transformed_df, scaler = standard(df)

    df.to_csv('data_for_model/data.csv', index=False)
    transformed_df.to_csv('data_for_model/transformed_data.csv', index=False)
