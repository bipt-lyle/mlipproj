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
    # evaluate_predict_plots()

    # 配置MLflow（这个函数需要在你的API脚本中定义，如果已经定义了可以忽略这步）
    configure_mlflow()

    # 开始MLflow记录
    with mlflow.start_run():
        # 执行预测和评估绘图
        current_date = datetime.now().strftime("%Y-%m-%d")
        mlflow.log_param("run_date", current_date)

        plots_divs, metrics, predictions_html = evaluate_predict_plots()

        mlflow.log_param("batch_size", 64)
        mlflow.log_param("epochs", 120)
        mlflow.log_param("number_of_features", 7)
        
        # 记录模型评估指标
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # 记录预测结果的HTML为工件
        html_file_path = 'statics/evaluate_predict.html'
        # 将 HTML 文件记录为工件
        mlflow.log_artifact(html_file_path)

        predictions_file = "predictions.html"
        with open(predictions_file, "w") as f:
            f.write(predictions_html)
        mlflow.log_artifact(predictions_file)
        # 可以选择记录其他信息，例如模型参数或配置

    print("MLflow tracking completed.")
