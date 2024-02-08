from api import train_test_data,build_and_train_model,make_predictions,evaluate_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
if __name__ == "__main__":
    file_path = 'data_for_model/data.csv'
    df = pd.read_csv(file_path)
    scaler = joblib.load('data_for_model/scaler.save')
    file2_path = 'data_for_model/transformed_data.csv'
    transformed_df = pd.read_csv(file2_path)

    train,label,test,test_label,val,val_label,number_of_features = train_test_data(df,transformed_df)
    build_and_train_model(number_of_features,train,label,val,val_label)
    
    
    
