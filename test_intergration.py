
import unittest
from api import load_and_preprocess_data,standard,build_and_train_model,make_predictions,train_test_data,evaluate_model
import numpy as np



def test_data_processing_to_model_training():
    df = load_and_preprocess_data()
    df, transformed_dataset, transformed_df, scaler = standard(df)
    train, label, test, test_label, val, val_label, number_of_features = train_test_data(df, transformed_df)
    model, history = build_and_train_model(number_of_features, train, label, val, val_label)
    assert model is not None, "Model was not built."
    assert history is not None, "Model was not trained."
def test_model_evaluation():
    df = load_and_preprocess_data()
    df, transformed_dataset, transformed_df, scaler = standard(df)
    train, label, test, test_label, val, val_label, number_of_features = train_test_data(df, transformed_df)
    model, _ = build_and_train_model(number_of_features, train, label, val, val_label)
    mse, rmse, mae, r2, _, _ = evaluate_model(model, test, test_label, scaler)
    assert mse >= 0, "MSE is negative."
    assert rmse >= 0, "RMSE is negative."
    assert mae >= 0, "MAE is negative."
    assert -1 <= r2 <= 1, "R2 score is out of expected range."

if __name__ == '__main__':
    unittest.main()