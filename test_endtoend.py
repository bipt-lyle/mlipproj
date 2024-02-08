from api import load_and_preprocess_data,standard,build_and_train_model,make_predictions,train_test_data,evaluate_model
import numpy as np
import unittest


def test_end_to_end_workflow():
    df = load_and_preprocess_data()
    df, transformed_dataset, transformed_df, scaler = standard(df)
    train, label, test, test_label, val, val_label, number_of_features = train_test_data(df, transformed_df)
    model, _ = build_and_train_model(number_of_features, train, label, val, val_label)
    predictions = make_predictions(df, scaler, model)
    assert not predictions.empty, "End-to-end workflow failed to generate predictions."


if __name__ == '__main__':
    unittest.main()