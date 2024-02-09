# tests/test_model.py

import unittest
from api import load_and_preprocess_data,standard,build_and_train_model,make_predictions
import numpy as np

def test_load_and_preprocess_data():
    df = load_and_preprocess_data()
    assert not df.empty, "DataFrame is empty after loading and preprocessing data."


def test_standard():
    df = load_and_preprocess_data()
    _, _, _, scaler = standard(df)
    assert scaler is not None, "Scaler object was not created."

def test_make_predictions():
    number_of_features = 7
    window_length = 5
    train_data = np.random.rand(100, window_length, number_of_features)
    train_labels = np.random.rand(100, number_of_features)
    validation_data = np.random.rand(20, window_length, number_of_features)
    validation_labels = np.random.rand(20, number_of_features)
    df, _, _, scaler = standard(load_and_preprocess_data())
    model, history = build_and_train_model(number_of_features, train_data, train_labels, validation_data, validation_labels)
    predictions = make_predictions(df, scaler, model)
    assert not predictions.empty, "No predictions were made."


if __name__ == '__main__':
    unittest.main()
