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
    df, _, _, scaler = standard(load_and_preprocess_data())
    model = build_and_train_model(7, np.empty((0)), np.empty((0)), np.empty((0)), np.empty((0)))[0] # Assuming 7 features for simplicity
    predictions = make_predictions(df, scaler, model)
    assert not predictions.empty, "No predictions were made."


if __name__ == '__main__':
    unittest.main()
