import pytest
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from src.ml.model import train_model, compute_model_metrics, inference, calculate_slices
from sklearn.ensemble import RandomForestClassifier


def test_compute_model_metrics_correct():
    # Test case 1: All predictions are correct
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_compute_model_metrics_incorrect():

    # Test case 2: All predictions are incorrect
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 0.0
    assert recall == 0.0
    assert fbeta == 0.0

def test_compute_model_metrics_mixed():
    # Test case 3: Mixed predictions
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    assert precision == expected_precision
    assert recall == expected_recall
    assert fbeta == expected_fbeta

def test_train_model():
    """
    # test the train model function to make sure it works
    """
    X = np.random.rand(35, 16)
    y = np.random.randint(2, size=35)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_three():
    """
    # test the output shape of the model prediction
    """
    X = np.random.rand(35, 16)
    y = np.random.randint(2, size=35)
    model = train_model(X, y)
    y_preds = inference(model, X)

    assert y.shape == y_preds.shape
