"""Tests for ML toolkit."""

import pytest
import numpy as np
from ml_tools.ml_toolkit import MLToolkit


def test_ml_toolkit_initialization():
    """Test ML toolkit initialization."""
    toolkit = MLToolkit()
    assert toolkit is not None
    assert isinstance(toolkit.models, dict)
    assert len(toolkit.models) == 0


def test_train_classifier():
    """Test training a classifier."""
    toolkit = MLToolkit()
    
    # Generate sample data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    result = toolkit.train_classifier(X, y, model_name="test_clf")
    
    assert result['success'] is True
    assert 'test_clf' in toolkit.models
    assert 0 <= result['train_accuracy'] <= 1
    assert 0 <= result['test_accuracy'] <= 1


def test_train_regressor():
    """Test training a regressor."""
    toolkit = MLToolkit()
    
    # Generate sample data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    result = toolkit.train_regressor(X, y, model_name="test_reg")
    
    assert result['success'] is True
    assert 'test_reg' in toolkit.models
    assert result['train_rmse'] >= 0
    assert result['test_rmse'] >= 0


def test_clustering():
    """Test clustering."""
    toolkit = MLToolkit()
    
    # Generate sample data
    X = np.random.randn(50, 3)
    
    result = toolkit.cluster_data(X, n_clusters=3, model_name="test_cluster")
    
    assert result['success'] is True
    assert 'test_cluster' in toolkit.models
    assert result['n_clusters'] == 3
    assert len(result['labels']) == 50


def test_predict():
    """Test making predictions."""
    toolkit = MLToolkit()
    
    # Train a model first
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    toolkit.train_classifier(X, y, model_name="test_clf")
    
    # Make predictions
    X_new = np.random.randn(5, 10)
    result = toolkit.predict(X_new, model_name="test_clf")
    
    assert result['success'] is True
    assert len(result['predictions']) == 5


def test_predict_nonexistent_model():
    """Test prediction with non-existent model."""
    toolkit = MLToolkit()
    
    X = np.random.randn(5, 10)
    result = toolkit.predict(X, model_name="nonexistent")
    
    assert result['success'] is False
    assert 'error' in result


def test_analyze():
    """Test general analyze method."""
    toolkit = MLToolkit()
    
    result = toolkit.analyze(None, task="classification")
    
    assert result['task'] == "classification"
    assert 'message' in result
