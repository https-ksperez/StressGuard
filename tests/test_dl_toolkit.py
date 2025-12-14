"""Tests for DL toolkit."""

import pytest
import numpy as np
from dl_tools.dl_toolkit import DLToolkit


def test_dl_toolkit_initialization():
    """Test DL toolkit initialization."""
    toolkit = DLToolkit()
    assert toolkit is not None
    assert isinstance(toolkit.models, dict)
    assert len(toolkit.models) == 0


def test_create_sequential_model():
    """Test creating a sequential model."""
    toolkit = DLToolkit()
    
    result = toolkit.create_sequential_model(
        input_shape=(10,),
        output_units=3,
        hidden_layers=[32, 16],
        model_name="test_model"
    )
    
    assert result['success'] is True
    assert 'test_model' in toolkit.models
    assert result['total_params'] > 0


def test_create_cnn_model():
    """Test creating a CNN model."""
    toolkit = DLToolkit()
    
    result = toolkit.create_cnn_model(
        input_shape=(28, 28, 1),
        output_units=10,
        conv_layers=[16, 32],
        model_name="test_cnn"
    )
    
    assert result['success'] is True
    assert 'test_cnn' in toolkit.models
    assert result['total_params'] > 0


def test_compile_model():
    """Test compiling a model."""
    toolkit = DLToolkit()
    
    # Create model first
    toolkit.create_sequential_model(
        input_shape=(10,),
        output_units=2,
        model_name="test_model"
    )
    
    # Compile it
    result = toolkit.compile_model(
        model_name="test_model",
        optimizer="adam",
        loss="sparse_categorical_crossentropy"
    )
    
    assert result['success'] is True
    assert result['optimizer'] == "adam"


def test_compile_nonexistent_model():
    """Test compiling a non-existent model."""
    toolkit = DLToolkit()
    
    result = toolkit.compile_model(model_name="nonexistent")
    
    assert result['success'] is False
    assert 'error' in result


def test_train_model():
    """Test training a model."""
    toolkit = DLToolkit()
    
    # Create and compile model
    toolkit.create_sequential_model(
        input_shape=(10,),
        output_units=2,
        model_name="test_model"
    )
    toolkit.compile_model(model_name="test_model")
    
    # Train model
    X_train = np.random.randn(50, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, 50)
    
    result = toolkit.train_model(
        model_name="test_model",
        X_train=X_train,
        y_train=y_train,
        epochs=2,
        batch_size=10,
        verbose=0
    )
    
    assert result['success'] is True
    assert result['epochs_completed'] == 2


def test_predict():
    """Test making predictions."""
    toolkit = DLToolkit()
    
    # Create, compile, and train model
    toolkit.create_sequential_model(
        input_shape=(10,),
        output_units=2,
        model_name="test_model"
    )
    toolkit.compile_model(model_name="test_model")
    
    X_train = np.random.randn(50, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, 50)
    toolkit.train_model(
        model_name="test_model",
        X_train=X_train,
        y_train=y_train,
        epochs=1,
        verbose=0
    )
    
    # Make predictions
    X_test = np.random.randn(5, 10).astype(np.float32)
    result = toolkit.predict(model_name="test_model", X=X_test)
    
    assert result['success'] is True
    assert len(result['predictions']) == 5


def test_predict_nonexistent_model():
    """Test prediction with non-existent model."""
    toolkit = DLToolkit()
    
    X = np.random.randn(5, 10).astype(np.float32)
    result = toolkit.predict(model_name="nonexistent", X=X)
    
    assert result['success'] is False
    assert 'error' in result


def test_process():
    """Test general process method."""
    toolkit = DLToolkit()
    
    result = toolkit.process(None, task="inference")
    
    assert result['task'] == "inference"
    assert 'message' in result
