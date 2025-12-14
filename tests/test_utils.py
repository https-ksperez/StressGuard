"""Tests for utility modules."""

import pytest
import os
import tempfile
from utils.config import Config
from utils.logger import setup_logger


def test_config_defaults():
    """Test loading default configuration."""
    config = Config()
    
    assert config.get('model.name') == 'microsoft/DialoGPT-medium'
    assert config.get('chatbot.enable_ml_tools') is True
    assert config.get('chatbot.enable_dl_tools') is True


def test_config_get_with_default():
    """Test getting config value with default."""
    config = Config()
    
    value = config.get('nonexistent.key', 'default_value')
    assert value == 'default_value'


def test_config_set():
    """Test setting config value."""
    config = Config()
    
    config.set('test.key', 'test_value')
    assert config.get('test.key') == 'test_value'


def test_config_save_and_load_json():
    """Test saving and loading JSON config."""
    config = Config()
    config.set('test.value', 123)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        config.save_config(temp_file)
        
        new_config = Config(temp_file)
        assert new_config.get('test.value') == 123
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_logger_setup():
    """Test logger setup."""
    logger = setup_logger('test_logger')
    
    assert logger is not None
    assert logger.name == 'test_logger'
    assert len(logger.handlers) > 0


def test_logger_with_file():
    """Test logger with file output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        temp_file = f.name
    
    try:
        logger = setup_logger('test_logger', log_file=temp_file)
        logger.info('Test message')
        
        assert os.path.exists(temp_file)
        with open(temp_file, 'r') as f:
            content = f.read()
            assert 'Test message' in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
