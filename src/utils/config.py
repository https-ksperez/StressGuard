"""Configuration management utilities."""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for StressGuard."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_data = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.load_defaults()
            
    def load_defaults(self):
        """Load default configuration."""
        self.config_data = {
            "model": {
                "name": "microsoft/DialoGPT-medium",
                "max_length": 1000,
                "temperature": 0.7,
            },
            "chatbot": {
                "enable_ml_tools": True,
                "enable_dl_tools": True,
                "max_conversation_history": 20,
            },
            "ml": {
                "default_test_size": 0.2,
                "random_state": 42,
            },
            "dl": {
                "default_epochs": 10,
                "default_batch_size": 32,
            },
        }
        
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file (YAML or JSON)
        """
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                self.config_data = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
                
    def save_config(self, output_path: str):
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save config file
        """
        with open(output_path, 'w') as f:
            if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(self.config_data, f, default_flow_style=False)
            elif output_path.endswith('.json'):
                json.dump(self.config_data, f, indent=2)
            else:
                raise ValueError("Config file must be YAML or JSON")
                
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
