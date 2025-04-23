#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Loader for MCP Trader

This module handles loading and validating configuration settings from config.json
"""

import os
import json
import logging
import MetaTrader5 as mt5

# Configure logging
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and validates configuration settings for the trading bot"""
    
    def __init__(self, config_path="config.json"):
        """Initialize the config loader
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Configuration file {self.config_path} not found. Using default settings.")
                return self._get_default_config()
                
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            # Validate the loaded config
            self._validate_config(config)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            return self._get_default_config()
    
    def _validate_config(self, config):
        """Validate configuration values
        
        Args:
            config (dict): Configuration dictionary to validate
        """
        # Ensure required sections exist
        required_sections = ['trading', 'indicators', 'notifications', 'data']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing section '{section}' in config. Using defaults for this section.")
                config[section] = self._get_default_config()[section]
        
        # Validate trading parameters
        if 'risk_percent' in config['trading'].get('risk_management', {}):
            risk = config['trading']['risk_management']['risk_percent']
            if risk <= 0 or risk > 5:
                logger.warning(f"Risk percent {risk} is outside recommended range (0-5%). Capping at 2%.")
                config['trading']['risk_management']['risk_percent'] = min(risk, 2.0)
        
        # Convert timeframe strings to MT5 constants
        if 'timeframes' in config['trading']:
            if 'analysis' in config['trading']['timeframes'] and isinstance(config['trading']['timeframes']['analysis'], list):
                config['trading']['timeframes']['analysis_mt5'] = [
                    self._timeframe_str_to_mt5(tf) for tf in config['trading']['timeframes']['analysis']
                ]
            
            if 'primary' in config['trading']['timeframes']:
                config['trading']['timeframes']['primary_mt5'] = self._timeframe_str_to_mt5(
                    config['trading']['timeframes']['primary']
                )
    
    def _timeframe_str_to_mt5(self, timeframe_str):
        """Convert timeframe string to MT5 constant
        
        Args:
            timeframe_str (str): Timeframe string (e.g., 'M1', 'H1', 'D1')
            
        Returns:
            int: MT5 timeframe constant
        """
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        return timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_H1)
    
    def _get_default_config(self):
        """Get default configuration
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            "trading": {
                "currency_pairs": ["EURUSD", "GBPUSD"],
                "timeframes": {
                    "analysis": ["M15", "H1", "H4", "D1"],
                    "primary": "H1",
                    "analysis_mt5": [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1],
                    "primary_mt5": mt5.TIMEFRAME_H1
                },
                "risk_management": {
                    "risk_percent": 1.5,
                    "max_balance_percent": 10.0,
                    "take_profit_ratios": [1.5, 2.75, 4.75]
                },
                "auto_trading": True,
                "min_confidence": 0.4
            },
            "indicators": {
                "moving_averages": {
                    "fast_period": 9,
                    "slow_period": 21,
                    "trend_period": 50
                },
                "ict_model": {
                    "order_blocks": True,
                    "fair_value_gaps": True,
                    "liquidity_levels": True
                }
            },
            "notifications": {
                "telegram_enabled": True,
                "signal_frequency": "minute"
            },
            "data": {
                "bars_to_analyze": 100,
                "economic_calendar_enabled": True,
                "economic_event_hours_window": 24
            }
        }
    
    def get_config(self):
        """Get the loaded configuration
        
        Returns:
            dict: Configuration dictionary
        """
        return self.config
    
    def save_config(self, config=None):
        """Save configuration to JSON file
        
        Args:
            config (dict, optional): Configuration to save. If None, saves the current config.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if config is None:
                config = self.config
            
            # Remove MT5 constants before saving
            if 'trading' in config and 'timeframes' in config['trading']:
                save_config = json.loads(json.dumps(config))  # Deep copy
                if 'analysis_mt5' in save_config['trading']['timeframes']:
                    del save_config['trading']['timeframes']['analysis_mt5']
                if 'primary_mt5' in save_config['trading']['timeframes']:
                    del save_config['trading']['timeframes']['primary_mt5']
            else:
                save_config = config
            
            with open(self.config_path, 'w') as f:
                json.dump(save_config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

# Singleton instance
_config_instance = None

def get_config(config_path="config.json"):
    """Get the configuration singleton instance
    
    Args:
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    global _config_instance
    if _config_instance is None:
        # If path doesn't exist, try to find it in the config directory
        if not os.path.exists(config_path):
            # Try to find the config file in the config directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            config_dir_path = os.path.join(project_root, 'config', os.path.basename(config_path))
            if os.path.exists(config_dir_path):
                config_path = config_dir_path
                logger.info(f"Using config file from config directory: {config_path}")
        _config_instance = ConfigLoader(config_path)
    return _config_instance.get_config()

def save_config(config=None, config_path="config.json"):
    """Save configuration to file
    
    Args:
        config (dict, optional): Configuration to save
        config_path (str, optional): Path to the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _config_instance
    if _config_instance is None:
        # If path doesn't exist, try to find it in the config directory
        if not os.path.exists(config_path):
            # Try to find the config file in the config directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            config_dir_path = os.path.join(project_root, 'config', os.path.basename(config_path))
            if os.path.exists(config_dir_path):
                config_path = config_dir_path
                logger.info(f"Using config file from config directory: {config_path}")
        _config_instance = ConfigLoader(config_path)
    return _config_instance.save_config(config)
