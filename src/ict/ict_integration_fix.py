#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICT Integration Fix

This module provides a fixed version of the generate_ict_signal method
to properly apply additional confidence factors.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ict.ict_mechanics import *
from src.ict.support_resistance import detect_support_resistance, calculate_sr_confidence
from src.ict.trend_indicators import calculate_adx, calculate_ma_trend, calculate_trend_confidence
from src.ict.volatility_measures import calculate_atr, calculate_bollinger_bands, calculate_volatility_confidence
from src.ict.confidence_boost import boost_confidence

# Configure logging
logger = logging.getLogger(__name__)

# This is a fixed version of the generate_ict_signal method
def generate_ict_signal_fixed(self, data, pair=None):
    """
    Generate a trading signal based on ICT analysis with additional confluence factors
    
    Args:
        data (pd.DataFrame): Price data with OHLC values
        pair (str, optional): Currency pair being analyzed
        
    Returns:
        dict: Trading signal with action, confidence, and details
    """
    try:
        # First analyze the price data
        df, analysis = self.analyze_price_data(data, pair)
        
        # Calculate additional confluence factors
        df, additional_factors = self.calculate_additional_factors(df)
        
        # Log the additional factors for debugging
        logger.info(f"Additional factors calculated: SR={additional_factors['support_resistance'].get('confidence', {})}, "
                  f"Trend Buy={additional_factors['trend'].get('buy_confidence', 0)}, Trend Sell={additional_factors['trend'].get('sell_confidence', 0)}, "
                  f"Vol Buy={additional_factors['volatility'].get('buy_confidence', 0)}, Vol Sell={additional_factors['volatility'].get('sell_confidence', 0)}")
        
        # Determine daily bias
        daily_bias = determine_daily_bias(df)
        
        # Check if we're in a killzone
        current_time = datetime.now()
        in_killzone, killzone_name = is_in_killzone(current_time)
        
        # Set default timeframe if not in analysis
        timeframe = 'H4'  # Default to H4
        if 'timeframe' in analysis:
            timeframe = analysis['timeframe']
        
        # Traditional ICT 2022 setup evaluation
        setup_2022 = evaluate_ict_setup(df, daily_bias)
        
        # Flexible duration ICT setup (works for both short-term and long-term)
        try:
            setup_flexible = evaluate_flexible_duration_setup(df, timeframe=timeframe, daily_bias=daily_bias, current_time=current_time)
        except Exception as e:
            logger.warning(f"Error in flexible duration setup: {e}")
            # Provide a default setup if the function fails
            setup_flexible = {
                'action': 'HOLD',
                'confidence': 0,
                'setup_type': 'Default',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'details': {
                    'daily_bias': daily_bias,
                    'in_killzone': False,
                    'killzone_name': None,
                    'premium': False
                }
            }
        
        # Choose the setup with the highest confidence
        if setup_flexible['confidence'] > setup_2022['confidence']:
            setup = setup_flexible
        else:
            setup = setup_2022
            
        # Add additional factors to the setup details
        if 'details' not in setup:
            setup['details'] = {}
        setup['details']['additional_factors'] = additional_factors
    
        # Create the signal with proper error handling
        try:
            # Ensure there's at least some confidence value to contribute to the overall calculation
            # This is a temporary fix to ensure ICT contributes to confidence until the root cause is fixed
            min_confidence = 0.15  # Set a minimum confidence value for ICT signals
            
            # Get base confidence from the setup
            base_confidence = max(setup['confidence'], min_confidence) if setup['action'] != 'HOLD' else setup['confidence']
            
            # Use our confidence boost module to calculate additional confidence
            final_confidence, confluence_factors, additional_confidence = boost_confidence(df, setup['action'], base_confidence)
            
            signal = {
                'action': setup['action'],
                'confidence': final_confidence,
                'reason': f"ICT {setup['setup_type']} setup",
                'details': {
                    'bias': setup['details'].get('daily_bias', 'NEUTRAL'),
                    'in_killzone': in_killzone,
                    'killzone_name': killzone_name,
                    'entry': setup['entry'],
                    'stop_loss': setup['stop_loss'],
                    'take_profit': setup['take_profit'],
                    'market_structure': analysis.get('market_structure', {}),
                    'order_blocks': analysis.get('order_blocks', {}),
                    'fair_value_gaps': analysis.get('fair_value_gaps', {}),
                    'liquidity': analysis.get('liquidity', {}),
                    'ote': analysis.get('ote', {}),
                    'confluence_factors': confluence_factors,
                    'base_confidence': base_confidence,
                    'additional_confidence': additional_confidence
                }
            }
        except KeyError as e:
            # Handle missing keys like 'bullish_mss'
            logger.warning(f"KeyError in ICT signal generation: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': f"Missing data: {e}"}
            
        # Add pair information if provided
        if pair:
            signal['pair'] = pair
            
        return signal
    except Exception as e:
        logger.error(f"Error generating ICT signal: {e}")
        return {'action': 'HOLD', 'confidence': 0, 'reason': f"Error: {e}"}

# Instructions for applying this fix:
# 1. Replace the existing generate_ict_signal method in ICTAnalyzer class with this fixed version
# 2. Make sure to import the boost_confidence function at the top of the file
