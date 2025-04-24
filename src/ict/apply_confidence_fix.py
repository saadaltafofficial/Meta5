#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply Confidence Fix

This script applies a direct fix to the ICT trading bot's confidence calculation.
It modifies the generate_ict_signal method in the ICTAnalyzer class to properly
apply additional confidence factors from support/resistance, trend, and volatility.
"""

import logging
import importlib
import types
from src.ict.ict_integration import ICTAnalyzer
from src.ict.confidence_boost import boost_confidence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fixed_generate_ict_signal(self, data, pair=None):
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
        
        # Import necessary functions from ict_mechanics
        from src.ict.ict_mechanics import determine_daily_bias, is_in_killzone, evaluate_ict_setup, evaluate_flexible_duration_setup
        
        # Determine daily bias
        daily_bias = determine_daily_bias(df)
        
        # Check if we're in a killzone
        from datetime import datetime
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
        
        # Apply confidence boost to both setups if they have actionable signals
        # This can potentially turn HOLD signals into BUY or SELL signals
        
        # Boost setup_2022 if it's actionable
        if setup_2022['action'] != 'HOLD':
            # Apply confidence boost
            base_confidence = setup_2022['confidence']
            final_confidence, confluence_factors, additional_confidence = boost_confidence(df, setup_2022['action'], base_confidence)
            setup_2022['confidence'] = final_confidence
            if 'details' not in setup_2022:
                setup_2022['details'] = {}
            setup_2022['details']['additional_confidence'] = additional_confidence
            setup_2022['details']['confluence_factors'] = confluence_factors
            logger.info(f"Boosted setup_2022 confidence: {base_confidence:.4f} -> {final_confidence:.4f} (+{additional_confidence:.4f})")
        
        # Boost setup_flexible if it's actionable
        if setup_flexible['action'] != 'HOLD':
            # Apply confidence boost
            base_confidence = setup_flexible['confidence']
            final_confidence, confluence_factors, additional_confidence = boost_confidence(df, setup_flexible['action'], base_confidence)
            setup_flexible['confidence'] = final_confidence
            if 'details' not in setup_flexible:
                setup_flexible['details'] = {}
            setup_flexible['details']['additional_confidence'] = additional_confidence
            setup_flexible['details']['confluence_factors'] = confluence_factors
            logger.info(f"Boosted setup_flexible confidence: {base_confidence:.4f} -> {final_confidence:.4f} (+{additional_confidence:.4f})")
        
        # Apply confidence boost to both setups regardless of action
        # This can potentially convert HOLD signals to BUY or SELL
        
        # Boost setup_2022 even if it's a HOLD
        base_confidence = setup_2022['confidence']
        final_confidence, confluence_factors, additional_confidence = boost_confidence(df, setup_2022['action'], base_confidence)
        setup_2022['confidence'] = final_confidence
        setup_2022['action'] = 'BUY' if setup_2022['action'] == 'BUY' or (setup_2022['action'] == 'HOLD' and 'Strong Bullish Indicators' in confluence_factors) else \
                         'SELL' if setup_2022['action'] == 'SELL' or (setup_2022['action'] == 'HOLD' and 'Strong Bearish Indicators' in confluence_factors) else \
                         setup_2022['action']
        if 'details' not in setup_2022:
            setup_2022['details'] = {}
        setup_2022['details']['additional_confidence'] = additional_confidence
        setup_2022['details']['confluence_factors'] = confluence_factors
        logger.info(f"Applied confidence boost to setup_2022: {base_confidence:.4f} -> {final_confidence:.4f} (+{additional_confidence:.4f}), Action: {setup_2022['action']}")
        
        # Boost setup_flexible even if it's a HOLD
        base_confidence = setup_flexible['confidence']
        final_confidence, confluence_factors, additional_confidence = boost_confidence(df, setup_flexible['action'], base_confidence)
        setup_flexible['confidence'] = final_confidence
        setup_flexible['action'] = 'BUY' if setup_flexible['action'] == 'BUY' or (setup_flexible['action'] == 'HOLD' and 'Strong Bullish Indicators' in confluence_factors) else \
                           'SELL' if setup_flexible['action'] == 'SELL' or (setup_flexible['action'] == 'HOLD' and 'Strong Bearish Indicators' in confluence_factors) else \
                           setup_flexible['action']
        if 'details' not in setup_flexible:
            setup_flexible['details'] = {}
        setup_flexible['details']['additional_confidence'] = additional_confidence
        setup_flexible['details']['confluence_factors'] = confluence_factors
        logger.info(f"Applied confidence boost to setup_flexible: {base_confidence:.4f} -> {final_confidence:.4f} (+{additional_confidence:.4f}), Action: {setup_flexible['action']}")
        
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
            
            # We've already applied the confidence boost to the setup evaluation
            # Just use the values from the setup
            final_confidence = setup['confidence']
            additional_confidence = setup['details'].get('additional_confidence', 0.0)
            confluence_factors = setup['details'].get('confluence_factors', [])
            
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

def apply_confidence_fix():
    """
    Apply the confidence fix to the ICTAnalyzer class.
    """
    # Monkey patch the generate_ict_signal method in ICTAnalyzer
    ICTAnalyzer.generate_ict_signal = fixed_generate_ict_signal
    
    # Add more visible logging
    logger.info("=== CONFIDENCE BOOST FIX APPLIED ===")
    logger.info("The ICT trading bot will now use enhanced confidence calculation with:")
    logger.info("1. Support/Resistance confidence (weighted 1.5x)")
    logger.info("2. Trend strength confidence (weighted 2.0x)")
    logger.info("3. Volatility measures confidence (weighted 1.75x)")
    logger.info("4. Extra boosts for strong trends and oversold/overbought conditions")
    logger.info("5. Execution threshold lowered to 25%")
    logger.info("=== CONFIDENCE BOOST ACTIVE ===")
    
    return True

# Apply the fix when this module is imported
apply_confidence_fix()
