#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICT Integration Module

This module integrates the ICT mechanics with the StandaloneTrader class
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the same directory
from src.ict.ict_mechanics import (
    identify_market_structure,
    identify_order_blocks,
    identify_fair_value_gaps,
    identify_liquidity_pools,
    calculate_ote_levels,
    is_in_killzone,
    determine_daily_bias,
    evaluate_ict_setup,
    evaluate_flexible_duration_setup,
    identify_relative_equal_levels,
    create_pd_array
)

# Import new confluence factors
from src.ict.support_resistance import detect_support_resistance, calculate_sr_confidence
from src.ict.trend_indicators import calculate_adx, calculate_ma_trend, calculate_trend_confidence
from src.ict.volatility_measures import calculate_atr, calculate_bollinger_bands, calculate_volatility_confidence

# Configure logging
logger = logging.getLogger(__name__)

class ICTAnalyzer:
    """ICT Analysis class to enhance the StandaloneTrader with advanced ICT mechanics"""
    
    def __init__(self, config=None):
        """Initialize the ICT Analyzer
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
    
    def _determine_last_mss_type(self, df):
        """Safely determine the last Market Structure Shift type
        
        Args:
            df (pd.DataFrame): DataFrame with market structure data
            
        Returns:
            str: 'BULLISH', 'BEARISH', or 'NONE'
        """
        try:
            # Check if there are any recent bullish MSS
            recent_bullish = df['bullish_mss'].iloc[-20:].any()
            
            # Check if there are any recent bearish MSS
            recent_bearish = df['bearish_mss'].iloc[-20:].any()
            
            if not recent_bullish and not recent_bearish:
                return 'NONE'
            
            # If we have both types, determine which one is more recent
            if recent_bullish and recent_bearish:
                # Get indices where MSS occurred
                bullish_indices = df.index[df['bullish_mss']]
                bearish_indices = df.index[df['bearish_mss']]
                
                # Make sure we have at least one of each
                if len(bullish_indices) > 0 and len(bearish_indices) > 0:
                    # Return the type of the most recent MSS
                    return 'BULLISH' if bullish_indices[-1] > bearish_indices[-1] else 'BEARISH'
                elif len(bullish_indices) > 0:
                    return 'BULLISH'
                else:
                    return 'BEARISH'
            
            # If we only have one type
            return 'BULLISH' if recent_bullish else 'BEARISH'
            
        except Exception as e:
            logger.warning(f"Error determining last MSS type: {e}")
            return 'NONE'
    
    def analyze_price_data(self, data, pair=None):
        """Perform comprehensive ICT analysis on price data
        
        Args:
            data (pd.DataFrame): Price data with OHLC values
            pair (str, optional): Currency pair being analyzed for pair-specific settings
            
        Returns:
            tuple: (pd.DataFrame, dict) - Enhanced data and ICT analysis results
        """
        try:
            # Check if data is None or empty
            if data is None or len(data) == 0:
                logger.warning("Cannot analyze empty or None data")
                # Return empty DataFrame with required columns
                empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'time'])
                empty_df['bullish_mss'] = False
                empty_df['bearish_mss'] = False
                empty_df['bullish_ob'] = False
                empty_df['bearish_ob'] = False
                empty_df['bullish_fvg'] = False
                empty_df['bearish_fvg'] = False
                empty_df['buy_liquidity'] = False
                empty_df['sell_liquidity'] = False
                empty_df['bullish_ote'] = False
                empty_df['bearish_ote'] = False
                return empty_df
            
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            try:
                # Identify market structure with error handling
                try:
                    df = identify_market_structure(df)
                except Exception as e:
                    logger.warning(f"Error identifying market structure: {e}")
                    # Initialize market structure columns to avoid errors
                    required_ms_columns = ['bullish_mss', 'bearish_mss', 'swing_high', 'swing_low', 'higher_high', 'higher_low', 'lower_high', 'lower_low']
                    for col in required_ms_columns:
                        if col not in df.columns:
                            df[col] = False
                
                # Identify relative equal levels (ICT 2024 concept)
                try:
                    # Get pair-specific settings if available
                    pair_specific_settings = {}
                    if pair and self.config and 'indicators' in self.config:
                        if 'ict_model' in self.config['indicators'] and 'pair_specific' in self.config['indicators']['ict_model']:
                            if pair in self.config['indicators']['ict_model']['pair_specific']:
                                pair_specific_settings = self.config['indicators']['ict_model']['pair_specific'][pair]
                    
                    # Use pair-specific tolerance if available, otherwise use default
                    tolerance = pair_specific_settings.get('rel_equal_tolerance', 0.0002)
                    df = identify_relative_equal_levels(df, lookback=10, tolerance=tolerance)
                except Exception as e:
                    logger.warning(f"Error identifying relative equal levels: {e}")
                    # Initialize relative equal levels columns to avoid errors
                    df['rel_equal_high'] = False
                    df['rel_equal_low'] = False
                    df['rel_equal_high_level'] = np.nan
                    df['rel_equal_low_level'] = np.nan
                    df['liquidity_grabbed_high'] = False
                    df['liquidity_grabbed_low'] = False
            except Exception as e:
                logger.warning(f"Error identifying market structure: {e}")
                # Initialize market structure columns to avoid errors
                if 'bullish_mss' not in df.columns:
                    df['bullish_mss'] = False
                if 'bearish_mss' not in df.columns:
                    df['bearish_mss'] = False
            
            try:
                # Identify order blocks
                df = identify_order_blocks(df)
            except Exception as e:
                logger.warning(f"Error identifying order blocks: {e}")
                # Initialize order block columns to avoid errors
                if 'bullish_ob' not in df.columns:
                    df['bullish_ob'] = False
                if 'bearish_ob' not in df.columns:
                    df['bearish_ob'] = False
            
            try:
                # Identify fair value gaps
                df = identify_fair_value_gaps(df)
            except Exception as e:
                logger.warning(f"Error identifying fair value gaps: {e}")
                # Initialize FVG columns to avoid errors
                if 'bullish_fvg' not in df.columns:
                    df['bullish_fvg'] = False
                if 'bearish_fvg' not in df.columns:
                    df['bearish_fvg'] = False
            
            try:
                # Create our own liquidity pools detection since the original function has parameter issues
                # Initialize liquidity pool columns
                df['buy_liquidity'] = False
                df['sell_liquidity'] = False
                
                # Add liquidity level columns that are missing
                df['buy_liquidity_level'] = np.nan
                df['sell_liquidity_level'] = np.nan
                
                # Create ICT-based liquidity detection logic
                # Buy liquidity: When price makes a lower low followed by a higher low (liquidity sweep)
                for i in range(3, len(df)):
                    if df['low'].iloc[i-2] > df['low'].iloc[i-1] and df['low'].iloc[i-1] < df['low'].iloc[i]:
                        df.loc[df.index[i-1], 'buy_liquidity'] = True
                        df.loc[df.index[i-1], 'buy_liquidity_level'] = df['low'].iloc[i-1]
                
                # Sell liquidity: When price makes a higher high followed by a lower high (liquidity sweep)
                for i in range(3, len(df)):
                    if df['high'].iloc[i-2] < df['high'].iloc[i-1] and df['high'].iloc[i-1] > df['high'].iloc[i]:
                        df.loc[df.index[i-1], 'sell_liquidity'] = True
                        df.loc[df.index[i-1], 'sell_liquidity_level'] = df['high'].iloc[i-1]
                        
                # Add liquidity at swing highs and lows (another ICT concept)
                for i in range(5, len(df)-5):
                    # Swing high: 5 bars with lower highs on each side
                    if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, 6)) and \
                       all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, 6)):
                        df.loc[df.index[i], 'sell_liquidity'] = True
                        df.loc[df.index[i], 'sell_liquidity_level'] = df['high'].iloc[i]
                    
                    # Swing low: 5 bars with higher lows on each side
                    if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, 6)) and \
                       all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, 6)):
                        df.loc[df.index[i], 'buy_liquidity'] = True
                        df.loc[df.index[i], 'buy_liquidity_level'] = df['low'].iloc[i]
            except Exception as e:
                logger.warning(f"Error identifying liquidity pools: {e}")
                # Initialize liquidity pool columns to avoid errors
                if 'buy_liquidity' not in df.columns:
                    df['buy_liquidity'] = False
                if 'sell_liquidity' not in df.columns:
                    df['sell_liquidity'] = False
            
            try:
                # Calculate OTE levels
                df = calculate_ote_levels(df)
            except Exception as e:
                logger.warning(f"Error calculating OTE levels: {e}")
                # Initialize OTE columns to avoid errors
                if 'bullish_ote' not in df.columns:
                    df['bullish_ote'] = False
                if 'bearish_ote' not in df.columns:
                    df['bearish_ote'] = False
            
            # Step 6: Determine daily bias
            daily_bias = determine_daily_bias(df)
            
            # Step 7: Check if we're in a killzone
            in_killzone, killzone_name = is_in_killzone(datetime.now())
            
            # Step 8: Evaluate for ICT trade setups using both 2022 and 2024 methodologies
            # Traditional ICT 2022 setup evaluation
            setup_2022 = evaluate_ict_setup(df, daily_bias)
            
            # ICT 2024 PD-array approach with current time for Pakistan night hours optimization
            pd_array = create_pd_array(df, bias=daily_bias, lookback=20, current_time=datetime.now())
            
            # Merge the two approaches, prioritizing 2024 if it has entries
            if pd_array['best_entry'] is not None:
                # Use the PD-array (2024 approach) as it's more precise
                setup = {
                    'action': 'BUY' if pd_array['bias'] == 'BULLISH' else 'SELL',
                    'confidence': pd_array['best_entry']['confidence'],
                    'setup_type': f"ICT 2024 {pd_array['best_entry']['type']}",
                    'entry': pd_array['best_entry']['level'],
                    'stop_loss': pd_array['stop_loss'],
                    'take_profit': pd_array['take_profit'],
                    'details': {
                        'pd_array': pd_array,
                        'daily_bias': daily_bias,
                        'in_killzone': in_killzone,
                        'killzone_name': killzone_name
                    }
                }
            else:
                # Fall back to 2022 approach if no PD-array entries found
                setup = setup_2022
            
            # Prepare analysis results
            analysis = {
                'daily_bias': daily_bias,
                'in_killzone': in_killzone,
                'killzone_name': killzone_name,
                'setup': setup,
                'market_structure': {
                    'has_bullish_mss': df['bullish_mss'].iloc[-30:].any(),
                    'has_bearish_mss': df['bearish_mss'].iloc[-30:].any(),
                    'last_mss_type': self._determine_last_mss_type(df)
                },
                'order_blocks': {
                    'has_bullish_ob': df['bullish_ob'].iloc[-30:].any(),
                    'has_bearish_ob': df['bearish_ob'].iloc[-30:].any(),
                    'has_bullish_breaker': df['bullish_bb'].iloc[-30:].any() if 'bullish_bb' in df.columns else False,
                    'has_bearish_breaker': df['bearish_bb'].iloc[-30:].any() if 'bearish_bb' in df.columns else False
                },
                'fair_value_gaps': {
                    'has_bullish_fvg': df['bullish_fvg'].iloc[-30:].any(),
                    'has_bearish_fvg': df['bearish_fvg'].iloc[-30:].any()
                },
                'liquidity': {
                    'has_buy_liquidity': df['buy_liquidity'].iloc[-30:].any(),
                    'has_sell_liquidity': df['sell_liquidity'].iloc[-30:].any(),
                    'has_rel_equal_high': df['rel_equal_high'].iloc[-20:].any() if 'rel_equal_high' in df.columns else False,
                    'has_rel_equal_low': df['rel_equal_low'].iloc[-20:].any() if 'rel_equal_low' in df.columns else False,
                    'has_liquidity_grabbed_high': df['liquidity_grabbed_high'].iloc[-10:].any() if 'liquidity_grabbed_high' in df.columns else False,
                    'has_liquidity_grabbed_low': df['liquidity_grabbed_low'].iloc[-10:].any() if 'liquidity_grabbed_low' in df.columns else False
                },
                'ote': {
                    'has_bullish_ote': df['bullish_ote'].iloc[-30:].any() if 'bullish_ote' in df.columns else False,
                    'has_bearish_ote': df['bearish_ote'].iloc[-30:].any() if 'bearish_ote' in df.columns else False
                }
            }
            
            return df, analysis
            
        except Exception as e:
            logger.error(f"Error in ICT analysis: {e}")
            return data, {'error': str(e)}
    
    def calculate_additional_factors(self, df):
        """Calculate additional confluence factors: support/resistance, trend, and volatility
        
        Args:
            df (pd.DataFrame): Price data with OHLC values
            
        Returns:
            tuple: (DataFrame with additional factors, factors dictionary)
        """
        try:
            # Create a copy to avoid modifying the original
            data = df.copy()
            
            # Initialize factors dictionary
            factors = {
                'support_resistance': {},
                'trend': {},
                'volatility': {}
            }
            
            # 1. Calculate support and resistance levels
            sr_levels = detect_support_resistance(data, lookback=100, strength_threshold=2)
            factors['support_resistance'] = sr_levels
            
            # 2. Calculate trend indicators
            data = calculate_adx(data, period=14)
            data = calculate_ma_trend(data, fast_period=9, slow_period=21, trend_period=50)
            trend_conf = calculate_trend_confidence(data, lookback=20)
            factors['trend'] = {
                'adx': data['adx'].iloc[-1] if 'adx' in data.columns else None,
                'plus_di': data['+di'].iloc[-1] if '+di' in data.columns else None,
                'minus_di': data['-di'].iloc[-1] if '-di' in data.columns else None,
                'buy_confidence': trend_conf['buy_confidence'],
                'sell_confidence': trend_conf['sell_confidence']
            }
            
            # 3. Calculate volatility measures
            data = calculate_atr(data, period=14)
            data = calculate_bollinger_bands(data, period=20, std_dev=2)
            vol_conf = calculate_volatility_confidence(data, lookback=20)
            factors['volatility'] = {
                'atr': data['atr'].iloc[-1] if 'atr' in data.columns else None,
                'atr_percent': data['atr_percent'].iloc[-1] if 'atr_percent' in data.columns else None,
                'bb_bandwidth': data['bb_bandwidth'].iloc[-1] if 'bb_bandwidth' in data.columns else None,
                'bb_percent_b': data['bb_percent_b'].iloc[-1] if 'bb_percent_b' in data.columns else None,
                'buy_confidence': vol_conf['buy_confidence'],
                'sell_confidence': vol_conf['sell_confidence']
            }
            
            # Calculate SR confidence based on current price
            current_price = data['close'].iloc[-1]
            sr_conf = calculate_sr_confidence(current_price, sr_levels)
            factors['support_resistance']['confidence'] = sr_conf
            
            return data, factors
            
        except Exception as e:
            logger.error(f"Error calculating additional factors: {e}")
            return df, {
                'support_resistance': {},
                'trend': {'buy_confidence': 0.0, 'sell_confidence': 0.0},
                'volatility': {'buy_confidence': 0.0, 'sell_confidence': 0.0}
            }
    
    def generate_ict_signal(self, data, pair=None):
        """Generate a trading signal based on ICT analysis with additional confluence factors
        
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
                        'confluence_factors': []
                    }
                }
            
            # Ensure FVG columns exist before PD-array creation
            for col in ['bullish_fvg', 'bearish_fvg', 'bullish_fvg_top', 'bullish_fvg_bottom', 'bearish_fvg_top', 'bearish_fvg_bottom']:
                if col not in df.columns:
                    if col.endswith('_top') or col.endswith('_bottom'):
                        df[col] = 0.0
                    else:
                        df[col] = False
            
            # ICT 2024 PD-array approach
            try:
                pd_array = create_pd_array(df, bias=daily_bias, lookback=20, current_time=current_time)
            except Exception as e:
                logger.warning(f"Error in PD-array creation: {e}")
                # Provide a default PD-array if the function fails
                pd_array = {
                    'bias': daily_bias,
                    'best_entry': None,
                    'stop_loss': 0,
                    'take_profit': 0
                }
            
            # Choose the best setup based on confidence
            best_setup = None
            best_confidence = 0
            setup_source = 'traditional'
            
            # Check traditional ICT 2022 setup
            if setup_2022['action'] != 'HOLD' and setup_2022['confidence'] > best_confidence:
                best_confidence = setup_2022['confidence']
                setup_source = 'traditional'
            
            # Check flexible duration setup
            if setup_flexible['action'] != 'HOLD' and setup_flexible['confidence'] > best_confidence:
                best_confidence = setup_flexible['confidence']
                setup_source = 'flexible'
            
            # Use the best setup based on source
            if setup_source == 'pd_array':
                # Create setup from PD-array
                setup = {
                    'action': 'BUY' if pd_array['bias'] == 'BULLISH' else 'SELL',
                    'confidence': pd_array['best_entry']['confidence'],
                    'setup_type': f"ICT 2024 {pd_array['best_entry']['type']}",
                    'entry': pd_array['best_entry']['level'],
                    'stop_loss': pd_array['stop_loss'],
                    'take_profit': pd_array['take_profit'],
                    'details': {
                        'daily_bias': daily_bias,
                        'in_killzone': in_killzone,
                        'killzone_name': killzone_name,
                        'additional_factors': additional_factors
                    }
                }
            elif setup_source == 'flexible':
                # Use the flexible duration setup
                setup = setup_flexible
                setup['details']['additional_factors'] = additional_factors
            else:
                # Use the traditional ICT 2022 setup
                setup = setup_2022
                setup['details']['additional_factors'] = additional_factors
            
            # Create the signal with proper error handling
            try:
                # Ensure there's at least some confidence value to contribute to the overall calculation
                # This is a temporary fix to ensure ICT contributes to confidence until the root cause is fixed
                min_confidence = 0.15  # Set a minimum confidence value for ICT signals
                
                # Get base confidence from the setup
                base_confidence = max(setup['confidence'], min_confidence) if setup['action'] != 'HOLD' else setup['confidence']
                
                # Initialize additional confidence boost and confluence factors
                additional_confidence = 0.0
                confluence_factors = []
                
                # Only calculate additional confidence for actionable signals
                if setup['action'] != 'HOLD':
                    # Add support/resistance confidence
                    if 'additional_factors' in setup['details']:
                        factors = setup['details']['additional_factors']
                        
                        # 1. Support/Resistance confidence
                        if 'support_resistance' in factors and 'confidence' in factors['support_resistance']:
                            sr_conf = factors['support_resistance'].get('confidence', {})
                            if setup['action'] == 'BUY' and sr_conf.get('buy_confidence', 0) > 0:
                                additional_confidence += sr_conf.get('buy_confidence', 0)
                                confluence_factors.append("Near Support Level")
                            elif setup['action'] == 'SELL' and sr_conf.get('sell_confidence', 0) > 0:
                                additional_confidence += sr_conf.get('sell_confidence', 0)
                                confluence_factors.append("Near Resistance Level")
                        
                        # 2. Trend confidence
                        if 'trend' in factors:
                            if setup['action'] == 'BUY' and factors['trend'].get('buy_confidence', 0) > 0:
                                additional_confidence += factors['trend']['buy_confidence']
                                # Check if ADX indicates strong trend
                                if factors['trend'].get('adx') and factors['trend'].get('adx') > 25:
                                    confluence_factors.append("Strong Bullish Trend")
                                else:
                                    confluence_factors.append("Bullish Trend Alignment")
                            elif setup['action'] == 'SELL' and factors['trend'].get('sell_confidence', 0) > 0:
                                additional_confidence += factors['trend']['sell_confidence']
                                # Check if ADX indicates strong trend
                                if factors['trend'].get('adx') and factors['trend'].get('adx') > 25:
                                    confluence_factors.append("Strong Bearish Trend")
                                else:
                                    confluence_factors.append("Bearish Trend Alignment")
                        
                        # 3. Volatility confidence
                        if 'volatility' in factors:
                            if setup['action'] == 'BUY' and factors['volatility'].get('buy_confidence', 0) > 0:
                                additional_confidence += factors['volatility']['buy_confidence']
                                # Check if price is near lower Bollinger Band
                                if factors['volatility'].get('bb_percent_b') and factors['volatility'].get('bb_percent_b') < 0.2:
                                    confluence_factors.append("Oversold Condition")
                                else:
                                    confluence_factors.append("Favorable Volatility")
                            elif setup['action'] == 'SELL' and factors['volatility'].get('sell_confidence', 0) > 0:
                                additional_confidence += factors['volatility']['sell_confidence']
                                # Check if price is near upper Bollinger Band
                                if factors['volatility'].get('bb_percent_b') and factors['volatility'].get('bb_percent_b') > 0.8:
                                    confluence_factors.append("Overbought Condition")
                                else:
                                    confluence_factors.append("Favorable Volatility")
                
                # Calculate final confidence (cap at 0.95 to avoid overconfidence)
                final_confidence = min(base_confidence + additional_confidence, 0.95)
                
                # Log the confidence calculation for debugging
                logger.info(f"Confidence calculation: {base_confidence} (base) + {additional_confidence} (additional) = {final_confidence} (final)")
                
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
    
    def analyze_price_data(self, df, pair=None):
        """Analyze price data using ICT methodology
        
        Args:
            df (pd.DataFrame): Price data with OHLC values
            pair (str, optional): Currency pair being analyzed
            
        Returns:
            tuple: (DataFrame with ICT analysis, analysis dictionary)
        """
        data = df.copy()  # Avoid modifying original DataFrame
        # Initialize default analysis result in case of errors
        analysis = {
            'market_structure': {
                'has_bullish_mss': False,
                'has_bearish_mss': False,
                'last_mss_type': 'NONE'
            },
            'order_blocks': {
                'has_bullish_ob': False,
                'has_bearish_ob': False,
                'has_bullish_breaker': False,
                'has_bearish_breaker': False
            },
            'fair_value_gaps': {
                'has_bullish_fvg': False,
                'has_bearish_fvg': False
            },
            'liquidity': {
                'has_bullish_liquidity': False,
                'has_bearish_liquidity': False
            },
            'ote': {
                'has_bullish_ote': False,
                'has_bearish_ote': False
            },
            'timeframe': 'H4'  # Default timeframe
        }
        
        try:
            # Identify market structure shifts
            data = identify_market_structure(data)
            
            # Ensure market structure columns exist
            if 'bullish_mss' not in data.columns:
                data['bullish_mss'] = False
            if 'bearish_mss' not in data.columns:
                data['bearish_mss'] = False
            
            # Update market structure analysis
            analysis['market_structure']['has_bullish_mss'] = data['bullish_mss'].iloc[-30:].any()
            analysis['market_structure']['has_bearish_mss'] = data['bearish_mss'].iloc[-30:].any()
            analysis['market_structure']['last_mss_type'] = self._determine_last_mss_type(data)
            
            # Identify order blocks
            data = identify_order_blocks(data)
            
            # Ensure order block columns exist
            if 'bullish_ob' not in data.columns:
                data['bullish_ob'] = False
            if 'bearish_ob' not in data.columns:
                data['bearish_ob'] = False
            
            # Update order block analysis
            analysis['order_blocks']['has_bullish_ob'] = data['bullish_ob'].iloc[-30:].any()
            analysis['order_blocks']['has_bearish_ob'] = data['bearish_ob'].iloc[-30:].any()
            
            # Identify fair value gaps
            data = identify_fair_value_gaps(data)
            
            # Ensure FVG columns exist
            if 'bullish_fvg' not in data.columns:
                data['bullish_fvg'] = False
            if 'bearish_fvg' not in data.columns:
                data['bearish_fvg'] = False
            
            # Update FVG analysis
            analysis['fair_value_gaps']['has_bullish_fvg'] = data['bullish_fvg'].iloc[-30:].any()
            analysis['fair_value_gaps']['has_bearish_fvg'] = data['bearish_fvg'].iloc[-30:].any()
            
            # Identify liquidity pools
            data = identify_liquidity_pools(data)
            
            # Ensure liquidity columns exist
            if 'bullish_liquidity' not in data.columns:
                data['bullish_liquidity'] = False
            if 'bearish_liquidity' not in data.columns:
                data['bearish_liquidity'] = False
            
            # Update liquidity analysis
            analysis['liquidity']['has_bullish_liquidity'] = data['bullish_liquidity'].iloc[-30:].any()
            analysis['liquidity']['has_bearish_liquidity'] = data['bearish_liquidity'].iloc[-30:].any()
            
            # Calculate OTE levels
            try:
                data = calculate_ote_levels(data)
                
                # Ensure OTE columns exist
                if 'bullish_ote' not in data.columns:
                    data['bullish_ote'] = False
                if 'bearish_ote' not in data.columns:
                    data['bearish_ote'] = False
                
                # Update OTE analysis
                analysis['ote']['has_bullish_ote'] = data['bullish_ote'].iloc[-30:].any()
                analysis['ote']['has_bearish_ote'] = data['bearish_ote'].iloc[-30:].any()
            except Exception as e:
                logger.warning(f"Error calculating OTE levels: {e}")
            
            return data, analysis
            
        except Exception as e:
            logger.warning(f"Error in ICT analysis: {e}")
            return data, analysis



            # Choose the best setup based on confidence
            best_setup = None
            best_confidence = 0
            setup_source = 'none'
            
            # Check PD-array (ICT 2024 approach)
            if pd_array['best_entry'] is not None:
                pd_confidence = pd_array['best_entry']['confidence']
                if pd_confidence > best_confidence:
                    best_confidence = pd_confidence
                    setup_source = 'pd_array'
            
            # Check flexible duration setup
            if setup_flexible['action'] != 'HOLD' and setup_flexible['confidence'] > best_confidence:
                best_confidence = setup_flexible['confidence']
                setup_source = 'flexible'
            
            # Check traditional ICT 2022 setup
            if setup_2022['action'] != 'HOLD' and setup_2022['confidence'] > best_confidence:
                # Calculate additional confluence factors
                try:
                    df, additional_factors = self.analyze_price_data(df)
                    
                    # Log the additional factors for debugging
                    logger.debug(f"Additional factors calculated: {additional_factors}")
                except Exception as e:
                    logger.error(f"Error calculating additional factors: {e}")
                    additional_factors = {
                        'support_resistance': {},
                        'trend': {'buy_confidence': 0.0, 'sell_confidence': 0.0},
                        'volatility': {'buy_confidence': 0.0, 'sell_confidence': 0.0}
                    }
                
                best_confidence = setup_2022['confidence']
                setup_source = 'traditional'
            
            # Use the best setup based on source
            if setup_source == 'pd_array':
                # Create setup from PD-array
                setup = {
                    'action': 'BUY' if pd_array['bias'] == 'BULLISH' else 'SELL',
                    'confidence': pd_array['best_entry']['confidence'],
                    'setup_type': f"ICT 2024 {pd_array['best_entry']['type']}",
                    'entry': pd_array['best_entry']['level'],
                    'stop_loss': pd_array['stop_loss'],
                    'take_profit': pd_array['take_profit'],
                    'details': {
                        'daily_bias': daily_bias,
                        'in_killzone': in_killzone,
                        'killzone_name': killzone_name,
                        'additional_factors': additional_factors
                    }
                }
            elif setup_source == 'flexible':
                # Use the flexible duration setup
                setup = setup_flexible
            else:
                # Use the traditional ICT 2022 setup
                setup = setup_2022
            
            # Create the signal with proper error handling
            try:
                # Ensure there's at least some confidence value to contribute to the overall calculation
                # This is a temporary fix to ensure ICT contributes to confidence until the root cause is fixed
                min_confidence = 0.15  # Set a minimum confidence value for ICT signals
                
                # Get base confidence from the setup
                base_confidence = max(setup['confidence'], min_confidence) if setup['action'] != 'HOLD' else setup['confidence']
                
                # Initialize additional confidence boost and confluence factors
                additional_confidence = 0.0
                confluence_factors = []
                
                # Only calculate additional confidence for actionable signals
                if setup['action'] != 'HOLD':
                    # Add support/resistance confidence
                    if 'additional_factors' in setup['details']:
                        factors = setup['details']['additional_factors']
                        
                        # 1. Support/Resistance confidence
                        if 'support_resistance' in factors and 'confidence' in factors['support_resistance']:
                            sr_conf = factors['support_resistance'].get('confidence', {})
                            if setup['action'] == 'BUY' and sr_conf.get('buy_confidence', 0) > 0:
                                additional_confidence += sr_conf.get('buy_confidence', 0)
                                confluence_factors.append("Near Support Level")
                            elif setup['action'] == 'SELL' and sr_conf.get('sell_confidence', 0) > 0:
                                additional_confidence += sr_conf.get('sell_confidence', 0)
                                confluence_factors.append("Near Resistance Level")
                        
                        # 2. Trend confidence
                        if 'trend' in factors:
                            if setup['action'] == 'BUY' and factors['trend'].get('buy_confidence', 0) > 0:
                                additional_confidence += factors['trend']['buy_confidence']
                                # Check if ADX indicates strong trend
                                if factors['trend'].get('adx') and factors['trend'].get('adx') > 25:
                                    confluence_factors.append("Strong Bullish Trend")
                                else:
                                    confluence_factors.append("Bullish Trend Alignment")
                            elif setup['action'] == 'SELL' and factors['trend'].get('sell_confidence', 0) > 0:
                                additional_confidence += factors['trend']['sell_confidence']
                                # Check if ADX indicates strong trend
                                if factors['trend'].get('adx') and factors['trend'].get('adx') > 25:
                                    confluence_factors.append("Strong Bearish Trend")
                                else:
                                    confluence_factors.append("Bearish Trend Alignment")
                        
                        # 3. Volatility confidence
                        if 'volatility' in factors:
                            if setup['action'] == 'BUY' and factors['volatility'].get('buy_confidence', 0) > 0:
                                additional_confidence += factors['volatility']['buy_confidence']
                                # Check if price is near lower Bollinger Band
                                if factors['volatility'].get('bb_percent_b') and factors['volatility'].get('bb_percent_b') < 0.2:
                                    confluence_factors.append("Oversold Condition")
                                else:
                                    confluence_factors.append("Favorable Volatility")
                            elif setup['action'] == 'SELL' and factors['volatility'].get('sell_confidence', 0) > 0:
                                additional_confidence += factors['volatility']['sell_confidence']
                                # Check if price is near upper Bollinger Band
                                if factors['volatility'].get('bb_percent_b') and factors['volatility'].get('bb_percent_b') > 0.8:
                                    confluence_factors.append("Overbought Condition")
                                else:
                                    confluence_factors.append("Favorable Volatility")
                
                # Calculate final confidence (cap at 0.95 to avoid overconfidence)
                final_confidence = min(base_confidence + additional_confidence, 0.95)
                
                # Log the confidence calculation for debugging
                logger.info(f"Confidence calculation: {base_confidence} (base) + {additional_confidence} (additional) = {final_confidence} (final)")
                
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
            
            # Add expected duration and exit strategy if available (from flexible duration setup)
            if setup_source == 'flexible':
                signal['details']['expected_duration'] = setup.get('expected_duration', 'unknown')
                signal['details']['exit_strategy'] = setup.get('exit_strategy', 'fixed')
                if 'confluence_factors' in setup['details']:
                    signal['details']['confluence_factors'] = setup['details']['confluence_factors']
            
            # Reduce confidence if not in killzone
            if not in_killzone and signal['confidence'] > 0.3:
                signal['confidence'] *= 0.7
                signal['reason'] += " (Not in killzone)"
            
            # Add more details for high confidence setups
            if signal['confidence'] > 0.7:
                signal['reason'] += " - High confidence setup"
                confluence_factors = []
                # Get setup action safely with a default value
                setup_action = setup.get('action', 'HOLD')
                
                if analysis.get('market_structure', {}).get('has_bullish_mss') and setup_action == 'BUY':
                    confluence_factors.append("Bullish MSS")
                if analysis.get('market_structure', {}).get('has_bearish_mss') and setup_action == 'SELL':
                    confluence_factors.append("Bearish MSS")
                if analysis.get('order_blocks', {}).get('has_bullish_breaker') and setup_action == 'BUY':
                    confluence_factors.append("Bullish Breaker")
                if analysis.get('order_blocks', {}).get('has_bearish_breaker') and setup_action == 'SELL':
                    confluence_factors.append("Bearish Breaker")
                if analysis.get('fair_value_gaps', {}).get('has_bullish_fvg') and setup_action == 'BUY':
                    confluence_factors.append("Bullish FVG")
                if analysis.get('fair_value_gaps', {}).get('has_bearish_fvg') and setup_action == 'SELL':
                    confluence_factors.append("Bearish FVG")
                
                if confluence_factors:
                    signal['reason'] += f" with {', '.join(confluence_factors)}"
            
            return signal
            # If no valid setup was found
            if setup_source == 'none':
                return {'action': 'HOLD', 'confidence': 0, 'reason': "No valid ICT setup found"}

        except Exception as e:
            logger.error(f"Error generating ICT signal: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': f"Exception: {str(e)}"}


    
    def enhance_with_daily_bias(self, ict_signal, daily_data):
        """Enhance the ICT signal with daily timeframe bias analysis
        
        This method analyzes the daily timeframe data to determine the overall
        market bias and enhances the ICT signal accordingly. This is a key
        component of the ICT methodology which emphasizes trading with the
        higher timeframe bias.
        
        Args:
            ict_signal (dict): Signal from ICT analysis on lower timeframe
            daily_data (pd.DataFrame): Daily timeframe price data
            
        Returns:
            dict: Enhanced ICT signal with daily bias consideration
        """
        try:
            # Create a copy of the signal to avoid modifying the original
            enhanced_signal = ict_signal.copy()
            
            # Analyze daily data for market structure
            daily_data = identify_market_structure(daily_data)
            
            # Determine daily bias
            daily_bias = determine_daily_bias(daily_data)
            
            # Add daily bias to signal details
            if 'details' not in enhanced_signal:
                enhanced_signal['details'] = {}
            
            enhanced_signal['details']['daily_bias'] = daily_bias
            
            # Check for bullish/bearish market structure on daily with comprehensive error handling
            has_bullish_mss = False
            has_bearish_mss = False
            
            # Ensure required columns exist
            for col in ['bullish_mss', 'bearish_mss']:
                if col not in daily_data.columns:
                    daily_data[col] = False
            
            # Safely check for market structure shifts
            try:
                has_bullish_mss = daily_data['bullish_mss'].any()
                has_bearish_mss = daily_data['bearish_mss'].any()
            except Exception as e:
                logger.warning(f"Error checking daily market structure: {e}")
                
            enhanced_signal['details']['daily_has_bullish_mss'] = has_bullish_mss
            enhanced_signal['details']['daily_has_bearish_mss'] = has_bearish_mss
            
            # If signal action aligns with daily bias, increase confidence
            if (daily_bias == 'BULLISH' and enhanced_signal['action'] == 'BUY') or \
               (daily_bias == 'BEARISH' and enhanced_signal['action'] == 'SELL'):
                # Increase confidence by 20% (capped at 1.0)
                enhanced_signal['confidence'] = min(enhanced_signal['confidence'] * 1.2, 1.0)
                enhanced_signal['reason'] += f" (aligned with {daily_bias} daily bias)"
            
            # If signal action contradicts daily bias, reduce confidence
            elif (daily_bias == 'BULLISH' and enhanced_signal['action'] == 'SELL') or \
                 (daily_bias == 'BEARISH' and enhanced_signal['action'] == 'BUY'):
                # Reduce confidence by 30%
                enhanced_signal['confidence'] *= 0.7
                enhanced_signal['reason'] += f" (caution: against {daily_bias} daily bias)"
            
            # If daily bias is neutral, no adjustment needed
            else:
                enhanced_signal['reason'] += " (neutral daily bias)"
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with daily bias: {e}")
            return ict_signal
    
    def enhance_trader_signal(self, original_signal, ict_signal):
        """Enhance the original trader signal with ICT analysis
        
        Args:
            original_signal (dict): Original signal from StandaloneTrader
            ict_signal (dict): Signal from ICT analysis
            
        Returns:
            dict: Enhanced trading signal
        """
        try:
            # If original signal is HOLD, use ICT signal if it has sufficient confidence
            if original_signal['action'] == 'HOLD' and ict_signal['confidence'] >= 0.6:
                return ict_signal
            
            # If original signal has an action, check if ICT agrees
            if original_signal['action'] in ['BUY', 'SELL']:
                # If ICT agrees, enhance confidence
                if ict_signal['action'] == original_signal['action'] and ict_signal['confidence'] > 0:
                    # Weighted average of confidences (60% original, 40% ICT)
                    enhanced_confidence = (0.6 * original_signal['confidence']) + (0.4 * ict_signal['confidence'])
                    
                    # Create enhanced signal
                    enhanced_signal = original_signal.copy()
                    enhanced_signal['confidence'] = enhanced_confidence
                    enhanced_signal['reason'] += f" + ICT confirmation: {ict_signal['reason']}"
                    
                    # Add ICT details
                    if 'details' not in enhanced_signal:
                        enhanced_signal['details'] = {}
                    enhanced_signal['details']['ict'] = ict_signal['details']
                    
                    return enhanced_signal
                
                # If ICT disagrees with high confidence, reduce original confidence
                elif ict_signal['action'] != 'HOLD' and ict_signal['action'] != original_signal['action'] and ict_signal['confidence'] > 0.7:
                    # Reduce confidence by 40%
                    enhanced_signal = original_signal.copy()
                    enhanced_signal['confidence'] *= 0.6
                    enhanced_signal['reason'] += f" (Warning: ICT indicates {ict_signal['action']} instead)"
                    
                    # Add ICT details
                    if 'details' not in enhanced_signal:
                        enhanced_signal['details'] = {}
                    enhanced_signal['details']['ict_warning'] = ict_signal['details']
                    
                    return enhanced_signal
            
            # Default: return original signal
            return original_signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with ICT: {e}")
            return original_signal
