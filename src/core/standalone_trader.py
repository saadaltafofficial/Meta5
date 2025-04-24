#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Forex Trading Bot for ICT-based Trading

Implements the Inner Circle Trader (ICT) methodology for multiple currency pairs
with optimized risk management and Pakistan killzone awareness.
"""

import os
import time
import logging
import warnings
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import from project modules with new structure
from src.utils.config_loader import get_config, save_config
from src.ict.ict_mechanics import *
from src.ict.ict_integration import ICTAnalyzer
from src.core.balance_monitor import BalanceMonitor
from src.core.database import TradeDatabase

# Disable pandas warnings globally
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import economic calendar module
try:
    from src.core.economic_calendar import EconomicCalendar
except ImportError:
    logging.warning("Economic calendar module not found. Economic event analysis will be disabled.")
    EconomicCalendar = None

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from config directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/.env'))

class StandaloneTrader:
    """Standalone Forex Trading Bot implementing ICT methodology for multiple currency pairs"""
    
    def __init__(self):
        # Load configuration from config directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/config.json')
        self.config = get_config(config_path)
        
        # Trading parameters from config
        self.currency_pairs = self.config['trading'].get('currency_pairs', ['EURUSD', 'GBPUSD'])
        self.min_confidence = self.config['trading'].get('min_confidence', 0.15)
        self.risk_percent = self.config['trading']['risk_management'].get('risk_percent', 1.5)
        self.max_balance_percent = self.config['trading']['risk_management'].get('max_balance_percent', 10.0)
        self.auto_trading = self.config['trading'].get('auto_trading', True)
        self.take_profit_ratios = self.config['trading']['risk_management'].get('take_profit_ratios', [1.5, 2.75, 4.75])
        
        # Indicator parameters
        self.ma_fast_period = self.config['indicators']['moving_averages'].get('fast_period', 9)
        self.ma_slow_period = self.config['indicators']['moving_averages'].get('slow_period', 21)
        self.ma_trend_period = self.config['indicators']['moving_averages'].get('trend_period', 50)
        
        # ICT model parameters
        self.use_order_blocks = self.config['indicators']['ict_model'].get('order_blocks', True)
        self.use_fair_value_gaps = self.config['indicators']['ict_model'].get('fair_value_gaps', True)
        self.use_liquidity_levels = self.config['indicators']['ict_model'].get('liquidity_levels', True)
        
        # Data parameters
        self.bars_to_analyze = self.config['data'].get('bars_to_analyze', 100)
        self.economic_event_hours_window = self.config['data'].get('economic_event_hours_window', 24)
        
        # MT5 parameters (still from env for security)
        self.mt5_server = os.getenv('MT5_SERVER')
        self.mt5_login = os.getenv('MT5_LOGIN')
        self.mt5_password = os.getenv('MT5_PASSWORD')
        
        # Initialize MT5
        self.initialize_mt5()
        
        # Initialize Economic Calendar if available and enabled
        self.economic_calendar = None
        if EconomicCalendar is not None and self.config['data'].get('economic_calendar_enabled', True):
            self.economic_calendar = EconomicCalendar()
            logger.info("Economic calendar initialized")
        
        # Initialize ICT Analyzer
        self.ict_analyzer = ICTAnalyzer(self.config)
        logger.info("ICT Analyzer initialized with advanced mechanics")
        
        # Initialize Balance Monitor
        self.balance_monitor = BalanceMonitor(self.config)
        logger.info("Balance Monitor initialized")
        
        # Initialize Trade Database
        self.trade_database = TradeDatabase(self.config)
        logger.info("Trade Database initialized")
        
        # Store signals and trades
        self.latest_signals = {}
        self.active_trades = {}
        
        logger.info(f"Trader initialized with {len(self.currency_pairs)} currency pairs: {', '.join(self.currency_pairs)}")
        logger.info(f"Trading on timeframes: {', '.join(self.config['trading']['timeframes'].get('analysis', ['H1']))}")
    
    def initialize_mt5(self):
        """Initialize MetaTrader 5 connection"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Connect to MT5 account
            if not mt5.login(
                login=int(self.mt5_login),
                password=self.mt5_password,
                server=self.mt5_server
            ):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            logger.info(f"Connected to MT5 account: {self.mt5_login} on {self.mt5_server}")
            return True
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False
    
    def get_forex_data(self, symbol, timeframe=None, bars=None):
        """Get forex data from MT5 for a specific timeframe"""
        try:
            # Prepare the symbol
            symbol = symbol.upper()
            
            # Use config values if not specified
            if timeframe is None:
                timeframe = self.config['trading']['timeframes'].get('primary_mt5', mt5.TIMEFRAME_H1)
            if bars is None:
                bars = self.config['data'].get('bars_to_analyze', 100)
            
            # Get bars
            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if bars is None or len(bars) == 0:
                logger.warning(f"No data for {symbol} on timeframe {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add timeframe information to the dataframe
            df.attrs['timeframe'] = timeframe
            df.attrs['timeframe_str'] = self._get_timeframe_str(timeframe)
            
            return df
        except Exception as e:
            logger.error(f"Error getting forex data for {symbol}: {e}")
            return None
    
    def _get_timeframe_str(self, timeframe):
        """Convert MT5 timeframe constant to string representation"""
        timeframe_dict = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_W1: "W1",
            mt5.TIMEFRAME_MN1: "MN1"
        }
        return timeframe_dict.get(timeframe, f"Unknown({timeframe})")
    
    def get_multi_timeframe_data(self, symbol, timeframes=None, bars=None):
        """Get forex data for multiple timeframes"""
        try:
            # Use config values if not specified
            if timeframes is None:
                timeframes = self.config['trading']['timeframes'].get('analysis_mt5', 
                                                                   [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1])
            if bars is None:
                bars = self.config['data'].get('bars_to_analyze', 100)
            
            multi_tf_data = {}
            
            for tf in timeframes:
                data = self.get_forex_data(symbol, timeframe=tf, bars=bars)
                if data is not None:
                    multi_tf_data[self._get_timeframe_str(tf)] = data
                else:
                    logger.warning(f"Could not get data for {symbol} on timeframe {self._get_timeframe_str(tf)}")
            
            return multi_tf_data if multi_tf_data else None
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data for {symbol}: {e}")
            return None
    
    def detect_price_patterns(self, data):
        """Detect price action patterns in the data"""
        try:
            # Create a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Calculate candle body and shadow sizes
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['range'] = df['high'] - df['low']
            
            # Calculate average body size for reference (last 10 candles)
            avg_body_size = df['body_size'].rolling(window=10).mean()
            
            # Initialize pattern columns
            patterns = {}
            
            # 1. Doji Pattern (small body, shadows on both sides)
            df['doji'] = (df['body_size'] < 0.1 * df['range']) & (df['upper_shadow'] > 0) & (df['lower_shadow'] > 0)
            
            # 2. Hammer Pattern (small body at top, long lower shadow, little/no upper shadow)
            df['hammer'] = ((df['body_size'] < 0.3 * df['range']) & 
                           (df['lower_shadow'] > 2 * df['body_size']) & 
                           (df['upper_shadow'] < 0.1 * df['range']))
            
            # 3. Shooting Star (small body at bottom, long upper shadow, little/no lower shadow)
            df['shooting_star'] = ((df['body_size'] < 0.3 * df['range']) & 
                                  (df['upper_shadow'] > 2 * df['body_size']) & 
                                  (df['lower_shadow'] < 0.1 * df['range']))
            
            # 4. Engulfing Patterns
            # Bullish engulfing (current green candle engulfs previous red candle)
            df['bullish_engulfing'] = False
            # Bearish engulfing (current red candle engulfs previous green candle)
            df['bearish_engulfing'] = False
            
            for i in range(1, len(df)):
                # Current candle is green (close > open)
                current_green = df['close'].iloc[i] > df['open'].iloc[i]
                # Previous candle is red (close < open)
                prev_red = df['close'].iloc[i-1] < df['open'].iloc[i-1]
                
                # Bullish engulfing: current green candle engulfs previous red candle
                if current_green and prev_red:
                    if (df['open'].iloc[i] <= df['close'].iloc[i-1] and
                        df['close'].iloc[i] >= df['open'].iloc[i-1]):
                        df['bullish_engulfing'].iloc[i] = True
                
                # Current candle is red (close < open)
                current_red = df['close'].iloc[i] < df['open'].iloc[i]
                # Previous candle is green (close > open)
                prev_green = df['close'].iloc[i-1] > df['open'].iloc[i-1]
                
                # Bearish engulfing: current red candle engulfs previous green candle
                if current_red and prev_green:
                    if (df['open'].iloc[i] >= df['close'].iloc[i-1] and
                        df['close'].iloc[i] <= df['open'].iloc[i-1]):
                        df['bearish_engulfing'].iloc[i] = True
            
            # 5. Pin Bar (long shadow in one direction, small body, small/no shadow in other direction)
            df['bullish_pin_bar'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                                    (df['upper_shadow'] < 0.2 * df['lower_shadow']) & 
                                    (df['body_size'] < 0.3 * df['range']))
            
            df['bearish_pin_bar'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                    (df['lower_shadow'] < 0.2 * df['upper_shadow']) & 
                                    (df['body_size'] < 0.3 * df['range']))
            
            # 6. Morning Star (three-candle bullish reversal pattern)
            df['morning_star'] = False
            
            # 7. Evening Star (three-candle bearish reversal pattern)
            df['evening_star'] = False
            
            # Detect 3-candle patterns
            for i in range(2, len(df)):
                # Morning Star: 1) long red candle, 2) small body, 3) long green candle
                first_red = df['close'].iloc[i-2] < df['open'].iloc[i-2]
                first_long_body = df['body_size'].iloc[i-2] > avg_body_size.iloc[i-2]
                
                middle_small_body = df['body_size'].iloc[i-1] < 0.5 * avg_body_size.iloc[i-1]
                
                last_green = df['close'].iloc[i] > df['open'].iloc[i]
                last_long_body = df['body_size'].iloc[i] > avg_body_size.iloc[i]
                
                if first_red and first_long_body and middle_small_body and last_green and last_long_body:
                    # Check if middle candle gaps down and last candle closes into first candle
                    if (df[['open', 'close']].max(axis=1).iloc[i-1] < df['close'].iloc[i-2] and
                        df['close'].iloc[i] > (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2):
                        df['morning_star'].iloc[i] = True
                
                # Evening Star: 1) long green candle, 2) small body, 3) long red candle
                first_green = df['close'].iloc[i-2] > df['open'].iloc[i-2]
                first_long_body = df['body_size'].iloc[i-2] > avg_body_size.iloc[i-2]
                
                last_red = df['close'].iloc[i] < df['open'].iloc[i]
                last_long_body = df['body_size'].iloc[i] > avg_body_size.iloc[i]
                
                if first_green and first_long_body and middle_small_body and last_red and last_long_body:
                    # Check if middle candle gaps up and last candle closes into first candle
                    if (df[['open', 'close']].min(axis=1).iloc[i-1] > df['close'].iloc[i-2] and
                        df['close'].iloc[i] < (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2):
                        df['evening_star'].iloc[i] = True
            
            # Check for patterns in the last 3 candles
            latest_patterns = {}
            pattern_names = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
                           'bullish_pin_bar', 'bearish_pin_bar', 'morning_star', 'evening_star']
            
            for pattern in pattern_names:
                # Check if pattern exists in the last 3 candles
                if df[pattern].iloc[-3:].any():
                    latest_patterns[pattern] = True
                else:
                    latest_patterns[pattern] = False
            
            # Determine pattern signals
            pattern_signals = {}
            
            # Bullish patterns
            bullish_patterns = ['hammer', 'bullish_engulfing', 'bullish_pin_bar', 'morning_star']
            for pattern in bullish_patterns:
                if latest_patterns.get(pattern, False):
                    pattern_signals[pattern] = {'action': 'BUY', 'weight': 1.2}  # Higher weight for price patterns
            
            # Bearish patterns
            bearish_patterns = ['shooting_star', 'bearish_engulfing', 'bearish_pin_bar', 'evening_star']
            for pattern in bearish_patterns:
                if latest_patterns.get(pattern, False):
                    pattern_signals[pattern] = {'action': 'SELL', 'weight': 1.2}  # Higher weight for price patterns
            
            # Neutral patterns (could go either way, needs confirmation)
            if latest_patterns.get('doji', False):
                # Doji after uptrend suggests hesitation/reversal
                if df['close'].iloc[-1] > df['ma_50'].iloc[-1]:
                    pattern_signals['doji'] = {'action': 'SELL', 'weight': 0.8}
                # Doji after downtrend suggests hesitation/reversal
                elif df['close'].iloc[-1] < df['ma_50'].iloc[-1]:
                    pattern_signals['doji'] = {'action': 'BUY', 'weight': 0.8}
                else:
                    pattern_signals['doji'] = {'action': 'NEUTRAL', 'weight': 0.8}
            
            return pattern_signals, latest_patterns
        except Exception as e:
            logger.error(f"Error detecting price patterns: {e}")
            return {}, {}
    
    def technical_analysis(self, data):
        """Perform enhanced technical analysis on forex data"""
        try:
            # Calculate indicators
            # Moving Averages
            data['ma_9'] = data['close'].rolling(window=9).mean()  # Fast MA from ICT model
            data['ma_21'] = data['close'].rolling(window=21).mean()  # Slow MA from ICT model
            data['ma_50'] = data['close'].rolling(window=50).mean()  # Trend MA from ICT model
            
            # MACD
            data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            
            # Calculate all BB components at once to avoid pandas warnings
            # Calculate middle band (SMA)
            bb_middle = data['close'].rolling(window=bb_period).mean()
            
            # Calculate standard deviation
            rolling_std = data['close'].rolling(window=bb_period).std()
            
            # Calculate upper and lower bands
            bb_upper = bb_middle + (rolling_std * bb_std)
            bb_lower = bb_middle - (rolling_std * bb_std)
            
            # Calculate bandwidth and %B
            bb_bandwidth = (bb_upper - bb_lower) / bb_middle
            bb_percent_b = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Assign all values at once
            data = data.assign(
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_bandwidth=bb_bandwidth,
                bb_percent_b=bb_percent_b
            )
            
            # Stochastic Oscillator
            k_period = 14
            d_period = 3
            
            # Calculate components
            low_min = data['low'].rolling(window=k_period).min()
            high_max = data['high'].rolling(window=k_period).max()
            
            # Calculate stochastic values
            stoch_k = 100 * ((data['close'] - low_min) / (high_max - low_min + 1e-9))  # Adding small value to avoid division by zero
            stoch_d = stoch_k.rolling(window=d_period).mean()
            
            # Assign values at once
            data = data.assign(
                stoch_k=stoch_k,
                stoch_d=stoch_d
            )
            
            # ADX (Average Directional Index)
            high_diff = data['high'].diff()
            low_diff = -data['low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr1 = data['high'] - data['low']
            tr2 = abs(data['high'] - data['close'].shift(1))
            tr3 = abs(data['low'] - data['close'].shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            atr_period = 14
            atr = tr.rolling(window=atr_period).mean()
            
            pos_di = 100 * (pos_dm.rolling(window=atr_period).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=atr_period).mean() / atr)
            
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9)  # Adding small value to avoid division by zero
            adx = dx.rolling(window=atr_period).mean()
            
            # Assign values at once
            data = data.assign(
                atr=atr,
                adx=adx
            )
            
            # ==================== ICT Model Components ====================
            # Identify candle colors (red/green)
            red_candle = data['close'] < data['open']
            green_candle = data['close'] > data['open']
            
            # Assign values at once
            data = data.assign(
                red_candle=red_candle,
                green_candle=green_candle
            )
            
            # Identify bullish and bearish moves (3 consecutive candles)
            bullish_move = pd.Series(False, index=data.index)
            bearish_move = pd.Series(False, index=data.index)
            
            # Look for 3 consecutive green candles (bullish move)
            for i in range(3, len(data)):
                idx = data.index[i]  # Get the actual index for this position
                if data['green_candle'].iloc[i-1] and data['green_candle'].iloc[i-2] and data['green_candle'].iloc[i-3]:
                    bullish_move.loc[idx] = True
            
            # Look for 3 consecutive red candles (bearish move)
            for i in range(3, len(data)):
                idx = data.index[i]  # Get the actual index for this position
                if data['red_candle'].iloc[i-1] and data['red_candle'].iloc[i-2] and data['red_candle'].iloc[i-3]:
                    bearish_move.loc[idx] = True
            
            # Assign values at once
            data = data.assign(
                bullish_move=bullish_move,
                bearish_move=bearish_move
            )
            
            # Identify Order Blocks
            bullish_ob = pd.Series(False, index=data.index)  # Red candle before bullish move
            bearish_ob = pd.Series(False, index=data.index)  # Green candle before bearish move
            
            # Look for bullish order blocks (red candle before bullish move)
            for i in range(4, len(data)):
                if i-4 >= 0:  # Make sure we don't go out of bounds
                    idx = data.index[i-4]  # Get the actual index for the order block position
                    if data['bullish_move'].iloc[i] and data['red_candle'].iloc[i-4]:
                        bullish_ob.loc[idx] = True
            
            # Look for bearish order blocks (green candle before bearish move)
            for i in range(4, len(data)):
                if i-4 >= 0:  # Make sure we don't go out of bounds
                    idx = data.index[i-4]  # Get the actual index for the order block position
                    if data['bearish_move'].iloc[i] and data['green_candle'].iloc[i-4]:
                        bearish_ob.loc[idx] = True
            
            # Initialize Order Block level series
            bullish_ob_top = pd.Series(np.nan, index=data.index)
            bullish_ob_bottom = pd.Series(np.nan, index=data.index)
            bearish_ob_top = pd.Series(np.nan, index=data.index)
            bearish_ob_bottom = pd.Series(np.nan, index=data.index)
            
            # Assign order block values
            data = data.assign(
                bullish_ob=bullish_ob,
                bearish_ob=bearish_ob,
                bullish_ob_top=bullish_ob_top,
                bullish_ob_bottom=bullish_ob_bottom,
                bearish_ob_top=bearish_ob_top,
                bearish_ob_bottom=bearish_ob_bottom
            )
            
            # Create new series for order block values
            bullish_ob_top_new = data['bullish_ob_top'].copy()
            bullish_ob_bottom_new = data['bullish_ob_bottom'].copy()
            bearish_ob_top_new = data['bearish_ob_top'].copy()
            bearish_ob_bottom_new = data['bearish_ob_bottom'].copy()
            
            # Set values for bullish order blocks
            for i in range(len(data)):
                if data['bullish_ob'].iloc[i]:
                    idx = data.index[i]
                    bullish_ob_top_new.loc[idx] = data['open'].iloc[i]
                    bullish_ob_bottom_new.loc[idx] = data['close'].iloc[i]
            
            # Set values for bearish order blocks
            for i in range(len(data)):
                if data['bearish_ob'].iloc[i]:
                    idx = data.index[i]
                    bearish_ob_top_new.loc[idx] = data['close'].iloc[i]
                    bearish_ob_bottom_new.loc[idx] = data['open'].iloc[i]
            
            # Update the dataframe with new values
            data = data.assign(
                bullish_ob_top=bullish_ob_top_new,
                bullish_ob_bottom=bullish_ob_bottom_new,
                bearish_ob_top=bearish_ob_top_new,
                bearish_ob_bottom=bearish_ob_bottom_new
            )
            
            # Identify Fair Value Gaps (FVG)
            bullish_fvg = pd.Series(False, index=data.index)
            bearish_fvg = pd.Series(False, index=data.index)
            
            # Bullish FVG: Current low > previous high
            for i in range(2, len(data)):
                idx = data.index[i]
                if data['low'].iloc[i] > data['high'].iloc[i-2]:
                    bullish_fvg.loc[idx] = True
            
            # Bearish FVG: Current high < previous low
            for i in range(2, len(data)):
                idx = data.index[i]
                if data['high'].iloc[i] < data['low'].iloc[i-2]:
                    bearish_fvg.loc[idx] = True
            
            # Assign FVG values to dataframe
            data = data.assign(
                bullish_fvg=bullish_fvg,
                bearish_fvg=bearish_fvg
            )
            
            # Initialize FVG level series
            bullish_fvg_top = pd.Series(np.nan, index=data.index)
            bullish_fvg_bottom = pd.Series(np.nan, index=data.index)
            bearish_fvg_top = pd.Series(np.nan, index=data.index)
            bearish_fvg_bottom = pd.Series(np.nan, index=data.index)
            
            # Set values for bullish FVG
            for i in range(2, len(data)):
                idx = data.index[i]
                if data['bullish_fvg'].iloc[i]:
                    bullish_fvg_bottom.loc[idx] = data['high'].iloc[i-2]
                    bullish_fvg_top.loc[idx] = data['low'].iloc[i]
            
            # Set values for bearish FVG
            for i in range(2, len(data)):
                idx = data.index[i]
                if data['bearish_fvg'].iloc[i]:
                    bearish_fvg_top.loc[idx] = data['low'].iloc[i-2]
                    bearish_fvg_bottom.loc[idx] = data['high'].iloc[i]
            
            # Assign FVG level values to dataframe
            data = data.assign(
                bullish_fvg_top=bullish_fvg_top,
                bullish_fvg_bottom=bullish_fvg_bottom,
                bearish_fvg_top=bearish_fvg_top,
                bearish_fvg_bottom=bearish_fvg_bottom
            )
            
            # Detect price action patterns
            price_pattern_signals, latest_patterns = self.detect_price_patterns(data)
            
            # Determine action based on indicators
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Initialize signals with weights
            signals = {}
            
            # ICT Model MA Crossover (higher weight)
            if latest['ma_9'] > latest['ma_21'] and prev['ma_9'] <= prev['ma_21']:
                signals['ict_ma_crossover'] = {'action': 'BUY', 'weight': 2.0}
            elif latest['ma_9'] < latest['ma_21'] and prev['ma_9'] >= prev['ma_21']:
                signals['ict_ma_crossover'] = {'action': 'SELL', 'weight': 2.0}
            else:
                signals['ict_ma_crossover'] = {'action': 'NEUTRAL', 'weight': 2.0}
            
            # Trend Filter
            if latest['close'] > latest['ma_50']:
                signals['trend'] = {'action': 'BUY', 'weight': 1.5}
            elif latest['close'] < latest['ma_50']:
                signals['trend'] = {'action': 'SELL', 'weight': 1.5}
            else:
                signals['trend'] = {'action': 'NEUTRAL', 'weight': 1.5}
            
            # ICT Order Blocks (highest weight)
            # Check for recent bullish order blocks (within last 10 bars)
            recent_bullish_ob = data['bullish_ob'].iloc[-10:].any()
            # Check for recent bearish order blocks (within last 10 bars)
            recent_bearish_ob = data['bearish_ob'].iloc[-10:].any()
            
            if recent_bullish_ob:
                signals['order_blocks'] = {'action': 'BUY', 'weight': 2.5}
            elif recent_bearish_ob:
                signals['order_blocks'] = {'action': 'SELL', 'weight': 2.5}
            else:
                signals['order_blocks'] = {'action': 'NEUTRAL', 'weight': 2.5}
            
            # ICT Fair Value Gaps (high weight)
            # Check for recent bullish FVG (within last 5 bars)
            recent_bullish_fvg = data['bullish_fvg'].iloc[-5:].any()
            # Check for recent bearish FVG (within last 5 bars)
            recent_bearish_fvg = data['bearish_fvg'].iloc[-5:].any()
            
            if recent_bullish_fvg:
                signals['fair_value_gaps'] = {'action': 'BUY', 'weight': 2.0}
            elif recent_bearish_fvg:
                signals['fair_value_gaps'] = {'action': 'SELL', 'weight': 2.0}
            else:
                signals['fair_value_gaps'] = {'action': 'NEUTRAL', 'weight': 2.0}
            
            # MACD
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                signals['macd'] = {'action': 'BUY', 'weight': 1.0}
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                signals['macd'] = {'action': 'SELL', 'weight': 1.0}
            else:
                signals['macd'] = {'action': 'NEUTRAL', 'weight': 1.0}
            
            # RSI
            if latest['rsi'] < 30:
                signals['rsi'] = {'action': 'BUY', 'weight': 1.0}  # Oversold
            elif latest['rsi'] > 70:
                signals['rsi'] = {'action': 'SELL', 'weight': 1.0}  # Overbought
            else:
                signals['rsi'] = {'action': 'NEUTRAL', 'weight': 1.0}
            
            # Bollinger Bands
            if latest['close'] < latest['bb_lower']:
                signals['bollinger'] = {'action': 'BUY', 'weight': 1.0}  # Price below lower band
            elif latest['close'] > latest['bb_upper']:
                signals['bollinger'] = {'action': 'SELL', 'weight': 1.0}  # Price above upper band
            else:
                signals['bollinger'] = {'action': 'NEUTRAL', 'weight': 1.0}
            
            # Stochastic
            if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
                signals['stochastic'] = {'action': 'BUY', 'weight': 1.0}  # Oversold and %K crossing above %D
            elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
                signals['stochastic'] = {'action': 'SELL', 'weight': 1.0}  # Overbought and %K crossing below %D
            else:
                signals['stochastic'] = {'action': 'NEUTRAL', 'weight': 1.0}
            
            # ADX - Trend Strength
            if latest['adx'] > 25:
                # Strong trend - use DI+ and DI- for direction
                if pos_di.iloc[-1] > neg_di.iloc[-1]:
                    signals['adx'] = {'action': 'BUY', 'weight': 1.5}
                else:
                    signals['adx'] = {'action': 'SELL', 'weight': 1.5}
            else:
                signals['adx'] = {'action': 'NEUTRAL', 'weight': 1.5}  # Weak trend
                
            # Add price pattern signals to our signals dictionary
            for pattern, pattern_signal in price_pattern_signals.items():
                signals[pattern] = pattern_signal
            
            # Calculate weighted confidence
            buy_weight = sum(signal['weight'] for signal in signals.values() if signal['action'] == 'BUY')
            sell_weight = sum(signal['weight'] for signal in signals.values() if signal['action'] == 'SELL')
            total_weight = sum(signal['weight'] for signal in signals.values())
            
            # Determine overall action
            if buy_weight > sell_weight:
                action = 'BUY'
                confidence = buy_weight / total_weight
            elif sell_weight > buy_weight:
                action = 'SELL'
                confidence = sell_weight / total_weight
            else:
                action = 'HOLD'
                confidence = 0
            
            # Create a simplified signals dict for display
            signal_actions = {k: v['action'] for k, v in signals.items()}
            
            return {
                'action': action,
                'confidence': confidence,  # Already in 0-1 scale
                'signals': signal_actions,
                'raw_signals': signals  # Include the detailed signals with weights
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'signals': {}}
    
    def multi_timeframe_analysis(self, pair):
        """Perform multi-timeframe analysis for a currency pair"""
        try:
            # Define timeframes to analyze (H1, H4, D1)
            timeframes = self.config['trading']['timeframes'].get('analysis_mt5', 
                                                               [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1])
            
            # Get data for all timeframes
            multi_tf_data = self.get_multi_timeframe_data(pair, timeframes)
            if not multi_tf_data:
                logger.warning(f"Could not get multi-timeframe data for {pair}")
                return {'action': 'HOLD', 'confidence': 0, 'reason': 'No multi-timeframe data available'}
                
            # Analyze each timeframe
            tf_signals = {}
            tf_weights = {
                mt5.TIMEFRAME_H1: 1.0,   # Lowest weight for H1 (shortest timeframe)
                mt5.TIMEFRAME_H4: 1.5,   # Medium weight for H4
                mt5.TIMEFRAME_D1: 2.0    # Highest weight for D1 (longest timeframe)
            }
            
            # Store all signals from different timeframes
            all_signals = {}
            all_raw_signals = {}
            
            # Store price patterns from each timeframe for detailed reporting
            all_timeframe_patterns = {}
            
            # Analyze each timeframe
            for tf, data in multi_tf_data.items():
                # Perform technical analysis on this timeframe
                analysis = self.technical_analysis(data)
                
                # Store the analysis with timeframe info
                tf_str = self._get_timeframe_str(tf)
                tf_signals[tf_str] = {
                    'action': analysis.get('action', 'HOLD'),
                    'confidence': analysis.get('confidence', 0),
                    'signals': analysis.get('signals', {}),
                    'raw_signals': analysis.get('raw_signals', {})
                }
                
                # Add to all signals with timeframe prefix
                for indicator, signal_data in analysis.get('signals', {}).items():
                    # Handle both dict and string formats for backward compatibility
                    if isinstance(signal_data, dict):
                        action = signal_data.get('action', 'NEUTRAL')
                        weight = signal_data.get('weight', 1.0) * tf_weights.get(tf, 1.0)
                    else:
                        # If it's a string, assume it's the action with default weight
                        action = signal_data
                        weight = 1.0 * tf_weights.get(tf, 1.0)
                    
                    all_signals[f"{tf_str}_{indicator}"] = {
                        'action': action,
                        'weight': weight  # Apply timeframe weight
                    }
                
                # Add raw signals
                for indicator, value in analysis.get('raw_signals', {}).items():
                    all_raw_signals[f"{tf_str}_{indicator}"] = value
                
                # Store price patterns detected in this timeframe
                _, latest_patterns = self.detect_price_patterns(data)
                if latest_patterns:
                    all_timeframe_patterns[tf_str] = latest_patterns
            
            # Calculate overall action and confidence based on weighted signals
            buy_weight = 0
            sell_weight = 0
            total_weight = 0
            
            for indicator, signal_data in all_signals.items():
                action = signal_data.get('action', 'NEUTRAL')
                weight = signal_data.get('weight', 1.0)
                
                if action == 'BUY':
                    buy_weight += weight
                elif action == 'SELL':
                    sell_weight += weight
                
                total_weight += weight
            
            # Determine action based on weights
            if total_weight > 0:
                if buy_weight > sell_weight:
                    action = 'BUY'
                    confidence = buy_weight / total_weight
                elif sell_weight > buy_weight:
                    action = 'SELL'
                    confidence = sell_weight / total_weight
                else:
                    action = 'HOLD'
                    confidence = 0
            else:
                action = 'HOLD'
                confidence = 0
            
            # Create reason string with timeframe information
            reason_parts = []
            
            # Add timeframe signals (highest timeframe first)
            for tf_str in ['D1', 'H4', 'H1']:
                if tf_str in tf_signals:
                    tf_action = tf_signals[tf_str]['action']
                    tf_conf = tf_signals[tf_str]['confidence']
                    reason_parts.append(f"{tf_str}: {tf_action} ({tf_conf:.2f})")
            
            # Create the final reason string
            reason = ", ".join(reason_parts)
            
            # Create enhanced multi-timeframe signal
            signal = {
                'pair': pair,
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'timeframe_signals': tf_signals,
                'signals': all_signals,
                'raw_signals': all_raw_signals,
                'price_patterns': all_timeframe_patterns,  # Add price patterns to the signal
                'timestamp': datetime.now()
            }
            
            return signal
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {pair}: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': f"Error: {str(e)}"}
    
    def generate_signal(self, pair):
        """Generate trading signal for a currency pair using enhanced multi-timeframe analysis and ICT mechanics"""
        try:
            # Check if market is open
            if not self.is_market_open():
                logger.info(f"Market is closed, no signal generated for {pair}")
                return {'action': 'HOLD', 'confidence': 0, 'reason': 'Market is closed'}
            
            # Check if trading is allowed based on balance monitor
            if not self.balance_monitor.can_trade():
                logger.warning(f"Trading paused due to drawdown limit being reached, no signal generated for {pair}")
                balance_status = self.balance_monitor.get_status()
                return {
                    'action': 'HOLD', 
                    'confidence': 0, 
                    'reason': f"Trading paused due to drawdown ({balance_status['drawdown_percent']:.2f}% > {balance_status['drawdown_limit']:.2f}%)",
                    'balance_status': balance_status
                }
            
            # Check for high-impact economic events - ICT methodology focuses on very immediate timeframes
            has_event, event = self.check_economic_events(pair, hours_window=self.config['data'].get('economic_event_hours_window', 0.5))
            if has_event and event:
                # If high impact event is very close (within 30 minutes), avoid trading per ICT methodology
                event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S')
                minutes_until = (event_time - datetime.now()).total_seconds() / 60
                
                # Only avoid trading if event is very close (within 30 minutes before or 15 minutes after)
                if -15 <= minutes_until <= 30:
                    logger.info(f"ICT Rule: Avoiding trade due to imminent economic event: {event['event']} in {minutes_until:.1f} minutes")
                    signal = {
                        'action': 'HOLD', 
                        'confidence': 0, 
                        'reason': f"ICT Rule: High impact economic event imminent: {event['event']} in {minutes_until:.1f} minutes",
                        'event': event
                    }
                    return signal
                else:
                    # For events further away, just log but continue with analysis
                    logger.info(f"Economic event detected but not imminent: {event['event']} in {minutes_until/60:.1f} hours - continuing with analysis")
            
            # === STEP 1: Get multi-timeframe data ===
            # Get data for the primary timeframe (H4 by default)
            primary_tf = self.config['trading']['timeframes'].get('primary_mt5', mt5.TIMEFRAME_H4)
            primary_data = self.get_forex_data(pair, primary_tf)
            if primary_data is None:
                return {'action': 'HOLD', 'confidence': 0, 'reason': 'No primary timeframe data available'}
            
            # Get data for the daily timeframe for bias determination
            daily_data = self.get_forex_data(pair, mt5.TIMEFRAME_D1)
            
            # === STEP 2: Perform traditional technical analysis ===
            # This is our existing analysis approach
            traditional_signal = self.multi_timeframe_analysis(pair)
            
            # === STEP 3: Perform advanced ICT analysis ===
            # Apply ICT mechanics to the primary timeframe data
            ict_signal = self.ict_analyzer.generate_ict_signal(primary_data, pair)
            
            # If we have daily data, determine bias and enhance ICT signal
            if daily_data is not None:
                ict_signal = self.ict_analyzer.enhance_with_daily_bias(ict_signal, daily_data)
            
            # === STEP 4: Combine signals ===
            # Weight the signals (ICT analysis has higher weight)
            traditional_weight = 0.5  # Changed from 0.4 to 0.5 for more balanced weighting
            ict_weight = 0.5  # Changed from 0.6 to 0.5 for more balanced weighting
            
            # Get actions and confidences
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_conf = traditional_signal.get('confidence', 0)
            ict_action = ict_signal.get('action', 'HOLD')
            ict_conf = ict_signal.get('confidence', 0)
            
            # Calculate weighted confidence for each action type
            buy_confidence = 0
            sell_confidence = 0
            
            # For traditional signals
            if trad_action == 'BUY':
                buy_confidence += trad_conf * traditional_weight
            elif trad_action == 'SELL':
                sell_confidence += trad_conf * traditional_weight
            
            # For ICT signals - even if HOLD, check if confidence boost suggests a direction
            if ict_action == 'BUY':
                buy_confidence += ict_conf * ict_weight
            elif ict_action == 'SELL':
                sell_confidence += ict_conf * ict_weight
            elif ict_action == 'HOLD' and ict_conf > 0:
                # Check if there are confluence factors suggesting a direction
                confluence_factors = ict_signal.get('confluence_factors', [])
                if any('Bullish' in factor for factor in confluence_factors):
                    # Apply half the confidence to BUY
                    buy_confidence += (ict_conf * ict_weight) * 0.5
                    logger.info(f"Applied partial ICT confidence to BUY due to bullish confluence factors: {ict_conf:.4f}")
                elif any('Bearish' in factor for factor in confluence_factors):
                    # Apply half the confidence to SELL
                    sell_confidence += (ict_conf * ict_weight) * 0.5
                    logger.info(f"Applied partial ICT confidence to SELL due to bearish confluence factors: {ict_conf:.4f}")
            
            # Log confidence values for debugging
            logger.info(f"Signal confidence calculation for {pair}: ICT={ict_action}({ict_conf:.4f}), Traditional={trad_action}({trad_conf:.4f})")
            logger.info(f"Combined confidence: BUY={buy_confidence:.4f}, SELL={sell_confidence:.4f}, Threshold={self.min_confidence:.4f}")
            
            # Determine final action based on highest confidence
            if buy_confidence > sell_confidence and buy_confidence > self.min_confidence:
                action = 'BUY'
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence > self.min_confidence:
                action = 'SELL'
                confidence = sell_confidence
            else:
                action = 'HOLD'
                confidence = max(buy_confidence, sell_confidence)
            
            # Create combined signal
            combined_signal = {
                'pair': pair,
                'action': action,
                'confidence': confidence,
                'reason': f"ICT: {ict_action} ({ict_conf:.2f}), Traditional: {trad_action} ({trad_conf:.2f})",
                'traditional_signal': traditional_signal,
                'ict_signal': ict_signal,
                'timestamp': datetime.now()
            }
            
            # Store the latest signal
            self.latest_signals[pair] = combined_signal
            
            return combined_signal
        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': f"Error: {str(e)}"}
    
    def get_performance_analytics(self, days=30, symbol=None):
        """Get performance analytics from the database
        
        Args:
            days (int, optional): Number of days to analyze. Defaults to 30.
            symbol (str, optional): Filter by symbol. Defaults to None.
            
        Returns:
            dict: Performance analytics data
        """
        try:
            # Get performance metrics
            performance_metrics = self.trade_database.get_performance_metrics(days)
            
            # Get trade statistics
            trade_statistics = self.trade_database.get_trade_statistics(symbol, days)
            
            # Combine data
            analytics = {
                'performance_metrics': performance_metrics,
                'trade_statistics': trade_statistics,
                'account_info': self.get_account_info(),
                'generated_at': datetime.now(),
                'period_days': days
            }
            
            # Log summary
            if performance_metrics and 'total_profit' in performance_metrics:
                logger.info(
                    f"Performance over last {days} days: "
                    f"Profit: ${performance_metrics['total_profit']:.2f} | "
                    f"Win Rate: {performance_metrics['win_rate']*100:.2f}% | "
                    f"Max Drawdown: {performance_metrics['max_drawdown_percent']:.2f}%"
                )
            
            return analytics

        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}
            
    def get_ai_performance_analysis(self, days=30, symbol=None):
        """Get AI-powered performance analysis and recommendations
        
        Uses OpenAI's GPT-4 to analyze trading performance and provide
        recommendations based on ICT methodology.
        
        Args:
            days (int, optional): Number of days to analyze. Defaults to 30.
            symbol (str, optional): Filter by symbol. Defaults to None.
            
        Returns:
            dict: AI analysis and recommendations
        """
        try:
            # Import here to avoid circular imports
            from ai_performance_analyzer import AIPerformanceAnalyzer
            
            # Create analyzer with current config
            analyzer = AIPerformanceAnalyzer(self.config)
            
            # Get analysis
            analysis = analyzer.analyze_performance(days=days, symbol=symbol)
            
            # Log summary
            if 'summary' in analysis:
                logger.info(f"AI Analysis Summary: {analysis['summary']}")
                
                if 'recommendations' in analysis and analysis['recommendations']:
                    logger.info("Top recommendation: " + analysis['recommendations'][0])
            
            return analysis
        except Exception as e:
            logger.error(f"Error getting AI performance analysis: {e}")
            return {
                'error': str(e),
                'recommendations': ['Check if OpenAI API key is set correctly.']
            }

    
    def is_market_open(self):
        """Check if forex market is open"""
        try:
            # Get current UTC time
            now = datetime.now()
            
            # Check if it's weekend (Saturday or Sunday)
            if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                return False
            
            # Check for major holidays (simplified)
            holidays = [
                datetime(now.year, 1, 1),  # New Year's Day
                datetime(now.year, 12, 25),  # Christmas
                datetime(now.year, 12, 26),  # Boxing Day
            ]
            
            if now.date() in [holiday.date() for holiday in holidays]:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking if market is open: {e}")
            return False
    
    def check_economic_events(self, pair, hours_window=None):
        """Check for major economic events affecting the currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            hours_window (int): Hours to look ahead for events
            
        Returns:
            tuple: (bool, list) - Whether high impact event is upcoming and list of event details
        """
        # If economic calendar is not available, return False
        if self.economic_calendar is None:
            return False, []
        
        # Use config value if not specified
        if hours_window is None:
            hours_window = self.config['data'].get('economic_event_hours_window', 24)
            
        # Check for high impact events
        try:
            has_event, events = self.economic_calendar.is_high_impact_event_soon(pair, hours_window)
            
            # Ensure we have a properly formatted list of event dictionaries
            formatted_events = []
            
            if has_event and events is not None:
                # Handle both single event dict and list of events
                if isinstance(events, dict):
                    events = [events]
                    
                for event in events:
                    if isinstance(event, dict):
                        # Create a properly formatted event dictionary
                        formatted_event = {
                            'event': event.get('event', 'Unknown Event'),
                            'date': event.get('date', ''),
                            'pair': pair,
                            'country': event.get('country', 'Unknown'),
                            'impact': event.get('impact', 'Medium'),
                            'forecast': event.get('forecast', 'N/A'),
                            'previous': event.get('previous', 'N/A')
                        }
                        
                        formatted_events.append(formatted_event)
                        
                        # Log the event if it's high impact
                        if formatted_event['impact'].lower() == 'high':
                            logger.info(f"High impact economic event detected for {pair}: {formatted_event['event']} at {formatted_event['date']}")
            
            return (len(formatted_events) > 0, formatted_events)
        except Exception as e:
            logger.error(f"Error checking economic events: {e}")
            return False, []
    
    def calculate_position_size(self, symbol, risk_percent=None, stop_loss_pips=None):
        """Calculate position size based on risk"""
        try:
            # Use config value if not specified
            if risk_percent is None:
                risk_percent = self.config['trading']['risk_management'].get('risk_percent', 1.5)
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.01  # Minimum lot size
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.01  # Minimum lot size
                
            # If stop_loss_pips is not provided, use a default based on ATR
            if stop_loss_pips is None:
                # Get recent data to calculate ATR
                data = self.get_forex_data(symbol)
                if data is not None and len(data) > 14:
                    # Calculate ATR (14-period)
                    data['tr'] = np.maximum(
                        data['high'] - data['low'],
                        np.maximum(
                            abs(data['high'] - data['close'].shift(1)),
                            abs(data['low'] - data['close'].shift(1))
                        )
                    )
                    atr = data['tr'].rolling(14).mean().iloc[-1]
                    
                    # Convert ATR to pips
                    digits = symbol_info.digits
                    point = symbol_info.point
                    multiplier = 10 if digits == 3 or digits == 5 else 1
                    atr_pips = atr / (point * multiplier)
                    
                    # Use 1.5 times ATR for stop loss
                    stop_loss_pips = atr_pips * 1.5
                else:
                    # Default to 50 pips if can't calculate ATR
                    stop_loss_pips = 50        
            # Calculate pip value
            pip_size = 0.0001 if symbol_info.digits == 4 else 0.00001
            contract_size = symbol_info.trade_contract_size
            price = symbol_info.ask
            
            # Calculate risk amount
            balance = account_info.balance
            risk_amount = balance * (risk_percent / 100)
            
            # Calculate position size
            pip_value = pip_size * contract_size
            stop_loss_value = stop_loss_pips * pip_value
            position_size = risk_amount / stop_loss_value
            
            # Convert to lots
            lot_size = position_size / contract_size
            
            # Round to 2 decimal places and ensure minimum lot size
            lot_size = max(0.01, round(lot_size, 2))
            
            return lot_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimum lot size
    
    def execute_trade(self, pair, signal):
        """Execute a trade based on signal with ICT risk management"""
        try:
            # Check if auto trading is enabled
            if not self.auto_trading:
                logger.info("Auto trading is disabled. Not executing trade.")
                return False
            
            # Check if trading is allowed based on balance monitor
            if not self.balance_monitor.can_trade():
                balance_status = self.balance_monitor.get_status()
                logger.warning(f"Trading paused due to drawdown limit ({balance_status['drawdown_limit']:.2f}%) being reached. " + 
                          f"Current drawdown: {balance_status['drawdown_percent']:.2f}%. Not executing trade.")
                return False
            
            # Check if confidence is high enough
            confidence = signal.get('confidence', 0)
            if confidence < self.min_confidence:
                logger.info(f"Signal confidence {confidence} below threshold {self.min_confidence}. Not executing trade.")
                return False
            
            # Check if action is actionable
            action = signal.get('action', 'HOLD')
            if action not in ['BUY', 'SELL']:
                logger.info(f"No actionable signal for {pair}. Current action: {action}")
                return False
            
            # Get account info for risk calculation
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            balance = account_info.balance
            risk_percentage = self.config['trading']['risk_management'].get('risk_percent', 1.5)
            risk_amount = balance * (risk_percentage / 100)
            
            # Get current price
            symbol_info = mt5.symbol_info(pair)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {pair}")
                return False
            
            # Get ATR for dynamic stop loss calculation
            data = self.get_forex_data(pair)
            if data is None:
                logger.error(f"Failed to get data for {pair}")
                return False
            
            # Calculate ATR if not already in data
            if 'atr' not in data.columns:
                tr1 = data['high'] - data['low']
                tr2 = abs(data['high'] - data['close'].shift(1))
                tr3 = abs(data['low'] - data['close'].shift(1))
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr_period = 14
                data['atr'] = tr.rolling(window=atr_period).mean()
            
            # Use ATR for stop loss calculation (1.5 * ATR)
            atr = data['atr'].iloc[-1]
            atr_multiplier = 1.5
            
            if action == 'BUY':
                price = symbol_info.ask
                # Set stop loss based on ATR
                sl = price - (atr * atr_multiplier)
                # Set multiple take profit levels based on risk:reward ratios
                tp1 = price + (price - sl) * 1.5  # R:R = 1.5
                tp2 = price + (price - sl) * 2.75  # R:R = 2.75
                tp3 = price + (price - sl) * 4.75  # R:R = 4.75
            else:  # SELL
                price = symbol_info.bid
                # Set stop loss based on ATR
                sl = price + (atr * atr_multiplier)
                # Set multiple take profit levels based on risk:reward ratios
                tp1 = price - (sl - price) * 1.5  # R:R = 1.5
                tp2 = price - (sl - price) * 2.75  # R:R = 2.75
                tp3 = price - (sl - price) * 4.75  # R:R = 4.75
            
            # Calculate position size (volume)
            point = mt5.symbol_info(pair).point
            pip_value = point * 10  # Standard pip value
            
            # Calculate pips at risk (from entry to stop loss)
            pips_at_risk = abs(price - sl) / pip_value
            
            # Calculate volume based on risk amount and pips at risk
            volume = risk_amount / (pips_at_risk * 10)  # Assuming $10 per pip for 1.0 lot
            
            # Round volume to standard lot size (0.01 lot steps)
            volume = round(volume / 0.01) * 0.01
            
            # Ensure minimum volume
            volume = max(volume, 0.01)
            
            # Split position into three parts for multiple take profits
            volume_part = round((volume / 3) / 0.01) * 0.01
            volume_part = max(volume_part, 0.01)  # Ensure minimum volume
            
            # Execute three separate trades with different take profit levels
            trades_executed = 0
            trade_tickets = []
            
            # First position (1/3 of total) with TP1
            request1 = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": volume_part,
                "type": mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp1,
                "deviation": 20,
                "magic": 234001,
                "comment": f"MCP ICT {action} TP1",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send first trade request
            result1 = mt5.order_send(request1)
            if result1.retcode == mt5.TRADE_RETCODE_DONE:
                trades_executed += 1
                trade_tickets.append(result1.order)
                logger.info(f"Trade 1 executed: {action} {volume_part} lot of {pair} at {price}, TP: {tp1}")
            else:
                logger.error(f"Trade 1 failed: {result1.retcode}, {result1.comment}")
            
            # Second position (1/3 of total) with TP2
            request2 = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": volume_part,
                "type": mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp2,
                "deviation": 20,
                "magic": 234002,
                "comment": f"MCP ICT {action} TP2",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send second trade request
            result2 = mt5.order_send(request2)
            if result2.retcode == mt5.TRADE_RETCODE_DONE:
                trades_executed += 1
                trade_tickets.append(result2.order)
                logger.info(f"Trade 2 executed: {action} {volume_part} lot of {pair} at {price}, TP: {tp2}")
            else:
                logger.error(f"Trade 2 failed: {result2.retcode}, {result2.comment}")
            
            # Third position (1/3 of total) with TP3
            request3 = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": volume_part,
                "type": mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp3,
                "deviation": 20,
                "magic": 234003,
                "comment": f"MCP ICT {action} TP3",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send third trade request
            result3 = mt5.order_send(request3)
            if result3.retcode == mt5.TRADE_RETCODE_DONE:
                trades_executed += 1
                trade_tickets.append(result3.order)
                logger.info(f"Trade 3 executed: {action} {volume_part} lot of {pair} at {price}, TP: {tp3}")
            else:
                logger.error(f"Trade 3 failed: {result3.retcode}, {result3.comment}")
            
            # Store trade data if at least one trade was executed
            if trades_executed > 0:
                trade_data = {
                    'tickets': trade_tickets,
                    'pair': pair,
                    'symbol': pair,  # For database consistency
                    'type': action,
                    'open_time': datetime.now(),
                    'open_price': price,
                    'lot_size': volume_part * trades_executed,
                    'stop_loss': sl,
                    'take_profits': [tp1, tp2, tp3][:trades_executed],
                    'confidence': confidence,
                    'risk_percentage': risk_percentage,
                    'risk_amount': risk_amount,
                    'pips_at_risk': pips_at_risk,
                    'atr': atr,
                    'signal_data': signal
                }
                
                # Add to active trades
                self.active_trades[pair] = trade_data
                
                # Store trade data in database for each ticket
                for i, ticket in enumerate(trade_tickets):
                    db_trade_data = {
                        'ticket': ticket,
                        'symbol': pair,
                        'type': action,
                        'open_time': datetime.now(),
                        'open_price': price,
                        'volume': volume_part,
                        'sl': sl,
                        'tp': [tp1, tp2, tp3][i],
                        'risk_percentage': risk_percentage,
                        'risk_amount': risk_amount / trades_executed,  # Divide by number of trades
                        'pips_at_risk': pips_at_risk,
                        'atr': atr,
                        'magic': [234001, 234002, 234003][i],
                        'confidence': confidence,
                        'status': 'open'
                    }
                    
                    # Store in database
                    self.trade_database.store_trade(db_trade_data)
                    logger.info(f"Trade data stored in database for ticket {ticket}")
            
            # Return True if at least one trade was executed
            return trades_executed > 0
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def get_account_info(self):
        """Get MT5 account information and update balance monitor"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return {}
            
            # Update balance monitor with current balance
            balance_status = self.balance_monitor.update_balance(account_info.balance)
            
            # Create account info dictionary with balance status
            account_info_dict = {
                'name': account_info.name,
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'balance_status': {
                    'drawdown_percent': balance_status['drawdown_percent'],
                    'change_percent': balance_status['change_percent'],
                    'trading_paused': balance_status['trading_paused'],
                    'drawdown_limit': balance_status['drawdown_limit']
                }
            }
            
            # Log balance status
            logger.info(f"Account balance: ${account_info.balance:.2f} | " + 
                       f"Drawdown: {balance_status['drawdown_percent']:.2f}% | " + 
                       f"Change from initial: {balance_status['change_percent']:.2f}%")
            
            if balance_status['trading_paused']:
                logger.warning(f"Trading paused due to drawdown limit ({balance_status['drawdown_limit']:.2f}%) being reached")
            
            return account_info_dict
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def _process_closed_trades(self, closed_tickets):
        """Process closed trades and update database
        
        Args:
            closed_tickets (set): Set of ticket numbers that have been closed
        """
        try:
            # Get history orders for the closed tickets
            for ticket in closed_tickets:
                # Get trade history for this ticket
                history = mt5.history_deals_get(ticket=ticket)
                
                if history is None or len(history) == 0:
                    logger.warning(f"No history found for ticket {ticket}")
                    continue
                
                # Get the closing deal (usually the last one)
                close_deal = history[-1]
                
                # Find the symbol for this ticket
                symbol = None
                for pair, trade_data in self.active_trades.items():
                    if 'tickets' in trade_data and ticket in trade_data['tickets']:
                        symbol = pair
                        break
                
                if symbol is None:
                    logger.warning(f"Could not find symbol for ticket {ticket}")
                    continue
                
                # Get trade details
                close_time = datetime.fromtimestamp(close_deal.time)
                close_price = close_deal.price
                profit = close_deal.profit
                volume = close_deal.volume
                
                # Create trade update data
                trade_update = {
                    'ticket': ticket,
                    'symbol': symbol,
                    'close_time': close_time,
                    'close_price': close_price,
                    'profit': profit,
                    'status': 'closed'
                }
                
                # Update trade in database
                self.trade_database.store_trade(trade_update)
                logger.info(f"Updated closed trade in database: Ticket {ticket}, Profit: ${profit:.2f}")
                
                # Update account info to reflect the trade closure
                self.get_account_info()
                
                # Update performance metrics
                self.trade_database.update_performance_metrics()
        except Exception as e:
            logger.error(f"Error processing closed trades: {e}")
    
    def get_active_trades(self):
        """Get active trades from MT5 and check for closed trades"""
        try:
            # Get current positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            # Get current active tickets
            current_tickets = set([position.ticket for position in positions])
            
            # Get previously stored active trades
            previous_tickets = set()
            for trade_data in self.active_trades.values():
                if 'tickets' in trade_data:
                    previous_tickets.update(trade_data['tickets'])
            
            # Find closed tickets (in previous but not in current)
            closed_tickets = previous_tickets - current_tickets
            
            # Process closed trades
            if closed_tickets:
                self._process_closed_trades(closed_tickets)
            
            # Update active trades dictionary
            active_trades = {}
            for position in positions:
                symbol = position.symbol
                active_trades[symbol] = {
                    'ticket': position.ticket,
                    'type': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'open_time': datetime.fromtimestamp(position.time),
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'profit': position.profit,
                    'lot_size': position.volume,
                    'stop_loss': position.sl,
                    'take_profit': position.tp
                }
            
            return active_trades
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return {}
    
    def is_market_open(self):
        """Check if forex market is open"""
        now = datetime.now()
        day_of_week = now.weekday()
        
        # Weekend check (Saturday = 5, Sunday = 6)
        if day_of_week == 5 or (day_of_week == 6 and now.hour < 22):
            return False
        
        # Sunday evening market open (after 22:00)
        if day_of_week == 6 and now.hour >= 22:
            return True
        
        # Friday market close (before 22:00)
        if day_of_week == 4 and now.hour >= 22:
            return False
        
        # Regular weekday
        return True

# Main function
def main():
    # Create trader instance
    trader = StandaloneTrader()
    
    # Initialize Telegram performance reporter for notifications
    try:
        from src.reporting.telegram_reporter import TelegramReporter
        performance_reporter = TelegramReporter()
        logger.info("Telegram performance reporter initialized. Will send reports every 12 hours.")
    except Exception as e:
        logger.error(f"Failed to initialize performance reporter: {e}")
    
    # Analyze pairs function
    def analyze_pairs():
        print("\n==== ANALYZING CURRENCY PAIRS ====\n")
        for pair in trader.currency_pairs:
            print(f"\nAnalyzing {pair}...")
            
            # Generate signal
            signal = trader.generate_signal(pair)
            
            # Store signal
            trader.latest_signals[pair] = signal
            
            # Print signal information
            print(f"  Action: {signal.get('action', 'UNKNOWN')}")
            print(f"  Confidence: {signal.get('confidence', 0):.2f}")
            print(f"  Reason: {signal.get('reason', 'No reason provided')}")
            
            # Execute trade if auto-trading is enabled
            if trader.auto_trading and signal:
                confidence = signal.get('confidence', 0)
                action = signal.get('action', 'HOLD')
                
                if confidence >= trader.min_confidence and action in ['BUY', 'SELL']:
                    print(f"  Executing {action} trade for {pair} with confidence {confidence:.2f}...")
                    trade_result = trader.execute_trade(pair, signal)
                    if trade_result:
                        print(f"  Trade executed successfully!")
                    else:
                        print(f"  Failed to execute trade.")
                else:
                    print(f"  No trade executed: {'Low confidence' if confidence < trader.min_confidence else 'No actionable signal'}")
    
    # Analyze pairs once
    analyze_pairs()
    
    # Start the terminal trader
    print("\n==== STARTING TERMINAL TRADER ====\n")
    print("Press Ctrl+C to exit")
    
    try:
        # Display the terminal interface
        while True:
            # Clear screen (Windows)
            os.system('cls')
            
            # Display market status
            is_market_open = trader.is_market_open()
            market_status = "OPEN 🟢" if is_market_open else "CLOSED 🔴"
            print("="*80)
            print(f"📊 FOREX MARKET STATUS: {market_status}")
            print("="*80)
            
            # Display MT5 account info
            account_info = trader.get_account_info()
            print("="*80)
            print("✅ MT5 TRADING: CONNECTED")
            print("="*80)
            print(f"Server: {trader.mt5_server}")
            print(f"Login: {trader.mt5_login}")
            print(f"Name: {account_info.get('name', 'Unknown')}")
            print(f"Balance: ${account_info.get('balance', 0):.2f}")
            print(f"Equity: ${account_info.get('equity', 0):.2f}")
            print(f"Profit: ${account_info.get('profit', 0):.2f}")
            print(f"Auto Trading: {'Enabled' if trader.auto_trading else 'Disabled'}")
            
            # Display monitored currency pairs
            print("="*80)
            print("💱 MONITORED CURRENCY PAIRS")
            print("="*80)
            
            # Get all available symbols from MT5
            all_symbols = mt5.symbols_get()
            available_symbols = [symbol.name for symbol in all_symbols] if all_symbols else []
            
            for pair in trader.currency_pairs:
                # Get current price
                try:
                    # Check if symbol is available in MT5
                    if pair not in available_symbols:
                        print(f"- {pair}: ⚠️ Not available in MT5 terminal")
                        continue
                        
                    symbol_info = mt5.symbol_info(pair)
                    if symbol_info:
                        bid = symbol_info.bid
                        ask = symbol_info.ask
                        spread = round((ask - bid) / symbol_info.point, 1)
                        print(f"- {pair}: Bid: {bid:.5f} | Ask: {ask:.5f} | Spread: {spread} points")
                    else:
                        print(f"- {pair}: Price data unavailable")
                except Exception as e:
                    print(f"- {pair}: ⚠️ Error: {str(e)}")
                    logger.error(f"Error getting price for {pair}: {str(e)}")
            
            # Disable pandas warnings during display to prevent garbled output
            import warnings
            pd.options.mode.chained_assignment = None  # Disable chained assignment warnings completely
            warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # Display signals
            print("="*80)
            print(f"🔍 TRADING SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            if not trader.latest_signals:
                print("No trading signals available.")
            else:
                for pair, signal in trader.latest_signals.items():
                    try:
                        # Determine emoji based on action
                        action = signal.get('action', 'HOLD')
                        action_emoji = '🟢' if action == 'BUY' else '🔴' if action == 'SELL' else '⚪'
                        
                        # Get confidence as percentage
                        confidence = signal.get('confidence', 0) * 100
                        
                        # Create a clean output string
                        output = f"{pair}: {action_emoji} {action} ({confidence:.1f}% confidence)\n"
                        
                        # Get reason and make sure it's not too long
                        reason = signal.get('reason', 'No reason provided')
                        if len(reason) > 100:  # Truncate long reasons
                            reason = reason[:97] + '...'
                        output += f"  Reason: {reason}\n"
                        
                        # Display execute information
                        execute_text = '✅ Yes' if signal.get('execute', False) else '❌ No (below threshold)'
                        output += f"  Execute: {execute_text}\n"
                        
                        execute_reason = signal.get('execute_reason', 'No execute reason provided')
                        if len(execute_reason) > 100:  # Truncate long reasons
                            execute_reason = execute_reason[:97] + '...'
                        output += f"  Reason: {execute_reason}\n"
                        
                        # Print the entire output at once to avoid interleaved warnings
                        print(output)
                    except Exception as e:
                        logger.error(f"Error displaying signal for {pair}: {e}")
                    
                    # Display economic event information if available
                    if 'upcoming_event' in signal:
                        event = signal['upcoming_event']
                        event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S')
                        hours_until = (event_time - datetime.now()).total_seconds() / 3600
                        print(f"  ⚠️ ECONOMIC EVENT: {event['event']} in {hours_until:.1f} hours")
                        print(f"     Impact: {event['impact']} | Country: {event['country']}")
                        print(f"     Forecast: {event.get('forecast', 'N/A')} | Previous: {event.get('previous', 'N/A')}")
                    
                    # Display multi-timeframe analysis if available
                    if 'timeframes' in signal:
                        print("  Timeframe Analysis:")
                        # Display timeframe signals (highest timeframe first)
                        for tf_str in ['D1', 'H4', 'H1']:
                            if tf_str in signal['timeframes']:
                                tf_data = signal['timeframes'][tf_str]
                                tf_action = tf_data.get('action', 'NEUTRAL')
                                tf_conf = tf_data.get('confidence', 0)
                                tf_emoji = '🔴' if tf_action == 'SELL' else '🟢' if tf_action == 'BUY' else '⚪'
                                print(f"    {tf_str}: {tf_emoji} {tf_action} (Confidence: {tf_conf:.2f})")
                    
                    # Display signals components
                    signals = signal.get('signals', {})
                    
                    # Show ICT model components if available
                    if 'order_blocks' in signals:
                        # Handle both dict and string formats for backward compatibility
                        if isinstance(signals['order_blocks'], dict):
                            ob_action = signals['order_blocks'].get('action', 'NEUTRAL')
                        else:
                            ob_action = signals['order_blocks']
                        ob_emoji = '🔴' if ob_action == 'SELL' else '🟢' if ob_action == 'BUY' else '⚪'
                        print(f"  Order Blocks: {ob_emoji} {ob_action}")
                    
                    if 'fair_value_gaps' in signals:
                        # Handle both dict and string formats for backward compatibility
                        if isinstance(signals['fair_value_gaps'], dict):
                            fvg_action = signals['fair_value_gaps'].get('action', 'NEUTRAL')
                        else:
                            fvg_action = signals['fair_value_gaps']
                        fvg_emoji = '🔴' if fvg_action == 'SELL' else '🟢' if fvg_action == 'BUY' else '⚪'
                        print(f"  Fair Value Gaps: {fvg_emoji} {fvg_action}")
                    
                    if 'ict_ma_crossover' in signals:
                        # Handle both dict and string formats for backward compatibility
                        if isinstance(signals['ict_ma_crossover'], dict):
                            ma_action = signals['ict_ma_crossover'].get('action', 'NEUTRAL')
                        else:
                            ma_action = signals['ict_ma_crossover']
                        ma_emoji = '🔴' if ma_action == 'SELL' else '🟢' if ma_action == 'BUY' else '⚪'
                        print(f"  ICT MA Crossover: {ma_emoji} {ma_action}")
                    
                    print(f"  Execute: {'✅ Yes' if confidence >= trader.min_confidence else '❌ No (below threshold)'}")
                    print()
            
            # Display upcoming economic events section
            if trader.economic_calendar is not None:
                print("="*80)
                print("📅 UPCOMING ECONOMIC EVENTS")
                print("="*80)
                
                # Check for upcoming events for all monitored pairs
                all_events = []
                for pair in trader.currency_pairs:
                    events = trader.check_economic_events(pair, hours_window=24)
                    if events[0]:  # If there are high impact events
                        all_events.extend(events[1])
                
                if all_events:
                    # Sort by date if it's a list of events
                    try:
                        all_events.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%dT%H:%M:%S'))
                    except Exception as e:
                        logger.warning(f"Error sorting economic events: {e}")
                
                    # Display events (deduplicate by event name and time)
                    displayed_events = set()  # Track events we've already displayed
                    
                    # Build all event outputs first, then print them all at once
                    event_outputs = []
                    
                    for event in all_events:
                        try:
                            # Check if event is a dictionary with required fields
                            if not isinstance(event, dict) or 'event' not in event or 'date' not in event or 'pair' not in event:
                                continue
                                
                            # Create a unique key for this event
                            event_key = f"{event['event']}_{event['date']}_{event['pair']}"
                            
                            # Skip if we've already displayed this event
                            if event_key in displayed_events:
                                continue
                                
                            displayed_events.add(event_key)
                            
                            # Format the event
                            event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S')
                            hours_until = (event_time - datetime.now()).total_seconds() / 3600
                            
                            # Determine impact emoji
                            impact = event.get('impact', 'Medium')
                            impact_emoji = '🔴' if impact == 'High' else '🟡' if impact == 'Medium' else '🟢'
                            
                            # Build the event output string
                            event_output = f"{event_time.strftime('%Y-%m-%d %H:%M')} ({hours_until:.1f}h) - {impact_emoji} {event['event']}\n"
                            event_output += f"  Pair: {event['pair']} | Country: {event.get('country', 'Unknown')} | Impact: {impact}\n"
                            event_output += f"  Forecast: {event.get('forecast', 'N/A')} | Previous: {event.get('previous', 'N/A')}\n"
                            
                            event_outputs.append(event_output)
                        except Exception as e:
                            logger.warning(f"Error formatting economic event: {e}")
                
                    # Print all events at once
                    for output in event_outputs:
                        print(output)
                else:
                    print("No high-impact economic events in the next 24 hours.")
            
            # Display active trades
            print("="*80)
            print("📈 ACTIVE TRADES (ICT MODEL)")
            print("="*80)
            
            active_trades = trader.get_active_trades()
            
            if not active_trades:
                print("No active trades.")
            else:
                for symbol, trade in active_trades.items():
                    try:
                        # Determine emoji based on type
                        type_emoji = '🟢' if trade['type'] == 'BUY' else '🔴'
                        
                        # Build the trade output string
                        trade_output = f"{symbol}: {type_emoji} {trade['type']} @ {trade['open_price']}\n"
                        trade_output += f"  Open Time: {trade['open_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        trade_output += f"  Lot Size: {trade['lot_size']}\n"
                        trade_output += f"  Stop Loss: {trade['stop_loss']}\n"
                        trade_output += f"  Take Profit: {trade['take_profit']}\n"
                        trade_output += f"  Ticket: {trade['ticket']}\n"
                        
                        # Determine profit emoji
                        profit_emoji = '🟢' if trade['profit'] >= 0 else '🔴'
                        trade_output += f"  Current Profit: {profit_emoji} ${trade['profit']:.2f}"
                        
                        # Print the entire output at once to avoid interleaved warnings
                        print(trade_output)
                    except Exception as e:
                        logger.error(f"Error displaying trade for {symbol}: {e}")
                    
                    # Add empty line after trade details
                    print()
            
            # Re-analyze pairs every 5 minutes
            if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
                print("\nRe-analyzing currency pairs...")
                analyze_pairs()
            
            # Sleep for a few seconds
            import time as time_module  # Import with a different name to avoid conflicts
            time_module.sleep(5)
            
    except KeyboardInterrupt:
        print("\nExiting terminal trader...")
    except Exception as e:
        logger.error(f"Error in terminal trader: {e}", exc_info=True)
    finally:
        # Shutdown MT5
        mt5.shutdown()

if __name__ == "__main__":
    main()
