#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for performing technical analysis on forex data"""

import logging
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ict_model import ICTModel

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Class for performing technical analysis on forex data"""
    
    def __init__(self):
        """Initialize the technical analyzer"""
        # Initialize the ICT model
        self.ict_model = ICTModel()
    
    def analyze(self, data, pair):
        """Perform technical analysis on forex data
        
        Args:
            data (pandas.DataFrame): DataFrame with forex data
            pair (str): Currency pair being analyzed
            
        Returns:
            dict: Dictionary with analysis results and trading signals
        """
        if data is None or data.empty:
            logger.warning(f"No data available for {pair}")
            return {
                'pair': pair,
                'timestamp': pd.Timestamp.now(),
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'No data available',
                'indicators': {}
            }
        
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate trading signals based on the indicators
            signals = self._generate_signals(indicators)
            
            # Perform ICT model analysis
            ict_analysis = self.ict_model.analyze(df, pair)
            
            # Integrate ICT signals with traditional technical analysis
            if ict_analysis.get('valid', False):
                signals['ict'] = ict_analysis.get('signals', {}).get('action', 'NEUTRAL')
                # Add ICT indicators to our indicators dictionary
                indicators['ict'] = {
                    'market_structure': ict_analysis.get('market_structure', {}),
                    'key_levels': ict_analysis.get('key_levels', {}),
                    'confidence': ict_analysis.get('confidence', 0)
                }
            
            # Determine the overall action (BUY, SELL, HOLD)
            action, confidence, reason = self._determine_action(signals)
            
            # Get the latest price
            latest_price = df['close'].iloc[-1] if not df.empty else None
            
            # Create the result dictionary
            result = {
                'pair': pair,
                'timestamp': pd.Timestamp.now(),
                'price': latest_price,
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'indicators': indicators,
                'signals': signals
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
            return {
                'pair': pair,
                'timestamp': pd.Timestamp.now(),
                'action': 'HOLD',
                'confidence': 0,
                'reason': f'Error during analysis: {str(e)}',
                'indicators': {}
            }
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with forex data
            
        Returns:
            dict: Dictionary with calculated indicators
        """
        indicators = {}
        
        # MACD
        macd = MACD(
            close=df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        indicators['macd'] = {
            'macd': macd.macd().iloc[-1],
            'signal': macd.macd_signal().iloc[-1],
            'histogram': macd.macd_diff().iloc[-1]
        }
        
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        indicators['rsi'] = rsi.rsi().iloc[-1]
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
        indicators['bollinger_bands'] = {
            'upper': bollinger.bollinger_hband().iloc[-1],
            'middle': bollinger.bollinger_mavg().iloc[-1],
            'lower': bollinger.bollinger_lband().iloc[-1],
            'width': bollinger.bollinger_wband().iloc[-1]
        }
        
        # Moving Averages
        sma_short = SMAIndicator(close=df['close'], window=10)
        sma_long = SMAIndicator(close=df['close'], window=50)
        ema_short = EMAIndicator(close=df['close'], window=10)
        ema_long = EMAIndicator(close=df['close'], window=50)
        
        indicators['moving_averages'] = {
            'sma_short': sma_short.sma_indicator().iloc[-1],
            'sma_long': sma_long.sma_indicator().iloc[-1],
            'ema_short': ema_short.ema_indicator().iloc[-1],
            'ema_long': ema_long.ema_indicator().iloc[-1]
        }
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        indicators['stochastic'] = {
            'k': stoch.stoch().iloc[-1],
            'd': stoch.stoch_signal().iloc[-1]
        }
        
        # Calculate price changes
        indicators['price_change'] = {
            'pct_change_1m': df['close'].pct_change(1).iloc[-1] * 100 if len(df) > 1 else 0,
            'pct_change_5m': df['close'].pct_change(5).iloc[-1] * 100 if len(df) > 5 else 0,
            'pct_change_15m': df['close'].pct_change(15).iloc[-1] * 100 if len(df) > 15 else 0
        }
        
        return indicators
    
    def _generate_signals(self, indicators):
        """Generate trading signals based on technical indicators
        
        Args:
            indicators (dict): Dictionary with calculated indicators
            
        Returns:
            dict: Dictionary with trading signals
        """
        signals = {}
        
        # MACD Signal
        macd = indicators['macd']
        if macd['macd'] > macd['signal']:
            signals['macd'] = 'BUY' if macd['histogram'] > 0 else 'WEAK BUY'
        else:
            signals['macd'] = 'SELL' if macd['histogram'] < 0 else 'WEAK SELL'
        
        # RSI Signal
        rsi = indicators['rsi']
        if rsi < 30:
            signals['rsi'] = 'BUY'  # Oversold
        elif rsi > 70:
            signals['rsi'] = 'SELL'  # Overbought
        else:
            signals['rsi'] = 'NEUTRAL'
        
        # Bollinger Bands Signal
        bb = indicators['bollinger_bands']
        price = indicators.get('price', bb['middle'])  # Use middle if price not available
        
        if price > bb['upper']:
            signals['bollinger'] = 'SELL'  # Price above upper band
        elif price < bb['lower']:
            signals['bollinger'] = 'BUY'  # Price below lower band
        else:
            signals['bollinger'] = 'NEUTRAL'
        
        # Moving Averages Signal
        ma = indicators['moving_averages']
        
        # SMA Crossover
        if ma['sma_short'] > ma['sma_long']:
            signals['sma_crossover'] = 'BUY'
        else:
            signals['sma_crossover'] = 'SELL'
        
        # EMA Crossover
        if ma['ema_short'] > ma['ema_long']:
            signals['ema_crossover'] = 'BUY'
        else:
            signals['ema_crossover'] = 'SELL'
        
        # Stochastic Oscillator Signal
        stoch = indicators['stochastic']
        
        if stoch['k'] < 20 and stoch['d'] < 20:
            signals['stochastic'] = 'BUY'  # Oversold
        elif stoch['k'] > 80 and stoch['d'] > 80:
            signals['stochastic'] = 'SELL'  # Overbought
        elif stoch['k'] > stoch['d']:
            signals['stochastic'] = 'WEAK BUY'  # %K crosses above %D
        elif stoch['k'] < stoch['d']:
            signals['stochastic'] = 'WEAK SELL'  # %K crosses below %D
        else:
            signals['stochastic'] = 'NEUTRAL'
        
        return signals
    
    def _determine_action(self, signals):
        """Determine the overall action based on trading signals
        
        Args:
            signals (dict): Dictionary with trading signals
            
        Returns:
            tuple: (action, confidence, reason)
        """
        # Count the number of buy and sell signals
        buy_count = sum(1 for signal in signals.values() if signal == 'BUY')
        weak_buy_count = sum(1 for signal in signals.values() if signal == 'WEAK BUY')
        sell_count = sum(1 for signal in signals.values() if signal == 'SELL')
        weak_sell_count = sum(1 for signal in signals.values() if signal == 'WEAK SELL')
        neutral_count = sum(1 for signal in signals.values() if signal == 'NEUTRAL')
        
        # Give extra weight to ICT model signals if present
        ict_signal = signals.get('ict')
        if ict_signal == 'BUY':
            buy_count += 2  # ICT BUY signal counts as 2 strong buy signals
        elif ict_signal == 'WEAK BUY':
            weak_buy_count += 2  # ICT WEAK BUY signal counts as 2 weak buy signals
        elif ict_signal == 'SELL':
            sell_count += 2  # ICT SELL signal counts as 2 strong sell signals
        elif ict_signal == 'WEAK SELL':
            weak_sell_count += 2  # ICT WEAK SELL signal counts as 2 weak sell signals
        
        # Calculate total signals
        total_signals = len(signals)
        
        # Calculate buy and sell strength
        buy_strength = (buy_count * 1.0 + weak_buy_count * 0.5) / total_signals
        sell_strength = (sell_count * 1.0 + weak_sell_count * 0.5) / total_signals
        
        # Determine the action
        if buy_strength > sell_strength:
            if buy_strength > 0.6:
                action = 'BUY'
                confidence = min(buy_strength * 100, 100)
                reason = f"Strong buy signals ({buy_count} strong, {weak_buy_count} weak)"
            else:
                action = 'WEAK BUY'
                confidence = min(buy_strength * 100, 100)
                reason = f"Moderate buy signals ({buy_count} strong, {weak_buy_count} weak)"
        elif sell_strength > buy_strength:
            if sell_strength > 0.6:
                action = 'SELL'
                confidence = min(sell_strength * 100, 100)
                reason = f"Strong sell signals ({sell_count} strong, {weak_sell_count} weak)"
            else:
                action = 'WEAK SELL'
                confidence = min(sell_strength * 100, 100)
                reason = f"Moderate sell signals ({sell_count} strong, {weak_sell_count} weak)"
        else:
            action = 'HOLD'
            confidence = max(neutral_count / total_signals * 100, 50)
            reason = "Mixed or neutral signals"
        
        return action, confidence, reason
