#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for fetching and processing forex data directly from MetaTrader 5"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ForexDataProviderMT5:
    """Provider for forex market data using MetaTrader 5"""
    
    def __init__(self):
        """Initialize the forex data provider using MT5"""
        # Initialize MT5 if not already initialized
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            else:
                logger.info("MT5 initialized for data provider")
        
        # Cache to store data and reduce API calls
        self.data_cache = {}
        self.last_update = {}
        
        # Timeframe mapping
        self.timeframe_map = {
            '1min': mt5.TIMEFRAME_M1,
            '5min': mt5.TIMEFRAME_M5,
            '15min': mt5.TIMEFRAME_M15,
            '30min': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            'daily': mt5.TIMEFRAME_D1,
            '1d': mt5.TIMEFRAME_D1,
            'weekly': mt5.TIMEFRAME_W1,
            '1w': mt5.TIMEFRAME_W1,
            'monthly': mt5.TIMEFRAME_MN1,
            '1mo': mt5.TIMEFRAME_MN1
        }
    
    def get_forex_data(self, pair, interval='1h', output_size='compact'):
        """Get forex data for a specific currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            interval (str): Time interval between data points
            output_size (str): 'compact' for latest 100 data points, 'full' for more data
            
        Returns:
            pandas.DataFrame: DataFrame with forex data or None if an error occurs
        """
        try:
            # Check if we need to update the cache
            current_time = datetime.now()
            if pair in self.last_update:
                time_diff = (current_time - self.last_update[pair]).total_seconds()
                # Only update if more than 60 seconds have passed
                if time_diff < 60:
                    return self.data_cache.get(pair)
            
            # Get the MT5 timeframe
            timeframe = self.timeframe_map.get(interval, mt5.TIMEFRAME_H1)
            
            # Determine number of bars to fetch
            bars = 100 if output_size == 'compact' else 500
            
            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(pair, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get rates for {pair}: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            # Rename columns to match Alpha Vantage format for compatibility
            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            # Update cache
            self.data_cache[pair] = df
            self.last_update[pair] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forex data for {pair}: {e}")
            return None
    
    def get_historical_data(self, pair, start_date=None, end_date=None, interval='1h'):
        """Get historical forex data for a specific currency pair and date range
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            interval (str): Time interval between data points
            
        Returns:
            pandas.DataFrame: DataFrame with historical forex data or None if an error occurs
        """
        try:
            # Parse dates
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            elif start_date is None:
                start_date = datetime.now() - timedelta(days=30)  # Default to last 30 days
                
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            elif end_date is None:
                end_date = datetime.now()
            
            # Get the MT5 timeframe
            timeframe = self.timeframe_map.get(interval, mt5.TIMEFRAME_H1)
            
            # Calculate the number of bars needed
            days_diff = (end_date - start_date).days
            if interval in ['1min', '5min', '15min', '30min']:
                bars = days_diff * 24 * 60 // int(interval.replace('min', ''))
            elif interval in ['1h', '4h']:
                bars = days_diff * 24 // int(interval.replace('h', ''))
            else:  # daily, weekly, monthly
                bars = days_diff
            
            # Limit bars to a reasonable number
            bars = min(bars, 5000)
            
            # Get rates from MT5
            rates = mt5.copy_rates_range(pair, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get historical rates for {pair}: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            # Rename columns to match Alpha Vantage format for compatibility
            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical forex data for {pair}: {e}")
            return None
    
    def get_latest_price(self, pair):
        """Get the latest price for a currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            
        Returns:
            float: Latest price or None if an error occurs
        """
        try:
            # Get the latest tick
            tick = mt5.symbol_info_tick(pair)
            
            if tick is None:
                logger.error(f"Failed to get tick for {pair}: {mt5.last_error()}")
                return None
            
            # Return the ask price (for buying)
            return tick.ask
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {pair}: {e}")
            return None
