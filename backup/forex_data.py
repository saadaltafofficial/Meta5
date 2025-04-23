#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for fetching and processing forex data"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ForexDataProvider:
    """Provider for forex market data"""
    
    def __init__(self):
        """Initialize the forex data provider"""
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found in .env file")
        
        # Cache to store data and reduce API calls
        self.data_cache = {}
        self.last_update = {}
    
    def get_forex_data(self, pair, interval='1min', output_size='compact'):
        """Get forex data for a specific currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            interval (str): Time interval between data points
            output_size (str): 'compact' for latest 100 data points, 'full' for up to 20 years of data
            
        Returns:
            pandas.DataFrame: DataFrame with forex data or None if an error occurs
        """
        try:
            # Check if we need to update the cache
            current_time = datetime.now(pytz.UTC)
            if pair in self.last_update:
                time_diff = (current_time - self.last_update[pair]).total_seconds()
                # Only update if more than 60 seconds have passed
                if time_diff < 60:
                    return self.data_cache.get(pair)
            
            # Format the currency pair for Alpha Vantage
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            # Alpha Vantage API endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "interval": interval,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors in the response
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if "Information" in data and "call frequency" in data["Information"]:
                logger.warning(f"API call frequency limit reached: {data['Information']}")
                # Return cached data if available
                return self.data_cache.get(pair)
            
            # Parse the response
            time_series_key = f"Time Series FX ({interval})"
            if time_series_key not in data:
                logger.error(f"Unexpected API response format: {data}")
                return None
            
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame(time_series).T
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add datetime index
            df.index = pd.to_datetime(df.index)
            
            # Sort by datetime
            df = df.sort_index()
            
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
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min', 'daily')
            
        Returns:
            pandas.DataFrame: DataFrame with historical forex data or None if an error occurs
        """
        try:
            # Format the currency pair for Alpha Vantage
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            # Convert dates to datetime objects if provided as strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # If no end date is provided, use current date
            if end_date is None:
                end_date = datetime.now()
                
            # If no start date is provided, use 30 days before end date
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Calculate days between start and end date for outputsize parameter
            days_diff = (end_date - start_date).days + 1
            
            # Map interval to Alpha Vantage format
            av_function = "FX_INTRADAY"
            av_interval = interval
            
            if interval == '1h' or interval == '60min':
                av_interval = '60min'
            elif interval == 'daily' or interval == '1d':
                av_function = "FX_DAILY"
                av_interval = None
            
            # Alpha Vantage API endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": av_function,
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": "full" if days_diff > 100 else "compact",
                "apikey": self.api_key
            }
            
            # Add interval parameter for intraday data
            if av_interval:
                params["interval"] = av_interval
            
            # Make the API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors in the response
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            # Parse the response - key depends on function and interval
            if av_function == "FX_DAILY":
                time_series_key = "Time Series FX (Daily)"
            else:  # FX_INTRADAY
                time_series_key = f"Time Series FX ({av_interval})"
                
            if time_series_key not in data:
                logger.error(f"Unexpected API response format: {data}")
                return None
            
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame(time_series).T
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add datetime index
            df.index = pd.to_datetime(df.index)
            
            # Sort by datetime
            df = df.sort_index()
            
            # Filter to the requested date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
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
            # Format the currency pair for Alpha Vantage
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            # Alpha Vantage API endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors in the response
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            # Parse the response
            result_key = "Realtime Currency Exchange Rate"
            if result_key not in data:
                logger.error(f"Unexpected API response format: {data}")
                return None
            
            # Get the exchange rate
            exchange_rate = float(data[result_key]["5. Exchange Rate"])
            
            return exchange_rate
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {pair}: {e}")
            return None
