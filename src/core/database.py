#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Module for MCP Trader

This module handles database operations for storing trade history and performance analytics
"""

import os
import logging
import pymongo
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TradeDatabase:
    """Handles database operations for trade history and performance analytics"""
    
    def __init__(self, config=None):
        """Initialize the trade database
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.db_client = None
        self.db = None
        self.trades_collection = None
        self.performance_collection = None
        self.connected = False
        
        # Connect to database
        self._connect_to_db()
    
    def _connect_to_db(self):
        """Connect to MongoDB database"""
        try:
            # Get MongoDB URI from environment variables
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                logger.warning("MongoDB URI not found in environment variables. Using local storage instead.")
                return False
            
            # Connect to MongoDB
            self.db_client = pymongo.MongoClient(mongodb_uri)
            self.db = self.db_client.mt5
            
            # Create collections if they don't exist
            self.trades_collection = self.db.trades
            self.performance_collection = self.db.performance
            
            # Create indexes
            self.trades_collection.create_index([('ticket', pymongo.ASCENDING)], unique=True)
            self.trades_collection.create_index([('symbol', pymongo.ASCENDING)])
            self.trades_collection.create_index([('open_time', pymongo.ASCENDING)])
            
            self.performance_collection.create_index([('date', pymongo.ASCENDING)], unique=True)
            
            # Test connection
            self.db_client.admin.command('ping')
            
            logger.info("Connected to MongoDB database")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def store_trade(self, trade_data):
        """Store trade data in the database
        
        Args:
            trade_data (dict): Trade data to store
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database. Cannot store trade.")
                return False
            
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now()
            
            # Check if trade already exists (by ticket)
            if 'ticket' in trade_data:
                existing_trade = self.trades_collection.find_one({'ticket': trade_data['ticket']})
                if existing_trade:
                    # Update existing trade
                    self.trades_collection.update_one(
                        {'ticket': trade_data['ticket']},
                        {'$set': trade_data}
                    )
                    logger.info(f"Updated trade {trade_data['ticket']} in database")
                    return True
            
            # Insert new trade
            result = self.trades_collection.insert_one(trade_data)
            logger.info(f"Stored trade in database with ID: {result.inserted_id}")
            
            # Update performance metrics after storing trade
            self.update_performance_metrics()
            
            return True
        except Exception as e:
            logger.error(f"Error storing trade in database: {e}")
            return False
    
    def get_trades(self, symbol=None, start_date=None, end_date=None, limit=100):
        """Get trades from the database
        
        Args:
            symbol (str, optional): Filter by symbol
            start_date (datetime, optional): Filter by start date
            end_date (datetime, optional): Filter by end date
            limit (int, optional): Maximum number of trades to return
            
        Returns:
            list: List of trades
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database. Cannot get trades.")
                return []
            
            # Build query
            query = {}
            if symbol:
                query['symbol'] = symbol
            
            if start_date or end_date:
                query['open_time'] = {}
                if start_date:
                    query['open_time']['$gte'] = start_date
                if end_date:
                    query['open_time']['$lte'] = end_date
            
            # Get trades
            trades = list(self.trades_collection.find(query).sort('open_time', pymongo.DESCENDING).limit(limit))
            return trades
        except Exception as e:
            logger.error(f"Error getting trades from database: {e}")
            return []
    
    def update_performance_metrics(self):
        """Calculate and update performance metrics"""
        try:
            if not self.connected:
                logger.warning("Not connected to database. Cannot update performance metrics.")
                return False
            
            # Get all closed trades
            closed_trades = list(self.trades_collection.find({'close_time': {'$exists': True}}))
            if not closed_trades:
                logger.info("No closed trades found. Cannot calculate performance metrics.")
                return False
            
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(closed_trades)
            
            # Calculate daily performance
            trades_df['date'] = pd.to_datetime(trades_df['close_time']).dt.date
            daily_performance = trades_df.groupby('date').agg({
                'profit': 'sum',
                'ticket': 'count'
            }).reset_index()
            daily_performance.rename(columns={'ticket': 'trades_count'}, inplace=True)
            
            # Calculate cumulative metrics
            daily_performance['cumulative_profit'] = daily_performance['profit'].cumsum()
            
            # Calculate win rate
            trades_df['is_win'] = trades_df['profit'] > 0
            win_rate = trades_df.groupby('date')['is_win'].mean().reset_index()
            win_rate.rename(columns={'is_win': 'win_rate'}, inplace=True)
            
            # Merge win rate with daily performance
            daily_performance = pd.merge(daily_performance, win_rate, on='date', how='left')
            
            # Calculate drawdown
            daily_performance['high_water_mark'] = daily_performance['cumulative_profit'].cummax()
            daily_performance['drawdown'] = daily_performance['high_water_mark'] - daily_performance['cumulative_profit']
            daily_performance['drawdown_percent'] = (daily_performance['drawdown'] / daily_performance['high_water_mark']) * 100
            
            # Calculate additional metrics
            daily_performance['avg_profit'] = daily_performance['profit'] / daily_performance['trades_count']
            
            # Store performance metrics in database
            for _, row in daily_performance.iterrows():
                performance_data = {
                    'date': datetime.combine(row['date'], datetime.min.time()),
                    'profit': float(row['profit']),
                    'trades_count': int(row['trades_count']),
                    'cumulative_profit': float(row['cumulative_profit']),
                    'win_rate': float(row['win_rate']),
                    'drawdown': float(row['drawdown']),
                    'drawdown_percent': float(row['drawdown_percent']),
                    'avg_profit': float(row['avg_profit']),
                    'updated_at': datetime.now()
                }
                
                # Check if performance data already exists for this date
                existing_performance = self.performance_collection.find_one({'date': performance_data['date']})
                if existing_performance:
                    # Update existing performance data
                    self.performance_collection.update_one(
                        {'date': performance_data['date']},
                        {'$set': performance_data}
                    )
                else:
                    # Insert new performance data
                    self.performance_collection.insert_one(performance_data)
            
            logger.info(f"Updated performance metrics for {len(daily_performance)} days")
            return True
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            return False
    
    def get_performance_metrics(self, days=30):
        """Get performance metrics from the database
        
        Args:
            days (int, optional): Number of days to get metrics for
            
        Returns:
            dict: Performance metrics
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database. Cannot get performance metrics.")
                return {}
            
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Get performance metrics
            metrics = list(self.performance_collection.find(
                {'date': {'$gte': start_date}}
            ).sort('date', pymongo.ASCENDING))
            
            if not metrics:
                logger.info(f"No performance metrics found for the last {days} days.")
                return {}
            
            # Calculate summary metrics
            total_profit = sum(metric['profit'] for metric in metrics)
            total_trades = sum(metric['trades_count'] for metric in metrics)
            avg_win_rate = np.mean([metric['win_rate'] for metric in metrics])
            max_drawdown = max([metric['drawdown_percent'] for metric in metrics])
            
            # Calculate Sharpe ratio (if we have enough data)
            if len(metrics) > 1:
                daily_returns = [metric['profit'] for metric in metrics]
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            summary = {
                'total_profit': total_profit,
                'total_trades': total_trades,
                'win_rate': avg_win_rate,
                'max_drawdown_percent': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'period_days': days,
                'daily_metrics': metrics
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_trade_statistics(self, symbol=None, days=30):
        """Get detailed trade statistics
        
        Args:
            symbol (str, optional): Filter by symbol
            days (int, optional): Number of days to get statistics for
            
        Returns:
            dict: Trade statistics
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database. Cannot get trade statistics.")
                return {}
            
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Build query
            query = {'close_time': {'$gte': start_date}}
            if symbol:
                query['symbol'] = symbol
            
            # Get closed trades
            trades = list(self.trades_collection.find(query))
            if not trades:
                logger.info(f"No trades found for the last {days} days.")
                return {}
            
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            
            # Calculate statistics
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if losing_trades['profit'].sum() != 0 else float('inf')
            
            # Calculate by symbol
            by_symbol = trades_df.groupby('symbol').agg({
                'profit': ['sum', 'mean', 'count'],
                'is_win': 'mean'
            }).reset_index()
            
            # Calculate by day of week
            trades_df['day_of_week'] = pd.to_datetime(trades_df['close_time']).dt.day_name()
            by_day = trades_df.groupby('day_of_week').agg({
                'profit': ['sum', 'mean', 'count'],
                'is_win': 'mean'
            }).reset_index()
            
            # Calculate by hour of day
            trades_df['hour'] = pd.to_datetime(trades_df['close_time']).dt.hour
            by_hour = trades_df.groupby('hour').agg({
                'profit': ['sum', 'mean', 'count'],
                'is_win': 'mean'
            }).reset_index()
            
            statistics = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor),
                'total_profit': float(trades_df['profit'].sum()),
                'by_symbol': by_symbol.to_dict(),
                'by_day': by_day.to_dict(),
                'by_hour': by_hour.to_dict(),
                'period_days': days
            }
            
            return statistics
        except Exception as e:
            logger.error(f"Error getting trade statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        try:
            if self.db_client:
                self.db_client.close()
                logger.info("Closed database connection")
                self.connected = False
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
