#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MongoDB database manager for the Forex Trading Bot"""

import os
import logging
from datetime import datetime
import pymongo
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseManager:
    """MongoDB database manager for storing trade history and bot data"""
    
    def __init__(self):
        """Initialize the database manager"""
        self.mongo_uri = os.getenv('MONGODB_URI')
        self.client = None
        self.db = None
        self.trades_collection = None
        self.signals_collection = None
        self.performance_collection = None
        
        # Connect to MongoDB
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            if not self.mongo_uri:
                logger.warning("MongoDB URI not found in .env file")
                return False
            
            self.client = pymongo.MongoClient(self.mongo_uri)
            self.db = self.client.get_database()
            
            # Create collections if they don't exist
            self.trades_collection = self.db['trades']
            self.signals_collection = self.db['signals']
            self.performance_collection = self.db['performance']
            
            # Create indexes
            self.trades_collection.create_index([('ticket', pymongo.ASCENDING)], unique=True)
            self.trades_collection.create_index([('open_time', pymongo.DESCENDING)])
            self.signals_collection.create_index([('timestamp', pymongo.DESCENDING)])
            
            logger.info("Connected to MongoDB successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False
    
    def save_trade(self, trade_data):
        """Save trade data to MongoDB
        
        Args:
            trade_data (dict): Trade data to save
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if self.trades_collection is None:
                logger.error("MongoDB not connected. Cannot save trade.")
                return False
            
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now()
            
            # Insert or update trade
            result = self.trades_collection.update_one(
                {'ticket': trade_data['ticket']},
                {'$set': trade_data},
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error saving trade to MongoDB: {e}")
            return False
    
    def update_trade(self, ticket, update_data):
        """Update an existing trade in MongoDB
        
        Args:
            ticket (int): Trade ticket number
            update_data (dict): Data to update
        
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            if self.trades_collection is None:
                logger.error("MongoDB not connected. Cannot update trade.")
                return False
            
            # Add last updated timestamp
            update_data['last_updated'] = datetime.now()
            
            # Update trade
            result = self.trades_collection.update_one(
                {'ticket': ticket},
                {'$set': update_data}
            )
            
            return result.acknowledged and result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating trade in MongoDB: {e}")
            return False
    
    def close_trade(self, ticket, close_data):
        """Mark a trade as closed in MongoDB
        
        Args:
            ticket (int): Trade ticket number
            close_data (dict): Closing data (price, profit, etc.)
        
        Returns:
            bool: True if closed successfully, False otherwise
        """
        try:
            if self.trades_collection is None:
                logger.error("MongoDB not connected. Cannot close trade.")
                return False
            
            # Add closing timestamp
            close_data['close_time'] = datetime.now()
            close_data['status'] = 'closed'
            
            # Update trade
            result = self.trades_collection.update_one(
                {'ticket': ticket},
                {'$set': close_data}
            )
            
            return result.acknowledged and result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error closing trade in MongoDB: {e}")
            return False
    
    def get_trade_history(self, limit=50, status=None):
        """Get trade history from MongoDB
        
        Args:
            limit (int): Maximum number of trades to return
            status (str, optional): Filter by trade status ('open', 'closed')
        
        Returns:
            list: List of trades
        """
        try:
            if self.trades_collection is None:
                logger.error("MongoDB not connected. Cannot get trade history.")
                return []
            
            # Build query
            query = {}
            if status:
                query['status'] = status
            
            # Get trades
            trades = list(self.trades_collection.find(
                query,
                sort=[('open_time', pymongo.DESCENDING)],
                limit=limit
            ))
            
            # Convert ObjectId to string for JSON serialization
            for trade in trades:
                if '_id' in trade:
                    trade['_id'] = str(trade['_id'])
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history from MongoDB: {e}")
            return []
    
    def save_signal(self, signal_data):
        """Save trading signal to MongoDB
        
        Args:
            signal_data (dict): Signal data to save
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if self.signals_collection is None:
                logger.error("MongoDB not connected. Cannot save signal.")
                return False
            
            # Add timestamp if not present
            if 'timestamp' not in signal_data:
                signal_data['timestamp'] = datetime.now()
            
            # Insert signal
            result = self.signals_collection.insert_one(signal_data)
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error saving signal to MongoDB: {e}")
            return False
    
    def get_recent_signals(self, limit=20):
        """Get recent trading signals from MongoDB
        
        Args:
            limit (int): Maximum number of signals to return
        
        Returns:
            list: List of signals
        """
        try:
            if self.signals_collection is None:
                logger.error("MongoDB not connected. Cannot get signals.")
                return []
            
            # Get signals
            signals = list(self.signals_collection.find(
                {},
                sort=[('timestamp', pymongo.DESCENDING)],
                limit=limit
            ))
            
            # Convert ObjectId to string for JSON serialization
            for signal in signals:
                if '_id' in signal:
                    signal['_id'] = str(signal['_id'])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals from MongoDB: {e}")
            return []
    
    def save_performance_metrics(self, metrics):
        """Save performance metrics to MongoDB
        
        Args:
            metrics (dict): Performance metrics to save
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if self.performance_collection is None:
                logger.error("MongoDB not connected. Cannot save performance metrics.")
                return False
            
            # Add timestamp
            metrics['timestamp'] = datetime.now()
            
            # Insert metrics
            result = self.performance_collection.insert_one(metrics)
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error saving performance metrics to MongoDB: {e}")
            return False
    
    def get_performance_summary(self):
        """Get performance summary from MongoDB
        
        Returns:
            dict: Performance summary
        """
        try:
            if self.trades_collection is None:
                logger.error("MongoDB not connected. Cannot get performance summary.")
                return {}
            
            # Get all closed trades
            closed_trades = list(self.trades_collection.find({'status': 'closed'}))
            
            if not closed_trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'profit_factor': 0
                }
            
            # Calculate metrics
            total_trades = len(closed_trades)
            winning_trades = sum(1 for trade in closed_trades if trade.get('profit', 0) > 0)
            losing_trades = sum(1 for trade in closed_trades if trade.get('profit', 0) < 0)
            
            total_profit = sum(trade.get('profit', 0) for trade in closed_trades)
            total_profit_wins = sum(trade.get('profit', 0) for trade in closed_trades if trade.get('profit', 0) > 0)
            total_profit_losses = sum(abs(trade.get('profit', 0)) for trade in closed_trades if trade.get('profit', 0) < 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            average_profit = total_profit_wins / winning_trades if winning_trades > 0 else 0
            average_loss = total_profit_losses / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit_wins / total_profit_losses if total_profit_losses > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary from MongoDB: {e}")
            return {}
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
