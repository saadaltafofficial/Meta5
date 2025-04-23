#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Balance Monitor for ICT Trader

This module monitors account balance and provides risk management functions
based on ICT risk management principles.
"""

import logging
import json
import os
import sys
from datetime import datetime

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalanceMonitor:
    """Monitors account balance and provides risk management functions"""
    
    def __init__(self, config=None):
        """Initialize the balance monitor
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.balance_history = []
        self.initial_balance = self._get_initial_balance()
        self.highest_balance = self.initial_balance
        self.current_balance = self.initial_balance
        self.drawdown_limit = self._get_drawdown_limit()
        self.trading_paused = False
        
        # Load balance history if available
        self._load_balance_history()
        
        logger.info(f"Balance monitor initialized with initial balance: ${self.initial_balance:.2f}")
        logger.info(f"Drawdown limit: {self.drawdown_limit:.2f}%")
    
    def _get_initial_balance(self):
        """Get the initial balance from configuration
        
        Returns:
            float: Initial balance
        """
        if self.config and 'trading' in self.config and 'risk_management' in self.config['trading']:
            if 'balance_monitoring' in self.config['trading']['risk_management']:
                return self.config['trading']['risk_management']['balance_monitoring'].get('initial_balance', 100000)
        return 100000  # Default initial balance
    
    def _get_drawdown_limit(self):
        """Get the drawdown limit from configuration
        
        Returns:
            float: Drawdown limit percentage
        """
        if self.config and 'trading' in self.config and 'risk_management' in self.config['trading']:
            if 'balance_monitoring' in self.config['trading']['risk_management']:
                return self.config['trading']['risk_management']['balance_monitoring'].get('drawdown_limit_percent', 5.0)
        return 5.0  # Default drawdown limit
    
    def _should_pause_trading(self):
        """Check if trading should be paused based on configuration
        
        Returns:
            bool: Whether trading should be paused on drawdown
        """
        if self.config and 'trading' in self.config and 'risk_management' in self.config['trading']:
            if 'balance_monitoring' in self.config['trading']['risk_management']:
                return self.config['trading']['risk_management']['balance_monitoring'].get('pause_trading_on_drawdown', True)
        return True  # Default to pausing trading on drawdown
    
    def _load_balance_history(self):
        """Load balance history from file"""
        try:
            # Use the new data directory for balance history
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
            history_file = os.path.join(data_dir, 'balance_history.json')
            
            # Create data directory if it doesn't exist
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            if os.path.exists(history_file):
                # Check file size first - if it's extremely large, reset it completely
                file_size_mb = os.path.getsize(history_file) / (1024 * 1024)  # Size in MB
                if file_size_mb > 5:  # If larger than 5MB, reset it
                    logger.warning(f"Balance history file is very large ({file_size_mb:.2f}MB). Resetting to prevent storage issues.")
                    self.balance_history = []
                    self._save_balance_history()
                    return
                
                # Load the file if it's a reasonable size
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.balance_history = data.get('history', [])
                    
                    # Truncate to latest 100 entries if needed
                    if len(self.balance_history) > 100:
                        logger.info(f"Truncating balance history from {len(self.balance_history)} to 100 entries")
                        self.balance_history = self.balance_history[-100:]
                        # Save the truncated history immediately
                        self._save_balance_history()
                    
                    # Update highest balance
                    if self.balance_history:
                        balances = [entry['balance'] for entry in self.balance_history]
                        self.highest_balance = max(balances)
                        self.current_balance = self.balance_history[-1]['balance']
                        
                    logger.info(f"Loaded balance history with {len(self.balance_history)} entries")
                    logger.info(f"Highest recorded balance: ${self.highest_balance:.2f}")
                    logger.info(f"Current balance: ${self.current_balance:.2f}")
        except Exception as e:
            logger.error(f"Error loading balance history: {e}")
    
    def _save_balance_history(self):
        """Save balance history to file"""
        try:
            # Use the new data directory for balance history
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
            history_file = os.path.join(data_dir, 'balance_history.json')
            
            # Create data directory if it doesn't exist
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            with open(history_file, 'w') as f:
                json.dump({'history': self.balance_history}, f, indent=2)
            logger.info(f"Saved balance history with {len(self.balance_history)} entries")
        except Exception as e:
            logger.error(f"Error saving balance history: {e}")
    
    def update_balance(self, new_balance):
        """Update the current balance and check for drawdown
        
        Args:
            new_balance (float): New account balance
            
        Returns:
            dict: Status update with drawdown information
        """
        # Update current balance
        self.current_balance = new_balance
        
        # Update highest balance if current balance is higher
        if new_balance > self.highest_balance:
            self.highest_balance = new_balance
            logger.info(f"New highest balance: ${self.highest_balance:.2f}")
        
        # Get previous balance (if available)
        previous_balance = self.initial_balance
        if self.balance_history:
            previous_balance = self.balance_history[-1]['balance']
        
        # Calculate drawdown from highest balance
        drawdown_amount = self.highest_balance - new_balance
        drawdown_percent = (drawdown_amount / self.highest_balance) * 100 if self.highest_balance > 0 else 0
        
        # Calculate change from initial balance
        change_from_initial = new_balance - self.initial_balance
        change_percent = (change_from_initial / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Calculate change from previous balance
        daily_change = new_balance - previous_balance
        daily_change_percent = (daily_change / previous_balance) * 100 if previous_balance > 0 else 0
        
        # Create balance history entry
        balance_entry = {
            'timestamp': datetime.now().isoformat(),
            'balance': new_balance,
            'drawdown_percent': drawdown_percent,
            'change_from_initial_percent': change_percent,
            'daily_change_percent': daily_change_percent
        }
        
        # Only add to balance history and save if:
        # 1. Balance has changed significantly (more than 0.01%)
        # 2. It's been at least 15 minutes since the last entry
        # 3. The balance history is empty
        
        significant_change = False
        time_threshold_met = False
        
        # Check for significant change
        if abs(daily_change_percent) > 0.01:
            significant_change = True
            
        # Check time threshold
        if not self.balance_history:
            time_threshold_met = True  # Always add first entry
        else:
            last_entry_time = datetime.fromisoformat(self.balance_history[-1]['timestamp'])
            time_since_last = (datetime.now() - last_entry_time).total_seconds() / 60  # minutes
            if time_since_last >= 15:  # 15 minutes threshold
                time_threshold_met = True
        
        # Add to history and save only if conditions are met
        if significant_change or time_threshold_met or not self.balance_history:
            self.balance_history.append(balance_entry)
            # Only keep the last 100 entries to limit file size
            if len(self.balance_history) > 100:
                self.balance_history = self.balance_history[-100:]
            # Save to file
            self._save_balance_history()
        
        # Check if we should pause trading due to drawdown
        should_pause = self._should_pause_trading()
        if should_pause and drawdown_percent >= self.drawdown_limit:
            if not self.trading_paused:
                logger.warning(f"Drawdown limit reached: {drawdown_percent:.2f}% > {self.drawdown_limit:.2f}%")
                logger.warning(f"Trading paused until balance recovers")
                self.trading_paused = True
        elif self.trading_paused and drawdown_percent < self.drawdown_limit:
            logger.info(f"Balance recovered, drawdown now {drawdown_percent:.2f}% < {self.drawdown_limit:.2f}%")
            logger.info(f"Trading resumed")
            self.trading_paused = False
        
        # Prepare status update
        status = {
            'current_balance': new_balance,
            'initial_balance': self.initial_balance,
            'highest_balance': self.highest_balance,
            'drawdown_amount': drawdown_amount,
            'drawdown_percent': drawdown_percent,
            'change_from_initial': change_from_initial,
            'change_percent': change_percent,
            'daily_change': daily_change,
            'daily_change_percent': daily_change_percent,
            'trading_paused': self.trading_paused,
            'drawdown_limit': self.drawdown_limit
        }
        
        return status
    
    def can_trade(self):
        """Check if trading is allowed based on balance monitoring
        
        Returns:
            bool: Whether trading is allowed
        """
        # If balance monitoring is not enabled, always allow trading
        if not self.config.get('trading', {}).get('risk_management', {}).get('balance_monitoring', {}).get('enabled', False):
            return True
        
        # If trading is paused due to drawdown, don't allow trading
        return not self.trading_paused
    
    def get_status(self):
        """Get the current balance status
        
        Returns:
            dict: Current balance status
        """
        # Calculate drawdown from highest balance
        drawdown_amount = self.highest_balance - self.current_balance
        drawdown_percent = (drawdown_amount / self.highest_balance) * 100 if self.highest_balance > 0 else 0
        
        # Calculate change from initial balance
        change_from_initial = self.current_balance - self.initial_balance
        change_percent = (change_from_initial / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'highest_balance': self.highest_balance,
            'drawdown_amount': drawdown_amount,
            'drawdown_percent': drawdown_percent,
            'change_from_initial': change_from_initial,
            'change_percent': change_percent,
            'trading_paused': self.trading_paused,
            'drawdown_limit': self.drawdown_limit,
            'history_entries': len(self.balance_history)
        }
