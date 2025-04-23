#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Terminal-based Forex Trading Bot
Runs the MCP Forex Trading Bot in terminal-only mode, showing signals and trades
"""

import os
import time
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
from main import ForexTradingBot

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TerminalTrader:
    """Terminal-based Forex Trading Bot"""
    
    def __init__(self):
        """Initialize the terminal trader"""
        self.bot = ForexTradingBot()
        self.running = False
        self.auto_trade = False
        
        # Set auto trade flag if specified in environment
        auto_trade_env = os.getenv('AUTO_TRADE', 'false').lower()
        if auto_trade_env in ['true', '1', 'yes']:
            self.auto_trade = True
            logger.info("Auto trading is enabled")
        else:
            logger.info("Auto trading is disabled (manual mode)")
    
    def start(self):
        """Start the terminal trader"""
        logger.info("Starting Terminal Forex Trading Bot")
        self.running = True
        
        # Initialize MT5 connection
        self.bot._initialize_mt5()
        
        # Start the main loop
        self._main_loop()
    
    def stop(self):
        """Stop the terminal trader"""
        logger.info("Stopping Terminal Forex Trading Bot")
        self.running = False
        self.bot.stop()
    
    def _main_loop(self):
        """Main loop for the terminal trader"""
        try:
            # Start the bot's analysis loop in a separate thread
            analysis_thread = threading.Thread(target=self.bot._analysis_loop, name="AnalysisThread")
            analysis_thread.daemon = True
            analysis_thread.start()
            
            # Print initial status
            self._print_market_status()
            self._print_global_markets()
            self._print_mt5_status()
            self._print_currency_pairs()
            
            # Main loop for displaying signals and executing trades
            while self.running:
                # Print trading signals
                self._print_signals()
                
                # Execute trades if auto trading is enabled
                if self.auto_trade:
                    self._execute_auto_trades()
                
                # Update active trades status
                self._print_active_trades()
                
                # Wait before next update
                time.sleep(60)  # Update every minute
                
        except KeyboardInterrupt:
            logger.info("Terminal trader stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f"Terminal trader stopped due to error: {e}")
            self.stop()
    
    def _print_market_status(self):
        """Print the current forex market status"""
        is_open = self.bot.market_status.is_market_open()
        market_hours_text = self.bot.market_status.get_market_hours_text()
        
        status_text = "OPEN 🟢" if is_open else "CLOSED 🔴"
        print("\n" + "=" * 80)
        print(f"📊 FOREX MARKET STATUS: {status_text}")
        print("=" * 80)
        print(market_hours_text)
    
    def _print_global_markets(self):
        """Print the status of global forex markets"""
        centers_text = self.bot.global_markets.get_centers_status_text()
        print("\n" + "=" * 80)
        print("🌎 GLOBAL FOREX MARKETS")
        print("=" * 80)
        print(centers_text)
    
    def _print_mt5_status(self):
        """Print the MT5 connection status"""
        if not self.bot.mt5_enabled:
            print("\n" + "=" * 80)
            print("❌ MT5 TRADING: DISABLED")
            print("=" * 80)
            print("MetaTrader 5 integration is not enabled.")
            return
        
        account_info = self.bot.get_mt5_account_info()
        if not account_info:
            print("\n" + "=" * 80)
            print("❌ MT5 TRADING: NOT CONNECTED")
            print("=" * 80)
            print("Could not connect to MetaTrader 5.")
            return
        
        print("\n" + "=" * 80)
        print("✅ MT5 TRADING: CONNECTED")
        print("=" * 80)
        print(f"Server: {self.bot.mt5_server}")
        print(f"Login: {self.bot.mt5_login}")
        print(f"Name: {account_info.get('name', 'Unknown')}")
        print(f"Balance: ${account_info.get('balance', 0):.2f}")
        print(f"Equity: ${account_info.get('equity', 0):.2f}")
        print(f"Profit: ${account_info.get('profit', 0):.2f}")
        print(f"Auto Trading: {'Enabled' if self.auto_trade else 'Disabled'}")
    
    def _print_currency_pairs(self):
        """Print the currency pairs being monitored"""
        print("\n" + "=" * 80)
        print("💱 MONITORED CURRENCY PAIRS")
        print("=" * 80)
        for pair in self.bot.currency_pairs:
            print(f"- {pair}")
    
    def _print_signals(self):
        """Print the latest trading signals"""
        print("\n" + "=" * 80)
        print("🔍 TRADING SIGNALS - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)
        
        if not self.bot.latest_signals:
            print("No trading signals available.")
            return
        
        for pair, signal in self.bot.latest_signals.items():
            if not signal:
                continue
                
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0)
            reason = signal.get("reason", "")
            timestamp = signal.get("timestamp", datetime.now())
            
            # Format the signal output
            if action == "BUY":
                action_text = "🟢 BUY"
            elif action == "SELL":
                action_text = "🔴 SELL"
            else:
                action_text = "⚪ HOLD"
            
            print(f"Pair: {pair}")
            print(f"Signal: {action_text}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Reason: {reason}")
            print(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
    
    def _execute_auto_trades(self):
        """Execute trades automatically based on signals"""
        if not self.bot.mt5_enabled:
            return
        
        for pair, signal in self.bot.latest_signals.items():
            if not signal:
                continue
                
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0)
            
            # Skip if action is HOLD or confidence is below threshold
            if action == "HOLD" or confidence < self.bot.min_confidence:
                continue
            
            # Skip if we already have an active trade for this pair
            if pair in self.bot.active_trades:
                continue
            
            # Execute the trade
            logger.info(f"Auto-executing {action} trade for {pair} with confidence {confidence:.2f}")
            result = self.bot._execute_trade(pair, signal)
            
            if result:
                logger.info(f"Successfully executed {action} trade for {pair}")
            else:
                logger.error(f"Failed to execute {action} trade for {pair}")
    
    def _print_active_trades(self):
        """Print active trades"""
        if not self.bot.mt5_enabled:
            return
        
        active_trades = self.bot.get_active_trades()
        
        print("\n" + "=" * 80)
        print("📈 ACTIVE TRADES")
        print("=" * 80)
        
        if not active_trades:
            print("No active trades.")
            return
        
        for pair, trade in active_trades.items():
            print(f"Pair: {pair}")
            print(f"Type: {trade.get('type', 'Unknown')}")
            print(f"Open Price: {trade.get('open_price', 0):.5f}")
            print(f"Current Price: {trade.get('current_price', 0):.5f}")
            print(f"Profit: ${trade.get('profit', 0):.2f}")
            print(f"Open Time: {trade.get('open_time', '')}")
            print("-" * 40)

if __name__ == "__main__":
    try:
        terminal = TerminalTrader()
        terminal.start()
    except KeyboardInterrupt:
        logger.info("Terminal trader stopped by user")
    except Exception as e:
        logger.error(f"Terminal trader stopped due to error: {e}")
