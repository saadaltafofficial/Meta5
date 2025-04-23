import os
import time
import logging
import threading
import requests
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai
import json
import sys

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('telegram_reporter')

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../config/.env'))

class TelegramReporter:
    def __init__(self, auto_start=False):
        # Telegram settings
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # OpenAI API settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Report settings
        self.send_reports = os.getenv('SEND_PERFORMANCE_REPORTS', 'true').lower() == 'true'
        self.report_interval_hours = int(os.getenv('REPORT_INTERVAL_HOURS', 12))
        
        # Initialize MT5 connection status
        self.mt5_connected = False
        
        # Validate Telegram settings
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram settings not found in .env file. Performance reports will not be sent.")
            self.send_reports = False
        
        # Start the reporting thread if enabled and auto_start is True
        if self.send_reports and auto_start:
            self.start_reporting_thread()
    
    def connect_to_mt5(self):
        """Connect to MetaTrader 5 platform"""
        if not mt5.initialize():
            logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            return False
        
        # Login to MT5
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"Failed to login to MT5: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        logger.info("Successfully connected to MT5")
        self.mt5_connected = True
        return True
    
    def get_account_info(self):
        """Get account information from MT5"""
        if not self.mt5_connected:
            if not self.connect_to_mt5():
                return None
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_level': account_info.margin_level,
            'margin_free': account_info.margin_free
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        if not self.mt5_connected:
            if not self.connect_to_mt5():
                return []
        
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return []
        
        position_list = []
        for position in positions:
            position_list.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': 'BUY' if position.type == 0 else 'SELL',
                'volume': position.volume,
                'open_price': position.price_open,
                'current_price': position.price_current,
                'profit': position.profit,
                'open_time': datetime.fromtimestamp(position.time).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return position_list
    
    def get_closed_positions(self, days=1):
        """Get closed positions from the last X days"""
        if not self.mt5_connected:
            if not self.connect_to_mt5():
                return []
        
        # Calculate the start time (X days ago)
        from_date = datetime.now() - timedelta(days=days)
        from_timestamp = int(from_date.timestamp())
        
        # Get the history
        history = mt5.history_deals_get(from_timestamp)
        if history is None:
            logger.error(f"Failed to get history: {mt5.last_error()}")
            return []
        
        # Process the history to get closed positions
        closed_positions = []
        for deal in history:
            if deal.entry == 1:  # Exit position
                closed_positions.append({
                    'ticket': deal.position_id,
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': deal.volume,
                    'profit': deal.profit,
                    'close_time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return closed_positions
    
    def generate_performance_report(self):
        """Generate a concise performance report for ICT trading bot"""
        # Get account info and positions
        account_info = self.get_account_info()
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions(days=1)
        
        if account_info is None:
            return "Failed to generate performance report: Could not connect to MT5."
        
        # Calculate performance metrics
        total_open_profit = sum(position['profit'] for position in open_positions)
        total_closed_profit = sum(position['profit'] for position in closed_positions)
        total_profit = total_open_profit + total_closed_profit
        
        # Calculate equity change percentage
        equity_change_pct = 0
        if account_info['balance'] > 0:
            equity_change_pct = ((account_info['equity'] - account_info['balance']) / account_info['balance']) * 100
        
        # Create a simple report string
        report = "*ICT BOT REPORT*\n"
        report += f"Bal: ${account_info['balance']:.2f} | Equity: ${account_info['equity']:.2f} ({equity_change_pct:+.2f}%)\n"
        
        # Add margin info (important for ICT traders)
        report += f"Margin: ${account_info['margin']:.2f} | Free: ${account_info['margin_free']:.2f}\n"
        
        # Add open positions with ICT-relevant details
        if open_positions:
            report += "\n*OPEN POSITIONS:*\n"
            for pos in open_positions:
                emoji = '🟢' if pos['profit'] > 0 else '🔴'
                # Calculate R-multiple (profit in terms of risk)
                r_multiple = 0
                if 'volume' in pos and pos['volume'] > 0:
                    # Estimate R-multiple based on standard ICT risk (1.5%)
                    estimated_risk = account_info['balance'] * 0.015  # 1.5% risk per ICT methodology
                    if estimated_risk > 0:
                        r_multiple = pos['profit'] / estimated_risk
                
                report += f"{emoji} {pos['symbol']} {pos['type']}: ${pos['profit']:.2f} ({r_multiple:+.1f}R)\n"
        
        # Add closed positions from last 24h
        if closed_positions:
            report += "\n*CLOSED (24h):*\n"
            for pos in closed_positions[:3]:  # Show max 3 recent closed positions
                emoji = '🟢' if pos['profit'] > 0 else '🔴'
                report += f"{emoji} {pos['symbol']} {pos['type']}: ${pos['profit']:.2f}\n"
            
            if len(closed_positions) > 3:
                report += f"...and {len(closed_positions)-3} more\n"
        
        # Check if we're in a Pakistan trading session
        current_hour = datetime.now().hour
        session_name = ""
        if 14 <= current_hour <= 19:  # London-NY overlap (7-12pm PKT)
            session_name = "London-NY Session"
        elif 19 <= current_hour <= 21:  # Super Prime (7-9pm PKT)
            session_name = "🔥 SUPER PRIME"
        elif 1 <= current_hour <= 5:  # Asian Session (1-5am PKT)
            session_name = "Asian Session"
        
        # Add summary line with session info if applicable
        summary = f"Total P/L: ${total_profit:.2f} | {datetime.now().strftime('%H:%M')} PKT"
        if session_name:
            summary += f" | {session_name}"
            
        report += f"\n\n{summary}"
        
        return report
    
    def send_telegram_message(self, message):
        """Send a message via Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.error("Telegram settings not configured. Cannot send report.")
            return False
        
        try:
            # Send the message via Telegram API
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload)
            result = response.json()
            
            if result.get('ok'):
                logger.info(f"Performance report sent to Telegram chat {self.telegram_chat_id}")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {result}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_performance_report(self):
        """Generate and send a performance report"""
        logger.info("Generating performance report...")
        report = self.generate_performance_report()
        
        # Format the report for Telegram (with markdown)
        header = f"""*ICT TRADING BOT - PERFORMANCE REPORT*
{'-' * 30}
*Report Time:* {datetime.now().strftime('%Y-%m-%d %H:%M')} (Pakistan Time)
{'-' * 30}

"""
        
        # Send the report via Telegram
        telegram_message = header + report
        self.send_telegram_message(telegram_message)
    
    def reporting_loop(self):
        """Main reporting loop that runs in a separate thread"""
        logger.info(f"Performance reporter started. Will send reports every {self.report_interval_hours} hours.")
        
        while True:
            try:
                # Send a performance report
                self.send_performance_report()
                
                # Sleep until the next report time
                sleep_seconds = self.report_interval_hours * 3600
                logger.info(f"Next report will be sent in {self.report_interval_hours} hours.")
                time.sleep(sleep_seconds)
                
            except Exception as e:
                logger.error(f"Error in reporting loop: {e}")
                # Sleep for a while before retrying
                time.sleep(300)
    
    def start_reporting_thread(self):
        """Start the reporting thread"""
        report_thread = threading.Thread(target=self.reporting_loop, daemon=True)
        report_thread.start()
        logger.info("Performance reporting thread started")

# For testing
if __name__ == "__main__":
    # When running as a script, we have two options:
    # 1. Just send a test message and exit
    # 2. Start the reporting thread and keep running
    
    import sys
    
    # Check if we should just run a test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Create reporter without auto-starting the thread
        reporter = TelegramReporter(auto_start=False)
        
        # Just send a test message
        report = reporter.generate_performance_report()
        print("Sending test report to Telegram...")
        
        # Format the report for Telegram (with markdown)
        header = f"""*ICT TRADING BOT - TEST REPORT*
{'-' * 30}
*Report Time:* {datetime.now().strftime('%Y-%m-%d %H:%M')} (Pakistan Time)
{'-' * 30}

"""
        
        # Send the report via Telegram
        telegram_message = header + report
        success = reporter.send_telegram_message(telegram_message)
        
        if success:
            print("\n✅ Test report sent successfully to Telegram!")
        else:
            print("\n❌ Failed to send test report to Telegram.")
    else:
        # Create reporter with auto-start enabled
        reporter = TelegramReporter(auto_start=True)
        
        # Keep the script running
        print("Telegram reporter is running. Reports will be sent every", reporter.report_interval_hours, "hours.")
        print("Press Ctrl+C to exit.")
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nTelegram reporter stopped.")
