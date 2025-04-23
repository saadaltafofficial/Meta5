#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check available symbols in MT5

This script connects to MT5 and lists all available symbols
"""

import os
import sys
import logging
from dotenv import load_dotenv
import MetaTrader5 as mt5

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/.env'))

def main():
    # MT5 parameters from environment
    mt5_server = os.getenv('MT5_SERVER')
    mt5_login = os.getenv('MT5_LOGIN')
    mt5_password = os.getenv('MT5_PASSWORD')
    
    print(f"Connecting to MT5 server: {mt5_server} with login: {mt5_login}")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    # Connect to MT5 account
    authorized = mt5.login(int(mt5_login), mt5_password, server=mt5_server)
    if not authorized:
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    print("MT5 connection successful!")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Account: {account_info.login} ({account_info.name})")
        print(f"Balance: ${account_info.balance:.2f}")
        print(f"Equity: ${account_info.equity:.2f}")
    
    # Get all available symbols
    symbols = mt5.symbols_get()
    if symbols:
        print(f"\nTotal symbols available: {len(symbols)}")
        print("\nForex pairs:")
        forex_pairs = [s.name for s in symbols if s.name.endswith('USD') or 
                      (len(s.name) == 6 and s.name[:3] in ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY'])]
        for pair in sorted(forex_pairs):
            print(f"- {pair}")
        
        # Check specific pairs from config
        print("\nChecking specific pairs from config:")
        config_pairs = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY", "XAUUSD"]
        for pair in config_pairs:
            if pair in forex_pairs or any(s.name == pair for s in symbols):
                symbol_info = mt5.symbol_info(pair)
                if symbol_info:
                    print(f"✅ {pair}: Available - Bid: {symbol_info.bid:.5f}, Ask: {symbol_info.ask:.5f}")
                else:
                    print(f"⚠️ {pair}: Symbol info not available")
            else:
                print(f"❌ {pair}: Not available in this MT5 terminal")
    else:
        print(f"Failed to get symbols: {mt5.last_error()}")
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nMT5 connection closed.")

if __name__ == "__main__":
    main()
