#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MetaTrader 5 integration for the Forex Trading Bot"""

import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MT5Trader:
    """Class for MetaTrader 5 trading integration"""
    
    def __init__(self, account_type="demo", server=None, login=None, password=None):
        """Initialize the MT5 trader
        
        Args:
            account_type (str): Account type ('demo' or 'real')
            server (str): MT5 server name
            login (int): MT5 account login
            password (str): MT5 account password
        """
        self.account_type = account_type
        self.server = server
        self.login = login
        self.password = password
        self.connected = False
        self.initialized = False
        self.account_info = None
        
        # Trading parameters
        self.default_lot_size = 0.01  # Micro lot
        self.max_risk_percent = 2.0   # Maximum risk per trade (2%)
        self.default_stop_loss_pips = 50
        self.default_take_profit_pips = 100
        
        # Initialize MT5
        self._initialize_mt5()
    
    def _initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        logger.info("Initializing MetaTrader 5 connection")
        
        # Check if MT5 is already initialized
        if mt5.terminal_info() is not None:
            logger.info("MetaTrader 5 is already running")
            self.initialized = True
        else:
            # Initialize MT5
            if not mt5.initialize():
                error_code, error_message = mt5.last_error()
                logger.error(f"MT5 initialization failed with error code: {error_code}, message: {error_message}")
                
                # Provide more helpful error messages
                if error_code == -6:
                    logger.error("Authorization failed. This could be because MetaTrader 5 is already running with a different account.")
                    logger.error("Please close all instances of MetaTrader 5 and try again, or log in manually and keep it running.")
                    logger.error("The bot will attempt to use the existing MT5 instance.")
                    self.initialized = True  # Try to use existing instance anyway
                return
            
            self.initialized = True
            logger.info("MetaTrader 5 initialized successfully")
        
        # Display MT5 terminal info
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is not None:
                terminal_info_dict = mt5.terminal_info()._asdict()
                logger.info(f"MT5 Terminal: {terminal_info_dict.get('name', 'Unknown')} - {terminal_info_dict.get('path', 'Unknown')}")
                if 'version' in terminal_info_dict:
                    logger.info(f"MT5 Version: {terminal_info_dict['version']}")
        except Exception as e:
            logger.warning(f"Could not get terminal info: {e}")
        
        # Connect to account if credentials provided
        if self.login and self.password and self.server:
            self.connect_account()
    
    def connect_account(self, server=None, login=None, password=None):
        """Connect to MT5 account
        
        Args:
            server (str, optional): MT5 server name
            login (int, optional): MT5 account login
            password (str, optional): MT5 account password
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not self.initialized:
            logger.error("MT5 not initialized. Cannot connect to account.")
            return False
        
        # Update credentials if provided
        if server:
            self.server = server
        if login:
            self.login = login
        if password:
            self.password = password
        
        # Check if we have all required credentials
        if not (self.server and self.login and self.password):
            logger.error("Missing MT5 account credentials. Cannot connect.")
            return False
        
        # Connect to account
        logger.info(f"Connecting to MT5 {self.account_type} account {self.login} on server {self.server}")
        
        # Try to convert login to integer if it's a string
        try:
            login_id = int(self.login)
        except ValueError:
            logger.error(f"Invalid login ID format: {self.login}. Must be a number.")
            self.connected = False
            return False
            
        # Skip server checking as servers_for is not available in this MT5 version
        logger.info(f"Attempting to connect to server {self.server} with login {self.login}")
        
        # Check if already connected to an account
        account_info = mt5.account_info()
        if account_info is not None:
            # Already connected to some account
            current_login = account_info._asdict().get('login', 0)
            if current_login == login_id:
                logger.info(f"Already connected to account {login_id}")
                self.connected = True
                self.account_info = account_info._asdict()
                return True
            else:
                logger.warning(f"Already connected to a different account: {current_login}. Will try to reconnect.")
        
        # Try to connect
        connected = mt5.login(login_id, self.password, self.server)
        
        if not connected:
            error_code = mt5.last_error()
            logger.error(f"MT5 login failed with error code: {error_code}")
            
            # More detailed error messages
            if error_code == 10000:
                logger.error("No error returned, but connection failed. Check if server name is correct.")
            elif error_code == 10001:
                logger.error("Connection to trading server failed. Check internet connection and server name.")
            elif error_code == 10002:
                logger.error("Invalid login or password. Check credentials.")
            elif error_code == 10003:
                logger.error("Invalid server. Check server name.")
            elif error_code == 10004:
                logger.error("Timeout error. Check internet connection.")
            elif error_code == 10006:
                logger.error("Account disabled. Contact broker.")
                
            # Check if we're already connected despite the error
            account_info = mt5.account_info()
            if account_info is not None:
                logger.info("Connected to MT5, but with a different account. Will try to use it anyway.")
                self.connected = True
                self.account_info = account_info._asdict()
                return True
            
            self.connected = False
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info. Error: {mt5.last_error()}")
            self.connected = False
            return False
            
        self.account_info = account_info._asdict()
        self.connected = True
        
        # Log detailed account information
        logger.info(f"Connected to MT5 account: {self.account_info['name']} (#{self.account_info['login']})")
        logger.info(f"Account details: Balance={self.account_info['balance']} {self.account_info['currency']}, " + 
                   f"Equity={self.account_info['equity']}, " + 
                   f"Margin={self.account_info['margin']}, " + 
                   f"Free Margin={self.account_info['margin_free']}")
        
        # Check available symbols
        symbols = self.get_symbols()
        logger.info(f"Available symbols: {len(symbols)}")
        logger.info(f"Sample forex pairs: {[s for s in symbols if 'USD' in s][:5]}")
        
        return True
    
    def create_demo_account(self, server, name="MCP Trader", leverage=100):
        """Create a new demo account
        
        Args:
            server (str): MT5 server name
            name (str): Account holder name
            leverage (int): Account leverage
            
        Returns:
            dict: New account info or None if failed
        """
        if not self.initialized:
            logger.error("MT5 not initialized. Cannot create demo account.")
            return None
        
        logger.info(f"Creating new MT5 demo account on server {server}")
        
        # Create demo account
        account = mt5.account_new(server, name, "", "", "demo", leverage)
        
        if account is None:
            logger.error(f"Failed to create demo account: {mt5.last_error()}")
            return None
        
        logger.info(f"Demo account created: Login={account.login}, Password={account.password}, Server={server}")
        
        # Connect to the new account
        self.login = account.login
        self.password = account.password
        self.server = server
        self.connect_account()
        
        return {
            "login": account.login,
            "password": account.password,
            "server": server,
            "leverage": leverage
        }
    
    def get_account_info(self):
        """Get current account information
        
        Returns:
            dict: Account information
        """
        if not self.connected:
            logger.warning("Not connected to MT5 account. Cannot get account info.")
            return None
        
        # Update account info
        self.account_info = mt5.account_info()._asdict()
        return self.account_info
    
    def get_symbols(self):
        """Get available symbols
        
        Returns:
            list: Available symbols
        """
        if not self.initialized:
            logger.warning("MT5 not initialized. Cannot get symbols.")
            return []
        
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]
    
    def get_forex_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, bars=500):
        """Get historical forex data from MT5
        
        Args:
            symbol (str): Symbol name (e.g., 'EURUSD')
            timeframe (int): MT5 timeframe constant
            bars (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.initialized:
            logger.warning("MT5 not initialized. Cannot get forex data.")
            return None
        
        # Prepare the symbol
        symbol = self._prepare_symbol(symbol)
        
        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data received for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to match the bot's expected format
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        return df
    
    def _prepare_symbol(self, symbol):
        """Prepare symbol for MT5 (add suffix if needed)
        
        Args:
            symbol (str): Symbol name (e.g., 'EURUSD')
            
        Returns:
            str: Prepared symbol
        """
        # Check if the symbol exists as is
        if symbol in self.get_symbols():
            return symbol
        
        # Try common suffixes
        for suffix in ['', '.', '-', '_', 'm']:
            test_symbol = f"{symbol}{suffix}"
            if test_symbol in self.get_symbols():
                return test_symbol
        
        # If no match found, return original and let MT5 handle the error
        logger.warning(f"Symbol {symbol} not found in MT5 symbols list")
        return symbol
    
    def calculate_position_size(self, symbol, risk_percent=None, stop_loss_pips=None):
        """Calculate position size based on risk percentage
        
        Args:
            symbol (str): Symbol name
            risk_percent (float, optional): Risk percentage (1.0 = 1%)
            stop_loss_pips (int, optional): Stop loss in pips
            
        Returns:
            float: Position size in lots
        """
        if not self.connected:
            logger.warning("Not connected to MT5 account. Using default lot size.")
            return self.default_lot_size
        
        # Use default values if not provided
        risk_percent = risk_percent if risk_percent is not None else self.max_risk_percent
        stop_loss_pips = stop_loss_pips if stop_loss_pips is not None else self.default_stop_loss_pips
        
        # Get account balance
        balance = self.account_info['balance']
        
        # Calculate risk amount
        risk_amount = balance * (risk_percent / 100.0)
        
        # Get symbol info
        symbol = self._prepare_symbol(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            logger.warning(f"Symbol {symbol} not found. Using default lot size.")
            return self.default_lot_size
        
        # Get pip value
        digits = symbol_info.digits
        point = symbol_info.point
        pip_value = point * 10 if digits == 3 or digits == 5 else point
        
        # Calculate pip cost
        contract_size = symbol_info.trade_contract_size
        price = mt5.symbol_info_tick(symbol).ask
        
        # For USD account with USD quote currency
        if self.account_info['currency'] == 'USD' and symbol[-3:] == 'USD':
            pip_cost = pip_value * contract_size
        # For USD account with non-USD quote currency
        elif self.account_info['currency'] == 'USD':
            quote_currency = symbol[-3:]
            conversion_symbol = f"{quote_currency}USD"
            
            # Try to get conversion rate
            conversion_rate = 1.0
            conversion_info = mt5.symbol_info_tick(conversion_symbol)
            if conversion_info is not None:
                conversion_rate = conversion_info.bid
            
            pip_cost = pip_value * contract_size * conversion_rate
        # For non-USD account
        else:
            # This is a simplified calculation
            pip_cost = pip_value * contract_size / price
        
        # Calculate lot size
        if pip_cost > 0 and stop_loss_pips > 0:
            lot_size = risk_amount / (pip_cost * stop_loss_pips)
        else:
            lot_size = self.default_lot_size
        
        # Round to standard lot sizes
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Ensure lot size is within limits
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        
        logger.info(f"Calculated position size for {symbol}: {lot_size} lots (Risk: {risk_percent}%, SL: {stop_loss_pips} pips)")
        return lot_size
    
    def open_trade(self, symbol, order_type, lot_size=None, price=None, stop_loss=None, take_profit=None, comment="MCP Trader"):
        """Open a new trade
        
        Args:
            symbol (str): Symbol name
            order_type (str): Order type ('BUY' or 'SELL')
            lot_size (float, optional): Position size in lots
            price (float, optional): Order price (market price if None)
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            comment (str, optional): Order comment
            
        Returns:
            dict: Order result or None if failed
        """
        if not self.connected:
            logger.warning("Not connected to MT5 account. Cannot open trade.")
            return None
            
        # Prepare symbol
        symbol = self._prepare_symbol(symbol)
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found. Cannot open trade.")
            return None
        
        # Enable symbol for trading if needed
        if not symbol_info.visible:
            logger.info(f"Symbol {symbol} is not visible, enabling...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        
        # Determine order type
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        current_price = tick.ask if mt5_order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Use calculated lot size if not provided
        if lot_size is None:
            lot_size = self.calculate_position_size(symbol)
        
        # Calculate stop loss and take profit if not provided
        point = symbol_info.point
        digits = symbol_info.digits
        pip_size = point * 10 if digits == 3 or digits == 5 else point
        
        if stop_loss is None and self.default_stop_loss_pips > 0:
            stop_loss_offset = self.default_stop_loss_pips * pip_size
            stop_loss = current_price - stop_loss_offset if mt5_order_type == mt5.ORDER_TYPE_BUY else current_price + stop_loss_offset
        
        if take_profit is None and self.default_take_profit_pips > 0:
            take_profit_offset = self.default_take_profit_pips * pip_size
            take_profit = current_price + take_profit_offset if mt5_order_type == mt5.ORDER_TYPE_BUY else current_price - take_profit_offset
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": current_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,  # Maximum price deviation in points
            "magic": 123456,  # Magic number for identifying bot orders
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Fill or kill
        }
        
        # Send order
        logger.info(f"Sending {order_type} order for {symbol}: {lot_size} lots at {current_price}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order executed: Ticket #{result.order}")
        return {
            "ticket": result.order,
            "symbol": symbol,
            "type": order_type,
            "volume": lot_size,
            "price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "comment": comment
        }
    
    def close_trade(self, ticket):
        """Close an existing trade
        
        Args:
            ticket (int): Order ticket
            
        Returns:
            bool: True if closed successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to MT5 account. Cannot close trade.")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position #{ticket} not found")
            return False
        
        position = position[0]._asdict()
        
        # Determine order type (opposite of position type)
        close_type = mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get current price
        symbol = position['symbol']
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position['volume'],
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "MCP Trader - Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        logger.info(f"Closing position #{ticket} for {symbol}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.retcode} - {result.comment}")
            return False
        
        logger.info(f"Position #{ticket} closed successfully")
        return True
    
    def get_open_positions(self):
        """Get all open positions
        
        Returns:
            list: Open positions
        """
        if not self.connected:
            logger.warning("Not connected to MT5 account. Cannot get positions.")
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            logger.warning(f"No positions found or error: {mt5.last_error()}")
            return []
        
        # Convert to list of dictionaries
        result = []
        for position in positions:
            pos_dict = position._asdict()
            result.append({
                "ticket": pos_dict['ticket'],
                "symbol": pos_dict['symbol'],
                "type": "BUY" if pos_dict['type'] == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos_dict['volume'],
                "open_price": pos_dict['price_open'],
                "current_price": pos_dict['price_current'],
                "stop_loss": pos_dict['sl'],
                "take_profit": pos_dict['tp'],
                "profit": pos_dict['profit'],
                "comment": pos_dict['comment'],
                "time": datetime.fromtimestamp(pos_dict['time']).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return result
    
    def modify_trade(self, ticket, stop_loss=None, take_profit=None):
        """Modify an existing trade's stop loss and take profit
        
        Args:
            ticket (int): Order ticket
            stop_loss (float, optional): New stop loss price
            take_profit (float, optional): New take profit price
            
        Returns:
            bool: True if modified successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to MT5 account. Cannot modify trade.")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position #{ticket} not found")
            return False
        
        position = position[0]._asdict()
        
        # Use existing values if not provided
        if stop_loss is None:
            stop_loss = position['sl']
        if take_profit is None:
            take_profit = position['tp']
        
        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position['symbol'],
            "position": ticket,
            "sl": stop_loss,
            "tp": take_profit
        }
        
        # Send order
        logger.info(f"Modifying position #{ticket} SL: {stop_loss}, TP: {take_profit}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify position: {result.retcode} - {result.comment}")
            return False
        
        logger.info(f"Position #{ticket} modified successfully")
        return True
    
    def close_all_positions(self):
        """Close all open positions
        
        Returns:
            int: Number of positions closed
        """
        if not self.connected:
            logger.error("Not connected to MT5 account. Cannot close positions.")
            return 0
        
        positions = self.get_open_positions()
        closed_count = 0
        
        for position in positions:
            if self.close_trade(position['ticket']):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} of {len(positions)} positions")
        return closed_count
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            logger.info("Shutting down MT5 connection")
            mt5.shutdown()
            self.initialized = False
            self.connected = False

# Example usage
if __name__ == "__main__":
    # Create MT5 trader instance
    mt5_trader = MT5Trader()
    
    # Create demo account (replace with your broker's server)
    account_info = mt5_trader.create_demo_account("MetaQuotes-Demo")
    
    if account_info:
        print(f"Demo account created: {account_info}")
        
        # Get forex data
        data = mt5_trader.get_forex_data("EURUSD")
        print(f"EURUSD data: {data.tail()}")
        
        # Open a trade
        trade = mt5_trader.open_trade("EURUSD", "BUY", 0.01)
        if trade:
            print(f"Trade opened: {trade}")
            
            # Wait a bit
            time.sleep(5)
            
            # Close the trade
            mt5_trader.close_trade(trade['ticket'])
    
    # Shutdown
    mt5_trader.shutdown()
