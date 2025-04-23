#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main entry point for the Forex Algorithmic Trading Bot"""

import os
import time
import logging
import threading
import asyncio
from dotenv import load_dotenv

# Import custom modules
from forex_data import ForexDataProvider
from forex_data_mt5 import ForexDataProviderMT5
from technical_analysis import TechnicalAnalyzer
from market_status import MarketStatus
from news_analyzer import NewsAnalyzer
from global_markets import GlobalMarkets
from mt5_trader import MT5Trader
from ict_model import ICTModel
from db_manager import DatabaseManager
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ForexTradingBot:
    """Main class for the Forex Trading Bot"""
    
    def __init__(self):
        """Initialize the trading bot components"""
        # Initialize components
        self.market_status = MarketStatus()
        
        # Try to use MT5 data provider first, fall back to Alpha Vantage if not available
        try:
            self.data_provider = ForexDataProviderMT5()
            logger.info("Using MT5 for forex data")
        except Exception as e:
            logger.warning(f"Failed to initialize MT5 data provider: {e}. Falling back to Alpha Vantage.")
            self.data_provider = ForexDataProvider()
            
        self.analyzer = TechnicalAnalyzer()
        self.global_markets = GlobalMarkets()
        self.ict_model = ICTModel()
        
        # Initialize database manager for MongoDB integration
        try:
            self.db_manager = DatabaseManager()
            logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize database manager: {e}")
            self.db_manager = None
        
        # Initialize news analyzer if OpenAI API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.news_analyzer = NewsAnalyzer(openai_api_key) if openai_api_key else None
        
        # Initialize MT5 trader
        self.mt5_trader = None
        self.mt5_enabled = False
        
        # Set auto trading flag based on environment variable
        auto_trading_env = os.getenv('MT5_AUTO_TRADING', 'false').lower()
        self.auto_trading = auto_trading_env == 'true'
        
        # Set MT5 enabled flag if credentials are available
        if os.getenv('MT5_LOGIN') and os.getenv('MT5_PASSWORD') and os.getenv('MT5_SERVER'):
            self.mt5_enabled = True
        
        # MT5 account credentials from environment variables
        self.mt5_server = os.getenv('MT5_SERVER')
        self.mt5_login = os.getenv('MT5_LOGIN')
        self.mt5_password = os.getenv('MT5_PASSWORD')
        
        # Default currency pairs to monitor
        self.currency_pairs = ['EURUSD', 'GBPUSD']  # Focused on primary pairs only
        
        # Store the latest signals
        self.latest_signals = {}
        self.active_trades = {}
        
        # Flag to control the analysis loop
        self.running = False
        
        # Flag to track if we've notified about closed market
        self.market_closed_notified = False
        
        # Trading parameters
        self.risk_percent = 1.0  # 1% risk per trade
        self.min_confidence = 0.6  # Minimum confidence to execute a trade
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Forex Trading Bot")
        self.running = True
        
        # Initialize MT5 if credentials are available
        if self.mt5_server or os.getenv('MT5_CREATE_DEMO', 'false').lower() == 'true':
            self._initialize_mt5()
        
        # Start analysis loop in a separate thread
        analysis_thread = threading.Thread(target=self._analysis_loop, name="AnalysisThread")
        analysis_thread.daemon = True
        analysis_thread.start()
        logger.info("Analysis loop started in background thread")
        
        # Start the main analysis loop
        self._analysis_loop()
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Forex Trading Bot")
        self.running = False
        
        # Shutdown MT5 if enabled
        if self.mt5_enabled and self.mt5_trader:
            logger.info("Shutting down MT5 connection")
            self.mt5_trader.shutdown()
            self.mt5_enabled = False
            
    def get_mt5_account_info(self):
        """Get MT5 account information"""
        if not self.mt5_enabled or not self.mt5_trader:
            return None
            
        try:
            return self.mt5_trader.get_account_info()
        except Exception as e:
            logger.error(f"Error getting MT5 account info: {e}")
            return None
    
    def get_active_trades(self):
        """Get active trades from MT5"""
        if not self.mt5_enabled or not self.mt5_trader:
            return {}
            
        try:
            return self.mt5_trader.get_open_positions()
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return {}
    
    def _generate_signals(self, pair, data, technical_analysis, ict_analysis, news=None):
        """Generate trading signals based on multiple analysis methods
        
        Args:
            pair (str): Currency pair
            data (pd.DataFrame): OHLCV data
            technical_analysis (dict): Technical analysis results
            ict_analysis (dict): ICT model analysis results
            news (list, optional): News items for the currency pair
            
        Returns:
            dict: Combined trading signal with confidence score
        """
        # Default signal (no action)
        signal = {
            'pair': pair,
            'action': 'HOLD',
            'confidence': 0,
            'reason': 'No strong signals detected',
            'timestamp': datetime.now(),
            'technical_factors': [],
            'ict_factors': [],
            'news_factors': []
        }
        
        try:
            # Get technical analysis factors
            tech_action = technical_analysis.get('action', 'HOLD') if isinstance(technical_analysis, dict) else 'HOLD'
            tech_confidence = technical_analysis.get('confidence', 0) if isinstance(technical_analysis, dict) else 0
            tech_reason = technical_analysis.get('reason', '') if isinstance(technical_analysis, dict) else ''
            
            # Get ICT model factors
            ict_signals = ict_analysis.get('signals', []) if isinstance(ict_analysis, dict) else []
            ict_confidence = ict_analysis.get('confidence', 0) if isinstance(ict_analysis, dict) else 0
            
            # Count buy and sell signals from ICT model
            ict_buy_count = sum(1 for s in ict_signals if isinstance(s, dict) and s.get('action') == 'BUY')
            ict_sell_count = sum(1 for s in ict_signals if isinstance(s, dict) and s.get('action') == 'SELL')
            
            # Determine ICT action based on signal count
            ict_action = 'HOLD'
            if ict_buy_count > ict_sell_count and ict_confidence >= 0.4:
                ict_action = 'BUY'
            elif ict_sell_count > ict_buy_count and ict_confidence >= 0.4:
                ict_action = 'SELL'
            
            # Analyze news impact if available
            news_impact = 'NEUTRAL'
            news_confidence = 0
            news_factors = []
            
            if news and len(news) > 0:
                # Count positive and negative news impacts
                positive_count = sum(1 for n in news if isinstance(n, dict) and n.get('impact_direction', 'Neutral') == 'Positive')
                negative_count = sum(1 for n in news if isinstance(n, dict) and n.get('impact_direction', 'Neutral') == 'Negative')
                
                # Determine overall news impact
                if positive_count > negative_count:
                    news_impact = 'POSITIVE'
                    news_confidence = 0.3 + (0.1 * min(positive_count, 5))  # Max 0.8
                elif negative_count > positive_count:
                    news_impact = 'NEGATIVE'
                    news_confidence = 0.3 + (0.1 * min(negative_count, 5))  # Max 0.8
                
                # Extract news factors
                for n in news:
                    if isinstance(n, dict) and n.get('impact', 'Low') != 'Low':
                        news_factors.append({
                            'title': n.get('title', 'Unknown news'),
                            'impact': n.get('impact', 'Medium'),
                            'direction': n.get('impact_direction', 'Neutral')
                        })
            
            # Combine all analysis results
            # Technical analysis has 40% weight
            # ICT model has 40% weight
            # News has 20% weight
            
            # Determine final action based on weighted consensus
            actions = []
            if tech_action != 'HOLD':
                actions.append((tech_action, 0.4))
            if ict_action != 'HOLD':
                actions.append((ict_action, 0.4))
            
            # Add news impact
            if news_impact == 'POSITIVE':
                actions.append(('BUY', 0.2))
            elif news_impact == 'NEGATIVE':
                actions.append(('SELL', 0.2))
            
            # Calculate weighted action
            if actions:
                buy_weight = sum(w for a, w in actions if a == 'BUY')
                sell_weight = sum(w for a, w in actions if a == 'SELL')
                
                if buy_weight > sell_weight and buy_weight >= 0.4:
                    final_action = 'BUY'
                    final_confidence = buy_weight
                elif sell_weight > buy_weight and sell_weight >= 0.4:
                    final_action = 'SELL'
                    final_confidence = sell_weight
                else:
                    final_action = 'HOLD'
                    final_confidence = 0
                
                # Build reason string
                reasons = []
                if tech_action != 'HOLD':
                    reasons.append(f"Technical: {tech_action} ({tech_confidence:.2f})")
                if ict_action != 'HOLD':
                    reasons.append(f"ICT: {ict_action} ({ict_confidence:.2f})")
                if news_impact != 'NEUTRAL':
                    reasons.append(f"News: {news_impact} ({news_confidence:.2f})")
                
                reason = ", ".join(reasons)
                
                # Update signal
                signal['action'] = final_action
                signal['confidence'] = round(final_confidence, 2)
                signal['reason'] = reason
                signal['technical_factors'] = technical_analysis.get('indicators', {}) if isinstance(technical_analysis, dict) else {}
                signal['ict_factors'] = ict_signals
                signal['news_factors'] = news_factors
        
        except Exception as e:
            logger.error(f"Error generating signals for {pair}: {e}")
            # Return default HOLD signal on error
        
        return signal
    
    def _initialize_mt5(self):
    """Initialize MetaTrader 5 connection"""
    try:
        logger.info("Initializing MetaTrader 5 connection")
        self.mt5_trader = MT5Trader()
        
        # Connect to MT5 with existing credentials
        if self.mt5_server and self.mt5_login and self.mt5_password:
            logger.info(f"Connecting to MT5 account on server {self.mt5_server}")
            
            # Connect to the account
            connected = self.mt5_trader.connect_account(
                server=self.mt5_server,
                login=int(self.mt5_login),
                password=self.mt5_password
            )
            
            if connected:
                logger.info(f"Connected to MT5 account: Login={self.mt5_login}, Server={self.mt5_server}")
                self.mt5_enabled = True
                
                # Get account info to verify connection
                account_info = self.mt5_trader.get_account_info()
                if account_info:
                    logger.info(f"MT5 Account: {account_info.get('login')}, Balance: {account_info.get('balance')}")
                
                # Set auto trading flag based on environment variable
                self.auto_trading = os.getenv('MT5_AUTO_TRADING', 'false').lower() == 'true'
                logger.info(f"MT5 auto trading is {'enabled' if self.auto_trading else 'disabled'}")
            else:
                logger.error(f"Failed to connect to MT5 account: Login={self.mt5_login}, Server={self.mt5_server}")
                self.mt5_trader = None
                self.mt5_enabled = False
        else:
            logger.warning("MT5 credentials not provided. Trading functionality disabled.")
            self.mt5_trader = None
            self.mt5_enabled = False
    except Exception as e:
        logger.error(f"Error initializing MT5: {e}")
        self.mt5_trader = None
        self.mt5_enabled = False

def _analysis_loop(self, single_run=False):
    """Main loop for analyzing forex data and generating signals
    
    Args:
        single_run (bool): If True, run the analysis only once and return
    """
    while self.running:
        try:
            # Check if forex market is open
            is_market_open = self.market_status.is_market_open()
            single_run (bool): If True, run the analysis only once and return
        """
        while self.running:
            try:
                # Check if forex market is open
                is_market_open = self.market_status.is_market_open()
                
                if is_market_open:
                    # Reset the notification flag when market opens
                    if self.market_closed_notified:
                        self.market_closed_notified = False
                    
                    logger.info("Forex market is open. Performing forex analysis...")
                    
                    # Analyze each currency pair
                    for pair in self.currency_pairs:
                        try:
                            # Get forex data
                            data = self.data_provider.get_forex_data(pair, interval='1h')
                            if data is None:
                                logger.warning(f"No data available for {pair}")
                                continue
                            
                            # Perform technical analysis
                            analysis = self.analyzer.analyze(data, pair)
                            
                            # Get ICT model analysis
                            ict_analysis = self.ict_model.analyze(data, pair)
                            logger.info(f"ICT analysis completed for {pair} with {len(ict_analysis.get('signals', []))} signals")
                            
                            # Get news for this currency pair
                            news = None
                            if self.news_analyzer:
                                news = self.news_analyzer.get_forex_news(pair)
                                if news:
                                    logger.info(f"News received for {pair}: {len(news)} items")
                                else:
                                    logger.info(f"No news available for {pair}")
                            
                            # Generate trading signals
                            signals = self._generate_signals(pair, data, analysis, ict_analysis, news)
                            
                            # Store the latest signals
                            self.latest_signals[pair] = signals
                            
                            # Save signals to database
                            if self.db_manager and signals and 'action' in signals:
                                signal_data = {
                                    'pair': pair,
                                    'action': signals.get('action'),
                                    'confidence': signals.get('confidence', 0),
                                    'timestamp': datetime.now(),
                                    'reason': signals.get('reason', ''),
                                    'ict_factors': signals.get('ict_factors', [])
                                }
                                self.db_manager.save_signal(signal_data)
                            
                            # Execute trade if auto-trading is enabled and signal confidence is high enough
                            if signals and 'action' in signals:
                                confidence = signals.get('confidence', 0)
                                min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.6))
                                
                                if self.auto_trading and confidence >= self.min_confidence:
                                    logger.info(f"Auto-trading: Executing {signals['action']} trade for {pair} with confidence {confidence}")
                                    self._execute_trade(pair, signals)
                        
                        except Exception as e:
                            logger.error(f"Error analyzing {pair}: {e}")
                    
                    # Update active trades status
                    self._update_active_trades()
                    
                    # Save performance metrics to database
                    if self.db_manager and self.mt5_enabled:
                        try:
                            account_info = self.mt5_trader.get_account_info()
                            if account_info:
                                metrics = {
                                    'balance': account_info.get('balance', 0),
                                    'equity': account_info.get('equity', 0),
                                    'profit': account_info.get('profit', 0),
                                    'active_trades_count': len(self.active_trades)
                                }
                                self.db_manager.save_performance_metrics(metrics)
                        except Exception as e:
                            logger.error(f"Error saving performance metrics: {e}")
                    
                    # Wait for next analysis cycle (1 minute)
                    time.sleep(60)
                else:
                    # Notify once that market is closed
                    if not self.market_closed_notified:
                        logger.info("Forex market is closed. Waiting for market to open...")
                        self.market_closed_notified = True
                    
                    # Check every 5 minutes if market has opened
                    time.sleep(300)
            
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                if single_run:
                    break
                time.sleep(60)  # Wait a bit before retrying
    
    def _execute_trade(self, pair, signals):
        """Execute a trade based on signals
        
        Args:
            pair (str): Currency pair
            signals (dict): Trading signals
        
        Returns:
            bool: True if trade executed, False otherwise
        """
        if not self.mt5_enabled or not self.mt5_trader:
            logger.warning("MT5 not enabled. Cannot execute trade.")
            return False
        
        # Check if confidence is high enough
        confidence = signals.get('confidence', 0)
        if confidence < self.min_confidence:
            logger.info(f"Signal confidence {confidence} below threshold {self.min_confidence}. Not executing trade.")
            return False
            
        # Save signal to database
        if self.db_manager:
            signal_data = {
                'pair': pair,
                'action': signals.get('action'),
                'confidence': confidence,
                'timestamp': datetime.now(),
                'reason': signals.get('reason', ''),
                'ict_factors': signals.get('ict_factors', [])
            }
            self.db_manager.save_signal(signal_data)
        
        # Check if signal confidence is high enough
        confidence = signals.get('confidence', 0)
        if confidence < self.min_confidence:
            logger.info(f"Signal confidence for {pair} ({confidence:.2f}) below threshold ({self.min_confidence}). Skipping.")
            return False
        
        action = signals.get('action', 'NEUTRAL')
        if action not in ['BUY', 'SELL']:
            logger.info(f"No actionable signal for {pair}. Current action: {action}")
            return False
        
        # Get key levels for stop loss and take profit
        key_levels = signals.get('key_levels', {})
        
        # Default stop loss and take profit in pips
        stop_loss_pips = 50
        take_profit_pips = 100
        
        # Try to use ICT model levels for more intelligent SL/TP
        if key_levels:
            # For a BUY trade
            if action == 'BUY':
                # Use nearest liquidity pool low as stop loss if available
                if 'liquidity_pools' in key_levels and 'lows' in key_levels['liquidity_pools']:
                    lows = key_levels['liquidity_pools']['lows']
                    if lows:
                        # Sort by distance from current price
                        current_price = self.mt5_trader.get_forex_data(pair).iloc[-1]['close']
                        sorted_lows = sorted(lows, key=lambda x: abs(x['level'] - current_price))
                        if sorted_lows:
                            # Use the nearest low as stop loss
                            stop_loss = sorted_lows[0]['level']
                            # Calculate pips
                            stop_loss_pips = int((current_price - stop_loss) / 0.0001)
                
                # Use nearest order block or fair value gap as take profit if available
                if 'order_blocks' in key_levels:
                    order_blocks = [b for b in key_levels['order_blocks'] if b['type'] == 'bearish_order_block']
                    if order_blocks:
                        # Sort by distance from current price (ascending)
                        current_price = self.mt5_trader.get_forex_data(pair).iloc[-1]['close']
                        sorted_blocks = sorted(order_blocks, key=lambda x: abs(x['level'] - current_price))
                        if sorted_blocks:
                            # Use the nearest block as take profit
                            take_profit = sorted_blocks[0]['level']
                            # Calculate pips
                            take_profit_pips = int((take_profit - current_price) / 0.0001)
            
            # For a SELL trade
            elif action == 'SELL':
                # Use nearest liquidity pool high as stop loss if available
                if 'liquidity_pools' in key_levels and 'highs' in key_levels['liquidity_pools']:
                    highs = key_levels['liquidity_pools']['highs']
                    if highs:
                        # Sort by distance from current price
                        current_price = self.mt5_trader.get_forex_data(pair).iloc[-1]['close']
                        sorted_highs = sorted(highs, key=lambda x: abs(x['level'] - current_price))
                        if sorted_highs:
                            # Use the nearest high as stop loss
                            stop_loss = sorted_highs[0]['level']
                            # Calculate pips
                            stop_loss_pips = int((stop_loss - current_price) / 0.0001)
                
                # Use nearest order block or fair value gap as take profit if available
                if 'order_blocks' in key_levels:
                    order_blocks = [b for b in key_levels['order_blocks'] if b['type'] == 'bullish_order_block']
                    if order_blocks:
                        # Sort by distance from current price (ascending)
                        current_price = self.mt5_trader.get_forex_data(pair).iloc[-1]['close']
                        sorted_blocks = sorted(order_blocks, key=lambda x: abs(x['level'] - current_price))
                        if sorted_blocks:
                            # Use the nearest block as take profit
                            take_profit = sorted_blocks[0]['level']
                            # Calculate pips
                            take_profit_pips = int((current_price - take_profit) / 0.0001)
        
        # Ensure stop loss and take profit are reasonable
        stop_loss_pips = max(20, min(stop_loss_pips, 200))  # Between 20 and 200 pips
        take_profit_pips = max(40, min(take_profit_pips, 400))  # Between 40 and 400 pips
        
        # Calculate position size based on risk
        lot_size = self.mt5_trader.calculate_position_size(
            pair, 
            risk_percent=self.risk_percent, 
            stop_loss_pips=stop_loss_pips
        )
        
        # Execute the trade
        logger.info(f"Executing {action} trade for {pair} with {lot_size} lots (SL: {stop_loss_pips} pips, TP: {take_profit_pips} pips)")
        
        # Add ICT factors to comment if available
        comment = "MCP Trader - ICT Model"
        ict_factors = signals.get('ict_factors', [])
        if ict_factors:
            factor_names = [f['factor'] for f in ict_factors]
            comment += f": {', '.join(factor_names[:3])}"  # Limit to first 3 factors
        
        # Open the trade
        trade_result = self.mt5_trader.open_trade(
            symbol=pair,
            order_type=action,
            lot_size=lot_size,
            comment=comment
        )
        
        if trade_result:
            logger.info(f"Trade executed successfully: Ticket #{trade_result['ticket']}")
            
            # Store the active trade
            trade_data = {
                'ticket': trade_result['ticket'],
                'pair': pair,
                'type': action,
                'open_time': datetime.now(),
                'open_price': trade_result['price'],
                'lot_size': lot_size,
                'stop_loss': trade_result['stop_loss'],
                'take_profit': trade_result['take_profit'],
                'signals': signals,
                'status': 'open',
                'confidence': signals.get('confidence', 0),
                'reason': signals.get('reason', '')
            }
            
            # Save to active trades
            self.active_trades[pair] = trade_data
            
            # Save to database
            if self.db_manager:
                self.db_manager.save_trade(trade_data)
            
            return True
        else:
            logger.error(f"Failed to execute trade for {pair}")
            return False
    
    def _update_active_trades(self):
        """Update status of active trades"""
        if not self.mt5_enabled or not self.mt5_trader:
            return
        
        # Get all open positions
        open_positions = self.mt5_trader.get_open_positions()
        open_tickets = [p['ticket'] for p in open_positions]
        
        # Check each active trade
        closed_pairs = []
        for pair, trade in self.active_trades.items():
            if trade['ticket'] not in open_tickets:
                logger.info(f"Trade for {pair} (Ticket #{trade['ticket']}) has been closed")
                closed_pairs.append(pair)
        
        # Remove closed trades from active trades
        for pair in closed_pairs:
            del self.active_trades[pair]
    
    def close_all_trades(self):
        """Close all active trades
        
        Returns:
            int: Number of trades closed
        """
        if not self.mt5_enabled or not self.mt5_trader:
            logger.warning("MT5 not enabled. Cannot close trades.")
            return 0
        
        # Get all active trades before closing
        active_trades = self.active_trades.copy()
        
        # Close all positions
        closed_count = self.mt5_trader.close_all_positions()
        
        # Update database for each closed trade
        if self.db_manager:
            for pair, trade in active_trades.items():
                close_data = {
                    'status': 'closed',
                    'close_time': datetime.now(),
                    'close_reason': 'manual_close_all'
                }
                self.db_manager.close_trade(trade['ticket'], close_data)
        
        self.active_trades = {}  # Clear active trades
        
        return closed_count
    
    def get_mt5_account_info(self):
        """Get MT5 account information
        
        Returns:
            dict: Account information or None if MT5 not enabled
        """
        if not self.mt5_enabled or not self.mt5_trader:
            return None
        
        return self.mt5_trader.get_account_info()
    
    def get_active_trades(self):
        """Get active trades
        
        Returns:
            dict: Active trades
        """
        return self.active_trades
    
    def add_currency_pair(self, pair):
        """Add a currency pair to the watchlist"""
        pair = pair.upper()
        if pair not in self.currency_pairs:
            self.currency_pairs.append(pair)
            return True
        return False
    
    def remove_currency_pair(self, pair):
        """Remove a currency pair from the watchlist"""
        pair = pair.upper()
        if pair in self.currency_pairs:
            self.currency_pairs.remove(pair)
            return True
        return False
    
    def get_currency_pairs(self):
        """Get the list of currency pairs being monitored"""
        return self.currency_pairs

if __name__ == "__main__":
    try:
        bot = ForexTradingBot()
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
