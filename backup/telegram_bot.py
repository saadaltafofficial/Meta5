#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for Telegram bot integration"""

import logging
import os
import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from global_markets import GlobalMarkets
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TelegramBot:
    """Class for Telegram bot integration"""
    
    def __init__(self, token, trading_bot):
        """Initialize the Telegram bot
        
        Args:
            token (str): Telegram bot token
            trading_bot: Reference to the trading bot instance
        """
        self.token = token
        self.trading_bot = trading_bot
        self.application = Application.builder().token(token).build()
        
        # Initialize global markets tracker
        self.global_markets = GlobalMarkets()
        
        # Register command handlers
        self.register_handlers()
        
        # Store active users
        self.active_users = set()
    
    def register_handlers(self):
        """Register command handlers for the bot"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("pairs", self.pairs_command))
        self.application.add_handler(CommandHandler("markets", self.markets_command))
        self.application.add_handler(CommandHandler("add_pair", self.add_pair_command))
        self.application.add_handler(CommandHandler("remove_pair", self.remove_pair_command))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start_polling(self):
        """Start the bot polling with retry logic"""
        max_retries = 5
        retry_count = 0
        retry_delay = 2  # seconds
        
        while retry_count < max_retries:
            try:
                # Set a reasonable timeout for all operations
                await asyncio.wait_for(self._initialize_and_start_polling(), timeout=15)
                logger.info("Telegram bot started polling successfully")
                return  # Success, exit the retry loop
            except asyncio.TimeoutError:
                retry_count += 1
                logger.warning(f"Telegram bot polling timed out. Retry {retry_count}/{max_retries}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error starting Telegram bot polling: {e}. Retry {retry_count}/{max_retries}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        # If we get here, all retries failed
        logger.error("Failed to start Telegram bot polling after multiple retries")
        raise Exception("Failed to start Telegram bot polling after multiple retries")
    
    async def _initialize_and_start_polling(self):
        """Internal method to initialize and start polling"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True, timeout=10)
        return True
    
    def start(self):
        """Start the bot in a non-blocking way"""
        try:
            # Create a new thread to run the async loop
            import threading
            
            def run_bot():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the polling in this thread's event loop
                    loop.run_until_complete(self.start_polling())
                except Exception as e:
                    logger.error(f"Error in Telegram bot thread: {e}")
                    logger.info("Bot will continue running in offline mode without Telegram functionality")
                    # Set a flag to indicate we're in offline mode
                    self.offline_mode = True
                finally:
                    loop.close()
            
            # Initialize offline mode flag
            self.offline_mode = False
            
            # Start the bot in a separate thread
            bot_thread = threading.Thread(target=run_bot, name="TelegramBotThread")
            bot_thread.daemon = True  # Thread will exit when main program exits
            bot_thread.start()
            
            logger.info("Telegram bot started in background thread")
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}")
            logger.info("Continuing without Telegram bot functionality")
            self.offline_mode = True
    
    async def stop_polling(self):
        """Stop the bot polling"""
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
        logger.info("Telegram bot stopped")
        
    def stop(self):
        """Stop the bot"""
        try:
            # Create a new thread to run the async loop
            import threading
            
            def stop_bot():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the stop polling in this thread's event loop
                loop.run_until_complete(self.stop_polling())
                loop.close()
            
            # Stop the bot in a separate thread
            stop_thread = threading.Thread(target=stop_bot)
            stop_thread.daemon = True
            stop_thread.start()
            stop_thread.join(timeout=5)  # Wait up to 5 seconds for clean shutdown
            
            logger.info("Telegram bot stopped completely")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        user = update.effective_user
        self.active_users.add(user.id)
        
        # Check if market is open
        market_status = self.trading_bot.market_status.get_market_hours_text()
        
        welcome_message = f"👋 Hello {user.first_name}!\n\n"
        welcome_message += f"Welcome to the Forex Trading Bot.\n\n"
        welcome_message += f"{market_status}\n\n"
        welcome_message += f"Use /help to see available commands."
        
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        help_text = "📚 *Available Commands*\n\n"
        help_text += "/start - Start the bot and check market status\n"
        help_text += "/status - Check if the forex market is open\n"
        help_text += "/signals - Get the latest forex trading signals\n"
        help_text += "/pairs - See the list of currency pairs being monitored\n"
        help_text += "/markets - View global forex markets status\n"
        help_text += "/add_pair SYMBOL - Add a currency pair (e.g., /add_pair EURUSD)\n"
        help_text += "/remove_pair SYMBOL - Remove a currency pair (e.g., /remove_pair EURUSD)\n"
        help_text += "/help - Show this help message"
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command"""
        market_status = self.trading_bot.market_status.get_market_hours_text()
        await update.message.reply_text(market_status)
    
    async def markets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /markets command to show global forex markets status"""
        markets_status = self.global_markets.get_centers_status_text()
        await update.message.reply_text(markets_status, parse_mode=ParseMode.MARKDOWN)
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /signals command"""
        signals = self.trading_bot.get_latest_signals()
        
        if not signals:
            await update.message.reply_text("No forex signals available yet. Please wait for the next analysis cycle.")
            return
        
        response = "📊 *Latest Forex Trading Signals*\n\n"
        
        for pair, signal in signals.items():
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)
            price = signal.get('price', 'N/A')
            
            # Format the action with emoji
            if action == 'BUY':
                action_emoji = "🟢 BUY"
            elif action == 'SELL':
                action_emoji = "🔴 SELL"
            elif action == 'WEAK BUY':
                action_emoji = "🟡 WEAK BUY"
            elif action == 'WEAK SELL':
                action_emoji = "🟠 WEAK SELL"
            else:
                action_emoji = "⚪ HOLD"
            
            response += f"*{pair}*: {action_emoji} (Confidence: {confidence:.1f}%)\n"
            response += f"Price: {price}\n"
            response += f"Reason: {signal.get('reason', 'N/A')}\n\n"
        
        # Add market status information
        is_open = self.trading_bot.market_status.is_market_open()
        if not is_open:
            market_status = self.trading_bot.market_status.get_market_hours_text()
            response += f"\n🔴 *Note:* {market_status}"
        
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    
    async def pairs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /pairs command"""
        pairs = self.trading_bot.get_currency_pairs()
        
        if not pairs:
            await update.message.reply_text("No currency pairs are being monitored.")
            return
        
        response = "💱 *Monitored Currency Pairs*\n\n"
        for pair in pairs:
            response += f"• {pair}\n"
        
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    
    async def add_pair_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /add_pair command"""
        if not context.args or len(context.args) != 1:
            await update.message.reply_text("Please specify a currency pair to add (e.g., /add_pair EURUSD)")
            return
        
        pair = context.args[0].upper()
        
        # Validate the pair format (should be 6 characters, e.g., EURUSD)
        if len(pair) != 6:
            await update.message.reply_text("Invalid currency pair format. Please use format like EURUSD.")
            return
        
        # Add the pair
        success = self.trading_bot.add_currency_pair(pair)
        
        if success:
            await update.message.reply_text(f"✅ Added {pair} to monitored currency pairs.")
        else:
            await update.message.reply_text(f"ℹ️ {pair} is already being monitored.")
    
    async def remove_pair_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /remove_pair command"""
        if not context.args or len(context.args) != 1:
            await update.message.reply_text("Please specify a currency pair to remove (e.g., /remove_pair EURUSD)")
            return
        
        pair = context.args[0].upper()
        
        # Remove the pair
        success = self.trading_bot.remove_currency_pair(pair)
        
        if success:
            await update.message.reply_text(f"✅ Removed {pair} from monitored currency pairs.")
        else:
            await update.message.reply_text(f"ℹ️ {pair} is not in the list of monitored pairs.")
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error} - {update}")
        
        # If update is available, send error message to user
        if update and update.effective_message:
            await update.effective_message.reply_text("Sorry, an error occurred while processing your request.")
    
    async def send_market_closed_notification(self, market_status_text):
        """Send a notification that the forex market is closed
        
        Args:
            market_status_text (str): Text describing market hours
        """
        # Check if we're in offline mode
        if hasattr(self, 'offline_mode') and self.offline_mode:
            logger.info(f"Market closed notification: {market_status_text} (not sent - offline mode)")
            return
        message = f"🔴 *FOREX MARKET CLOSED* 🔴\n\n"
        message += f"{market_status_text}\n\n"
        message += f"The bot will resume forex analysis when the market reopens."
        
        # Send the message to all active users
        for user_id in self.active_users:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Error sending market closed notification to user {user_id}: {e}")
    
    async def send_signal(self, pair, signal):
        """Send a trading signal to all active users
        
        Args:
            pair (str): Currency pair
            signal (dict): Signal data
        """
        # Check if we're in offline mode
        if hasattr(self, 'offline_mode') and self.offline_mode:
            logger.info(f"Signal generated for {pair}: {signal.get('action', 'UNKNOWN')} (not sent - offline mode)")
            return
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        price = signal.get('price', 'N/A')
        reason = signal.get('reason', 'N/A')
        timestamp = signal.get('timestamp', 'N/A')
        
        # Format the action with emoji
        if action == 'BUY':
            action_emoji = "🟢 BUY"
        elif action == 'SELL':
            action_emoji = "🔴 SELL"
        elif action == 'WEAK BUY':
            action_emoji = "🟡 WEAK BUY"
        elif action == 'WEAK SELL':
            action_emoji = "🟠 WEAK SELL"
        else:
            action_emoji = "⚪ HOLD"
        
        # Create the message
        message = f"📊 *FOREX SIGNAL ALERT*\n\n"
        
        message += f"*{pair}*: {action_emoji}\n"
        message += f"Price: {price}\n"
        message += f"Confidence: {confidence:.1f}%\n"
        message += f"Reason: {reason}\n"
        message += f"Time: {timestamp}\n\n"
        
        # Add indicator details
        indicators = signal.get('indicators', {})
        if indicators:
            message += "*Technical Indicators*:\n"
            
            # RSI
            rsi = indicators.get('rsi')
            if rsi is not None:
                message += f"RSI: {rsi:.2f}\n"
            
            # MACD
            macd = indicators.get('macd', {})
            if macd:
                message += f"MACD: {macd.get('macd', 'N/A'):.4f}, "
                message += f"Signal: {macd.get('signal', 'N/A'):.4f}, "
                message += f"Hist: {macd.get('histogram', 'N/A'):.4f}\n"
            
            # Stochastic
            stoch = indicators.get('stochastic', {})
            if stoch:
                message += f"Stoch %K: {stoch.get('k', 'N/A'):.2f}, "
                message += f"%D: {stoch.get('d', 'N/A'):.2f}\n"
        
        # Send the message to all active users
        for user_id in self.active_users:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Error sending signal to user {user_id}: {e}")
    
    async def send_news(self, pair, news):
        """Send news updates to all active users
        
        Args:
            pair (str): Currency pair
            news (list): List of news items
        """
        # Check if we're in offline mode
        if hasattr(self, 'offline_mode') and self.offline_mode:
            logger.info(f"News received for {pair} (not sent - offline mode)")
            return
        if not news:
            return
        
        # Create the message
        message = f"📰 *FOREX NEWS UPDATE - {pair}*\n\n"
        
        for item in news[:3]:  # Limit to 3 news items to avoid too long messages
            title = item.get('title', 'No title')
            summary = item.get('summary', 'No summary')
            source = item.get('source', 'Unknown source')
            url = item.get('url', '')
            impact = item.get('impact', 'Unknown')
            
            # Format impact with emoji
            if impact.lower() == 'high':
                impact_emoji = "🔴 High"
            elif impact.lower() == 'medium':
                impact_emoji = "🟠 Medium"
            else:
                impact_emoji = "🟡 Low"
            
            message += f"*{title}*\n"
            message += f"Impact: {impact_emoji}\n"
            message += f"{summary}\n"
            message += f"Source: {source}\n"
            if url:
                message += f"[Read more]({url})\n"
            message += "\n"
        
        # Send the message to all active users
        for user_id in self.active_users:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
            except Exception as e:
                logger.error(f"Error sending news to user {user_id}: {e}")
    

    

    

    

    

