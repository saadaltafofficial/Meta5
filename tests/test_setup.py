#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script to verify the forex trading bot setup"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import modules to test
try:
    from forex_data import ForexDataProvider
    from technical_analysis import TechnicalAnalyzer
    from market_status import MarketStatus
    logger.info("✅ Core modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Error importing core modules: {e}")

# Load environment variables
load_dotenv()

def test_environment():
    """Test the environment setup"""
    # Check Python environment
    import sys
    logger.info(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'ta', 'python-telegram-bot',
        'pytz', 'requests', 'python-dotenv', 'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is NOT installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Run 'pip install -r requirements.txt' to install missing packages")
    else:
        logger.info("All required packages are installed")

def test_api_keys():
    """Test API keys in the .env file"""
    # Check Telegram bot token
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_token or telegram_token == 'your_telegram_bot_token_here':
        logger.warning("⚠️ Telegram bot token not set in .env file")
    else:
        logger.info("✅ Telegram bot token is set")
    
    # Check Alpha Vantage API key
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not alpha_vantage_key or alpha_vantage_key == 'your_alpha_vantage_api_key_here':
        logger.warning("⚠️ Alpha Vantage API key not set in .env file")
    else:
        logger.info("✅ Alpha Vantage API key is set")
    
    # Check OpenAI API key (optional)
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key == 'your_openai_api_key_here':
        logger.warning("⚠️ OpenAI API key not set in .env file (optional for news analysis)")
    else:
        logger.info("✅ OpenAI API key is set")

def test_market_status():
    """Test the market status module"""
    try:
        market_status = MarketStatus()
        is_open = market_status.is_market_open()
        market_hours_text = market_status.get_market_hours_text()
        
        logger.info(f"Market open status: {is_open}")
        logger.info(f"Market hours: {market_hours_text}")
        
        logger.info("✅ Market status module is working")
    except Exception as e:
        logger.error(f"❌ Error testing market status: {e}")

if __name__ == "__main__":
    logger.info("Starting forex trading bot setup test")
    
    # Run tests
    test_environment()
    test_api_keys()
    test_market_status()
    
    logger.info("Setup test completed")
    
    # Instructions for next steps
    print("\n" + "-"*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. Update the .env file with your API keys:")
    print("   - Get a Telegram bot token from BotFather (https://t.me/botfather)")
    print("   - Get an Alpha Vantage API key from https://www.alphavantage.co/support/#api-key")
    print("   - Optionally get an OpenAI API key from https://platform.openai.com/api-keys")
    print("\n2. Run the main bot:")
    print("   python main.py")
    print("-"*80)
