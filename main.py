#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICT Forex Trading Bot - Main Entry Point

This is the main entry point for the ICT-based forex trading bot.
It initializes the trading system and starts the bot.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('config/.env')

# Add all project directories to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main trader class
from src.core.standalone_trader import main

if __name__ == "__main__":
    # Start the ICT trading bot
    logger.info("Starting ICT Forex Trading Bot...")
    main()
