#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean Run Script for ICT Trading Bot

This script runs the ICT trading bot with improved output formatting.
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging with a cleaner format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv('config/.env')

# Import and apply the confidence fix
from src.ict.apply_confidence_fix import apply_confidence_fix
apply_confidence_fix()

# Import the main trader class
from src.core.standalone_trader import main

def clean_run():
    """
    Run the ICT trading bot with improved output formatting.
    """
    # Print a clean header
    print("\n" + "=" * 80)
    print(f"ICT TRADING BOT - CLEAN RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Start the ICT trading bot
    logger.info("Starting ICT Forex Trading Bot with enhanced confidence calculation...")
    main()

if __name__ == "__main__":
    clean_run()
