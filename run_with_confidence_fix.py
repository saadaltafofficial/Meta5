#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Trading Bot with Confidence Fix

This script runs the trading bot with the confidence fix applied.
"""

import logging
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import and apply the confidence fix
from src.ict.apply_confidence_fix import apply_confidence_fix
apply_confidence_fix()

# Run the trading bot
from src.core.standalone_trader import main
main()
