#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for the standalone_trader.py file to properly handle confidence boost

This script will patch the generate_signal method in the StandaloneTrader class
to properly handle HOLD signals with non-zero confidence.
"""

import os
import logging
import shutil

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def fix_standalone_trader():
    """Fix the generate_signal method in the StandaloneTrader class"""
    try:
        # Backup the original file
        standalone_trader_path = os.path.join('src', 'core', 'standalone_trader.py')
        backup_path = os.path.join('src', 'core', 'standalone_trader.py.bak')
        
        # Create backup if it doesn't exist
        if not os.path.exists(backup_path):
            shutil.copy2(standalone_trader_path, backup_path)
            logger.info(f"Created backup of {standalone_trader_path} at {backup_path}")
        
        # Read the original file
        with open(standalone_trader_path, 'r') as f:
            content = f.read()
        
        # Find the generate_signal method
        start_marker = "def generate_signal(self, pair):"
        end_marker = "def get_performance_analytics"
        
        # Split the content
        parts = content.split(start_marker)
        if len(parts) != 2:
            logger.error("Could not find generate_signal method in standalone_trader.py")
            return False
        
        prefix = parts[0]
        rest = parts[1].split(end_marker)
        if len(rest) != 2:
            logger.error("Could not find end of generate_signal method in standalone_trader.py")
            return False
        
        method_body = rest[0]
        suffix = end_marker + rest[1]
        
        # Find the section to replace
        target_section = """            # Get actions and confidences
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_conf = traditional_signal.get('confidence', 0)
            ict_action = ict_signal.get('action', 'HOLD')
            ict_conf = ict_signal.get('confidence', 0)
            
            # Calculate weighted confidence for each action type
            buy_confidence = 0
            sell_confidence = 0
            
            if trad_action == 'BUY':
                buy_confidence += trad_conf * traditional_weight
            elif trad_action == 'SELL':
                sell_confidence += trad_conf * traditional_weight
                
            if ict_action == 'BUY':
                buy_confidence += ict_conf * ict_weight
            elif ict_action == 'SELL':
                sell_confidence += ict_conf * ict_weight
            
            # Determine final action based on highest confidence
            if buy_confidence > sell_confidence and buy_confidence > self.min_confidence:
                action = 'BUY'
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence > self.min_confidence:
                action = 'SELL'
                confidence = sell_confidence
            else:
                action = 'HOLD'
                confidence = max(buy_confidence, sell_confidence)"""
        
        replacement_section = """            # Get actions and confidences
            trad_action = traditional_signal.get('action', 'HOLD')
            trad_conf = traditional_signal.get('confidence', 0)
            ict_action = ict_signal.get('action', 'HOLD')
            ict_conf = ict_signal.get('confidence', 0)
            
            # Calculate weighted confidence for each action type
            buy_confidence = 0
            sell_confidence = 0
            
            # For traditional signals
            if trad_action == 'BUY':
                buy_confidence += trad_conf * traditional_weight
            elif trad_action == 'SELL':
                sell_confidence += trad_conf * traditional_weight
            
            # For ICT signals - even if HOLD, check if confidence boost suggests a direction
            if ict_action == 'BUY':
                buy_confidence += ict_conf * ict_weight
            elif ict_action == 'SELL':
                sell_confidence += ict_conf * ict_weight
            elif ict_action == 'HOLD' and ict_conf > 0:
                # Check if there are confluence factors suggesting a direction
                confluence_factors = ict_signal.get('confluence_factors', [])
                if any('Bullish' in factor for factor in confluence_factors):
                    # Apply half the confidence to BUY
                    buy_confidence += (ict_conf * ict_weight) * 0.5
                    logger.info(f"Applied partial ICT confidence to BUY due to bullish confluence factors: {ict_conf:.4f}")
                elif any('Bearish' in factor for factor in confluence_factors):
                    # Apply half the confidence to SELL
                    sell_confidence += (ict_conf * ict_weight) * 0.5
                    logger.info(f"Applied partial ICT confidence to SELL due to bearish confluence factors: {ict_conf:.4f}")
            
            # Log confidence values for debugging
            logger.info(f"Signal confidence calculation for {pair}: ICT={ict_action}({ict_conf:.4f}), Traditional={trad_action}({trad_conf:.4f})")
            logger.info(f"Combined confidence: BUY={buy_confidence:.4f}, SELL={sell_confidence:.4f}, Threshold={self.min_confidence:.4f}")
            
            # Determine final action based on highest confidence
            if buy_confidence > sell_confidence and buy_confidence > self.min_confidence:
                action = 'BUY'
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence > self.min_confidence:
                action = 'SELL'
                confidence = sell_confidence
            else:
                action = 'HOLD'
                confidence = max(buy_confidence, sell_confidence)"""
        
        # Replace the section
        new_method_body = method_body.replace(target_section, replacement_section)
        
        # Combine everything back
        new_content = prefix + start_marker + new_method_body + suffix
        
        # Write the new content
        with open(standalone_trader_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Successfully fixed {standalone_trader_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing standalone_trader.py: {e}")
        return False

if __name__ == "__main__":
    if fix_standalone_trader():
        print("\n✅ Successfully fixed standalone_trader.py")
        print("\nThe fix improves how confidence is calculated by:")
        print("1. Properly handling HOLD signals with non-zero confidence")
        print("2. Applying partial confidence to BUY/SELL based on confluence factors")
        print("3. Adding detailed logging of confidence calculations")
        print("\nThis should result in more actionable trade signals.")
    else:
        print("\n❌ Failed to fix standalone_trader.py")
        print("\nPlease check the logs for more information.")
