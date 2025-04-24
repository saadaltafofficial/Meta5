#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix syntax error in standalone_trader.py

This script will restore the standalone_trader.py file from the backup.
"""

import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def fix_syntax_error():
    """Restore standalone_trader.py from backup"""
    try:
        # Check if backup exists
        standalone_trader_path = os.path.join('src', 'core', 'standalone_trader.py')
        backup_path = os.path.join('src', 'core', 'standalone_trader.py.bak')
        
        if os.path.exists(backup_path):
            # Restore from backup
            shutil.copy2(backup_path, standalone_trader_path)
            logger.info(f"Restored {standalone_trader_path} from {backup_path}")
            return True
        else:
            logger.error(f"Backup file {backup_path} not found")
            return False
    except Exception as e:
        logger.error(f"Error restoring standalone_trader.py: {e}")
        return False

if __name__ == "__main__":
    if fix_syntax_error():
        print("\n✅ Successfully fixed syntax error in standalone_trader.py")
    else:
        print("\n❌ Failed to fix syntax error in standalone_trader.py")
        print("\nPlease check the logs for more information.")
