import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('balance_truncator')

def truncate_balance_history(max_entries=100):
    """Truncate balance history file to keep only the latest entries"""
    try:
        history_file = 'balance_history.json'
        
        if not os.path.exists(history_file):
            logger.error(f"Balance history file not found: {history_file}")
            return False
        
        # Get file size before truncation
        file_size_before = os.path.getsize(history_file) / (1024 * 1024)  # Size in MB
        
        # Load the file
        with open(history_file, 'r') as f:
            data = json.load(f)
            history = data.get('history', [])
            
            # Get current entry count
            entry_count_before = len(history)
            
            # Truncate to latest max_entries
            if len(history) > max_entries:
                logger.info(f"Truncating balance history from {len(history)} to {max_entries} entries")
                history = history[-max_entries:]
                data['history'] = history
                
                # Save the truncated history
                with open(history_file, 'w') as f_write:
                    json.dump(data, f_write, indent=2)
                
                # Get file size after truncation
                file_size_after = os.path.getsize(history_file) / (1024 * 1024)  # Size in MB
                
                logger.info(f"Balance history truncated successfully")
                logger.info(f"Entries: {entry_count_before} → {len(history)}")
                logger.info(f"File size: {file_size_before:.2f}MB → {file_size_after:.2f}MB")
                return True
            else:
                logger.info(f"No truncation needed. Current entries: {len(history)} (max: {max_entries})")
                return False
    except Exception as e:
        logger.error(f"Error truncating balance history: {e}")
        return False

if __name__ == "__main__":
    print("ICT Trading Bot - Balance History Truncator")
    print("-" * 50)
    truncate_balance_history(100)
