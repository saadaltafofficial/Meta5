#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Manager for MCP Trader

This utility allows users to easily view and modify trading configurations
"""

import os
import json
import argparse
import logging
from config_loader import get_config, save_config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def print_config(config, section=None):
    """Print configuration in a readable format
    
    Args:
        config (dict): Configuration dictionary
        section (str, optional): Section to print. If None, prints all sections.
    """
    if section and section in config:
        print(f"\n===== {section.upper()} CONFIGURATION =====")
        _print_section(config[section], 0)
    elif not section:
        for section_name, section_data in config.items():
            print(f"\n===== {section_name.upper()} CONFIGURATION =====")
            _print_section(section_data, 0)
    else:
        print(f"Section '{section}' not found in configuration")

def _print_section(data, indent=0):
    """Recursively print a section of the configuration
    
    Args:
        data: Configuration data (dict, list, or scalar)
        indent (int): Indentation level
    """
    indent_str = '  ' * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent_str}{key}:")
                _print_section(value, indent + 1)
            else:
                print(f"{indent_str}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _print_section(item, indent + 1)
            else:
                print(f"{indent_str}- {item}")
    else:
        print(f"{indent_str}{data}")

def update_config(config, path, value):
    """Update a specific configuration value
    
    Args:
        config (dict): Configuration dictionary
        path (str): Path to the configuration value (e.g., 'trading.risk_management.risk_percent')
        value (str): New value (will be converted to appropriate type)
    
    Returns:
        dict: Updated configuration
    """
    # Split the path into parts
    parts = path.split('.')
    
    # Navigate to the target location
    current = config
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            print(f"Error: Path '{'.'.join(parts[:i+1])}' not found in configuration")
            return config
        current = current[part]
    
    # Get the last part (the actual key to update)
    last_part = parts[-1]
    if last_part not in current:
        print(f"Error: Key '{last_part}' not found in '{'.'.join(parts[:-1])}'")
        return config
    
    # Determine the type of the existing value and convert the new value accordingly
    existing_value = current[last_part]
    try:
        if isinstance(existing_value, bool):
            # Handle boolean values
            if value.lower() in ('true', 'yes', 'y', '1'):
                new_value = True
            elif value.lower() in ('false', 'no', 'n', '0'):
                new_value = False
            else:
                raise ValueError(f"Cannot convert '{value}' to boolean")
        elif isinstance(existing_value, int):
            new_value = int(value)
        elif isinstance(existing_value, float):
            new_value = float(value)
        elif isinstance(existing_value, list):
            # Handle lists - parse as JSON
            try:
                new_value = json.loads(value)
                if not isinstance(new_value, list):
                    new_value = [new_value]  # Convert single value to list
            except json.JSONDecodeError:
                # Try comma-separated format
                new_value = [item.strip() for item in value.split(',')]
                
                # Try to convert list items to appropriate types
                if all(existing_value) and isinstance(existing_value[0], (int, float)):
                    try:
                        if all(isinstance(x, int) for x in existing_value):
                            new_value = [int(x) for x in new_value]
                        else:
                            new_value = [float(x) for x in new_value]
                    except ValueError:
                        pass  # Keep as strings if conversion fails
        else:
            # String or other type
            new_value = value
            
        # Update the value
        current[last_part] = new_value
        print(f"Updated {path}: {existing_value} -> {new_value}")
        
    except Exception as e:
        print(f"Error updating {path}: {e}")
    
    return config

def add_currency_pair(config, pair):
    """Add a currency pair to the configuration
    
    Args:
        config (dict): Configuration dictionary
        pair (str): Currency pair to add (e.g., 'USDJPY')
    
    Returns:
        dict: Updated configuration
    """
    if 'trading' not in config or 'currency_pairs' not in config['trading']:
        print("Error: Configuration does not have currency_pairs section")
        return config
    
    pair = pair.upper()
    if pair in config['trading']['currency_pairs']:
        print(f"Currency pair {pair} already exists in configuration")
        return config
    
    config['trading']['currency_pairs'].append(pair)
    print(f"Added currency pair: {pair}")
    return config

def remove_currency_pair(config, pair):
    """Remove a currency pair from the configuration
    
    Args:
        config (dict): Configuration dictionary
        pair (str): Currency pair to remove (e.g., 'USDJPY')
    
    Returns:
        dict: Updated configuration
    """
    if 'trading' not in config or 'currency_pairs' not in config['trading']:
        print("Error: Configuration does not have currency_pairs section")
        return config
    
    pair = pair.upper()
    if pair not in config['trading']['currency_pairs']:
        print(f"Currency pair {pair} not found in configuration")
        return config
    
    config['trading']['currency_pairs'].remove(pair)
    print(f"Removed currency pair: {pair}")
    return config

def main():
    """Main function for the configuration manager"""
    parser = argparse.ArgumentParser(description='MCP Trader Configuration Manager')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View configuration')
    view_parser.add_argument('--section', '-s', help='Section to view')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update configuration value')
    update_parser.add_argument('path', help='Path to configuration value (e.g., trading.risk_management.risk_percent)')
    update_parser.add_argument('value', help='New value')
    
    # Add currency pair command
    add_pair_parser = subparsers.add_parser('add-pair', help='Add currency pair')
    add_pair_parser.add_argument('pair', help='Currency pair to add (e.g., USDJPY)')
    
    # Remove currency pair command
    remove_pair_parser = subparsers.add_parser('remove-pair', help='Remove currency pair')
    remove_pair_parser.add_argument('pair', help='Currency pair to remove (e.g., USDJPY)')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset configuration to defaults')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    if args.command == 'view':
        print_config(config, args.section)
    elif args.command == 'update':
        config = update_config(config, args.path, args.value)
        save_config(config)
    elif args.command == 'add-pair':
        config = add_currency_pair(config, args.pair)
        save_config(config)
    elif args.command == 'remove-pair':
        config = remove_currency_pair(config, args.pair)
        save_config(config)
    elif args.command == 'reset':
        if args.confirm:
            # Delete the config file to reset to defaults
            if os.path.exists('config.json'):
                os.remove('config.json')
                print("Configuration reset to defaults")
            else:
                print("No configuration file found")
        else:
            print("Please use --confirm to reset configuration")
    else:
        # If no command specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
