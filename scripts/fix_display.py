#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix display issues in the standalone_trader.py file
"""

import os
import sys
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def fix_display_issues():
    # Path to the standalone_trader.py file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'core', 'standalone_trader.py')
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the display issues in the terminal output
    # 1. Fix the signals display
    signals_pattern = r'(\s+# Display signals[\s\S]+?if not trader\.latest_signals:[\s\S]+?for pair, signal in trader\.latest_signals\.items\(\):[\s\S]+?except Exception as e:[\s\S]+?print\(f"Error displaying signal for {pair}: {e}"\))'
    signals_replacement = '''
            # Display signals
            print("="*80)
            print(f"🔍 TRADING SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            if not trader.latest_signals:
                print("No trading signals available.")
            else:
                # Sort pairs to ensure consistent display order
                sorted_pairs = sorted(trader.latest_signals.keys())
                for pair in sorted_pairs:
                    try:
                        signal = trader.latest_signals[pair]
                        # Determine emoji based on action
                        action = signal.get('action', 'HOLD')
                        confidence = signal.get('confidence', 0)
                        
                        if action == 'BUY':
                            emoji = "🟢"
                        elif action == 'SELL':
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                        
                        print(f"{pair}: {emoji} {action} ({confidence:.1%} confidence)")
                        
                        # Print reason
                        reason = signal.get('reason', 'No reason provided')
                        print(f"  Reason: {reason}")
                        
                        # Print execution status
                        if confidence >= trader.min_confidence and action in ['BUY', 'SELL']:
                            print(f"  Execute: ✅ Yes")
                            print(f"  Confidence: {confidence:.2f}")
                        else:
                            print(f"  Execute: ❌ No (below threshold)")
                            print(f"  Reason: No execute reason provided")
                        
                        # Add a blank line between pairs for better readability
                        if pair != sorted_pairs[-1]:
                            print()
                    except Exception as e:
                        print(f"Error displaying signal for {pair}: {e}")
'''
    
    # 2. Fix the active trades display
    trades_pattern = r'(\s+# Display active trades[\s\S]+?if not active_trades:[\s\S]+?for trade in active_trades:[\s\S]+?except Exception as e:[\s\S]+?print\(f"Error displaying trade: {e}"\))'
    trades_replacement = '''
            # Display active trades
            print("="*80)
            print("📈 ACTIVE TRADES (ICT MODEL)")
            print("="*80)
            
            active_trades = trader.get_active_trades()
            if not active_trades:
                print("No active trades.")
            else:
                # Sort trades by symbol for consistent display
                sorted_trades = sorted(active_trades, key=lambda x: x.get('symbol', ''))
                for i, trade in enumerate(sorted_trades):
                    try:
                        # Extract trade information
                        symbol = trade.get('symbol', 'Unknown')
                        direction = trade.get('direction', 'Unknown')
                        entry_price = trade.get('entry_price', 0)
                        open_time = trade.get('open_time', datetime.now())
                        lot_size = trade.get('lot_size', 0)
                        sl = trade.get('stop_loss', 0)
                        tp = trade.get('take_profit', 0)
                        ticket = trade.get('ticket', 0)
                        profit = trade.get('profit', 0)
                        
                        # Determine emoji based on direction
                        direction_emoji = "🟢" if direction == "BUY" else "🔴"
                        profit_emoji = "🟢" if profit > 0 else "🔴"
                        
                        print(f"{symbol}: {direction_emoji} {direction} @ {entry_price}")
                        print(f"  Open Time: {open_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"  Lot Size: {lot_size}")
                        print(f"  Stop Loss: {sl}")
                        print(f"  Take Profit: {tp}")
                        print(f"  Ticket: {ticket}")
                        print(f"  Current Profit: {profit_emoji} ${profit:.2f}")
                        
                        # Add a blank line between trades for better readability, but not after the last one
                        if i < len(sorted_trades) - 1:
                            print()
                    except Exception as e:
                        print(f"Error displaying trade: {e}")
'''
    
    # Apply the replacements
    content = re.sub(signals_pattern, signals_replacement, content)
    content = re.sub(trades_pattern, trades_replacement, content)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Display issues fixed in {file_path}")

if __name__ == "__main__":
    fix_display_issues()
