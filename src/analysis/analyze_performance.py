#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line tool to analyze trading performance with AI

This script provides a simple interface to run the AI performance analyzer
and view the results.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from config_loader import get_config
from standalone_trader import StandaloneTrader
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Run the AI Performance Analyzer"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Performance Analyzer for MCP Trader')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--symbol', type=str, help='Filter by symbol (e.g., EURUSD)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI analysis and only show database statistics')
    args = parser.parse_args()
    
    # Check for OpenAI API key if AI analysis is requested
    if not args.no_ai and not os.getenv('OPENAI_API_KEY'):
        print("\nWARNING: OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        print("You can still view performance statistics without AI analysis.\n")
        args.no_ai = True
    
    # Create trader instance
    trader = StandaloneTrader()
    
    # Get performance analytics
    print("\n==== FETCHING PERFORMANCE DATA ====\n")
    analytics = trader.get_performance_analytics(days=args.days, symbol=args.symbol)
    
    # Display basic performance metrics
    print("\n==== PERFORMANCE METRICS ====\n")
    
    if not analytics or not analytics.get('performance_metrics'):
        print("No performance data available. Make some trades first.")
        return
    
    metrics = analytics.get('performance_metrics', {})
    stats = analytics.get('trade_statistics', {})
    
    print(f"Period: Last {args.days} days")
    if args.symbol:
        print(f"Symbol: {args.symbol}")
    else:
        print("Symbol: All pairs")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Total Profit/Loss: ${metrics.get('total_profit', 0):.2f}")
    print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Maximum Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
    
    print("Trade Statistics:")
    print(f"Winning Trades: {stats.get('winning_trades', 0)}")
    print(f"Losing Trades: {stats.get('losing_trades', 0)}")
    print(f"Average Win: ${stats.get('avg_win', 0):.2f}")
    print(f"Average Loss: ${stats.get('avg_loss', 0):.2f}")
    print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}\n")
    
    # Run AI analysis if requested
    if not args.no_ai:
        print("\n==== AI PERFORMANCE ANALYSIS ====\n")
        print("Analyzing performance with GPT-4...\n")
        
        analysis = trader.get_ai_performance_analysis(days=args.days, symbol=args.symbol)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
        else:
            print(f"Summary: {analysis.get('summary', 'No summary available')}\n")
            
            print("Strengths:")
            for strength in analysis.get('strengths', []):
                print(f"- {strength}")
            print()
            
            print("Weaknesses:")
            for weakness in analysis.get('weaknesses', []):
                print(f"- {weakness}")
            print()
            
            print("Recommendations:")
            for recommendation in analysis.get('recommendations', []):
                print(f"- {recommendation}")
            print()
            
            if analysis.get('parameter_adjustments'):
                print("Suggested Parameter Adjustments:")
                for param, value in analysis.get('parameter_adjustments', {}).items():
                    print(f"- {param}: {value}")
                print()
            
            # Save to file if requested
            if args.output:
                try:
                    with open(args.output, 'w') as f:
                        json.dump(analysis, f, indent=2)
                    print(f"\nAnalysis saved to {args.output}")
                except Exception as e:
                    print(f"\nError saving analysis to file: {e}")
    
    # Close trader
    trader.close()

if __name__ == "__main__":
    main()
