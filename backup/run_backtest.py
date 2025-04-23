#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to run backtests on forex trading strategies"""

import os
import logging
from datetime import datetime, timedelta
from backtest import Backtester
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def run_eurusd_backtest():
    """Run a backtest on EURUSD for the past 30 days"""
    # Check if Alpha Vantage API key is available
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Alpha Vantage API key not found in .env file")
        logger.info("Please add your API key to the .env file as ALPHA_VANTAGE_API_KEY=your_key")
        return
    
    # Create a backtester with initial capital
    backtester = Backtester(initial_capital=10000)
    
    # Define date range for backtesting (last 30 days to avoid API limitations)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"Running EURUSD backtest from {start_date} to {end_date}")
    
    # Run backtest for EURUSD with 1-hour candles
    results = backtester.run_backtest(
        pair='EURUSD',
        start_date=start_date,
        end_date=end_date,
        interval='1h',
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.04  # 4% take profit
    )
    
    if results:
        # Visualize the results
        backtester.plot_results('EURUSD')
        
        # Generate and save a detailed report
        report_file = backtester.save_report('EURUSD')
        logger.info(f"Report saved to {report_file}")
        
        # Print summary statistics
        stats = results['stats']
        print("\n===== BACKTEST SUMMARY =====")
        print(f"Initial Capital: ${backtester.initial_capital:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {stats['total_return_pct']:.2f}%")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print("===========================\n")
        
        return results
    else:
        logger.error("Backtest failed to run")
        return None

def run_multi_pair_backtest():
    """Run backtests on multiple currency pairs"""
    # Check if Alpha Vantage API key is available
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Alpha Vantage API key not found in .env file")
        logger.info("Please add your API key to the .env file as ALPHA_VANTAGE_API_KEY=your_key")
        return
    
    # Create a backtester with initial capital
    backtester = Backtester(initial_capital=10000)
    
    # Define date range for backtesting (last 20 days to avoid API limitations)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    
    # List of currency pairs to test
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    
    # Store results for comparison
    all_results = {}
    
    for pair in pairs:
        logger.info(f"Running backtest for {pair} from {start_date} to {end_date}")
        
        # Run backtest for the pair
        results = backtester.run_backtest(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            interval='1h',
            stop_loss_pct=0.02,  # 2% stop loss
            take_profit_pct=0.04  # 4% take profit
        )
        
        if results:
            # Save results for comparison
            all_results[pair] = results
            
            # Generate and save a detailed report
            report_file = backtester.save_report(pair)
            logger.info(f"Report for {pair} saved to {report_file}")
            
            # Visualize the results
            backtester.plot_results(pair)
    
    # Print comparison summary
    if all_results:
        print("\n===== MULTI-PAIR COMPARISON =====")
        print(f"{'Pair':<10} {'Return %':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD':<10}")
        print("-" * 55)
        
        for pair, results in all_results.items():
            stats = results['stats']
            print(f"{pair:<10} {stats['total_return_pct']:>8.2f}% {stats['win_rate']:>8.2f}% {stats['profit_factor']:>13.2f} {stats['max_drawdown']:>8.2f}%")
        
        print("================================\n")
    
    return all_results

def run_parameter_optimization():
    """Run backtests with different parameter combinations to find optimal settings"""
    # Check if Alpha Vantage API key is available
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Alpha Vantage API key not found in .env file")
        logger.info("Please add your API key to the .env file as ALPHA_VANTAGE_API_KEY=your_key")
        return
    
    # Define date range for backtesting (last 20 days to avoid API limitations)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    
    # Currency pair to optimize
    pair = 'EURUSD'
    
    # Parameter combinations to test
    stop_loss_options = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
    take_profit_options = [0.02, 0.04, 0.06]  # 2%, 4%, 6%
    
    # Store results for comparison
    optimization_results = []
    
    logger.info(f"Running parameter optimization for {pair}")
    
    for sl in stop_loss_options:
        for tp in take_profit_options:
            # Skip invalid combinations (take profit should be greater than stop loss)
            if tp <= sl:
                continue
                
            # Create a new backtester for each parameter combination
            backtester = Backtester(initial_capital=10000)
            
            logger.info(f"Testing SL={sl*100}%, TP={tp*100}%")
            
            # Run backtest with current parameters
            results = backtester.run_backtest(
                pair=pair,
                start_date=start_date,
                end_date=end_date,
                interval='1h',
                stop_loss_pct=sl,
                take_profit_pct=tp
            )
            
            if results:
                stats = results['stats']
                
                # Store results for comparison
                optimization_results.append({
                    'stop_loss': sl,
                    'take_profit': tp,
                    'return': stats['total_return_pct'],
                    'win_rate': stats['win_rate'],
                    'profit_factor': stats['profit_factor'],
                    'max_drawdown': stats['max_drawdown'],
                    'sharpe_ratio': stats['sharpe_ratio'],
                    'total_trades': stats['total_trades']
                })
    
    # Sort results by return (descending)
    optimization_results.sort(key=lambda x: x['return'], reverse=True)
    
    # Print optimization results
    if optimization_results:
        print("\n===== PARAMETER OPTIMIZATION RESULTS =====")
        print(f"{'SL%':<6} {'TP%':<6} {'Return%':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD':<10} {'Trades':<8}")
        print("-" * 70)
        
        for result in optimization_results:
            print(f"{result['stop_loss']*100:<5.1f}% {result['take_profit']*100:<5.1f}% {result['return']:>8.2f}% {result['win_rate']:>8.2f}% {result['profit_factor']:>13.2f} {result['max_drawdown']:>8.2f}% {result['total_trades']:>6}")
        
        print("=========================================\n")
        
        # Get the best parameter combination
        best_result = optimization_results[0]
        logger.info(f"Best parameters: SL={best_result['stop_loss']*100}%, TP={best_result['take_profit']*100}%")
        logger.info(f"Best return: {best_result['return']:.2f}%")
    
    return optimization_results

if __name__ == "__main__":
    print("\nForex Trading Bot - Backtest Runner\n")
    print("Select an option:")
    print("1. Run EURUSD backtest (30 days)")
    print("2. Run multi-pair comparison (20 days)")
    print("3. Run parameter optimization for EURUSD (15 days)")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        run_eurusd_backtest()
    elif choice == '2':
        run_multi_pair_backtest()
    elif choice == '3':
        run_parameter_optimization()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
