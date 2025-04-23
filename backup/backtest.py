#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for backtesting trading strategies"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from forex_data import ForexDataProvider
from technical_analysis import TechnicalAnalyzer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies"""
    
    def __init__(self, initial_capital=10000):
        """Initialize the backtester
        
        Args:
            initial_capital (float): Initial capital for backtesting
        """
        self.data_provider = ForexDataProvider()
        self.analyzer = TechnicalAnalyzer()
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_backtest(self, pair, start_date, end_date, interval='1h', stop_loss_pct=0.02, take_profit_pct=0.04):
        """Run a backtest for a specific currency pair and time period
        
        Args:
            pair (str): Currency pair to backtest (e.g., 'EURUSD')
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            interval (str): Time interval for data (e.g., '1h', '4h', '1d')
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            
        Returns:
            dict: Backtest results
        """
        try:
            # Get historical data for the specified period
            logger.info(f"Getting historical data for {pair} from {start_date} to {end_date}")
            data = self.data_provider.get_historical_data(pair, start_date, end_date, interval)
            
            if data is None or len(data) < 10:
                logger.error(f"Insufficient data for {pair} in the specified period")
                return None
            
            # Initialize backtest variables
            capital = self.initial_capital
            position = None
            position_size = 0
            entry_price = 0
            trades = []
            equity_curve = []
            
            # Run the backtest
            logger.info(f"Running backtest for {pair}")
            
            # Process each candle
            for i in range(1, len(data)):
                # Get current window of data for analysis
                current_data = data.iloc[:i+1]
                
                # Get the current price
                current_price = current_data['close'].iloc[-1]
                current_time = current_data.index[-1]
                
                # Add current equity to equity curve
                if position:
                    # If in a position, calculate unrealized P&L
                    if position == 'long':
                        unrealized_pnl = (current_price - entry_price) * position_size
                    else:  # short
                        unrealized_pnl = (entry_price - current_price) * position_size
                    current_equity = capital + unrealized_pnl
                else:
                    current_equity = capital
                
                equity_curve.append({'time': current_time, 'equity': current_equity})
                
                # Check if we need to close a position (stop loss or take profit)
                if position:
                    if position == 'long':
                        # Check stop loss
                        if current_price <= entry_price * (1 - stop_loss_pct):
                            pnl = (current_price - entry_price) * position_size
                            capital += pnl
                            trades.append({
                                'type': 'exit',
                                'position': position,
                                'time': current_time,
                                'price': current_price,
                                'reason': 'stop_loss',
                                'pnl': pnl
                            })
                            position = None
                            logger.debug(f"Closed long position at {current_price} (stop loss)")
                        
                        # Check take profit
                        elif current_price >= entry_price * (1 + take_profit_pct):
                            pnl = (current_price - entry_price) * position_size
                            capital += pnl
                            trades.append({
                                'type': 'exit',
                                'position': position,
                                'time': current_time,
                                'price': current_price,
                                'reason': 'take_profit',
                                'pnl': pnl
                            })
                            position = None
                            logger.debug(f"Closed long position at {current_price} (take profit)")
                    
                    elif position == 'short':
                        # Check stop loss
                        if current_price >= entry_price * (1 + stop_loss_pct):
                            pnl = (entry_price - current_price) * position_size
                            capital += pnl
                            trades.append({
                                'type': 'exit',
                                'position': position,
                                'time': current_time,
                                'price': current_price,
                                'reason': 'stop_loss',
                                'pnl': pnl
                            })
                            position = None
                            logger.debug(f"Closed short position at {current_price} (stop loss)")
                        
                        # Check take profit
                        elif current_price <= entry_price * (1 - take_profit_pct):
                            pnl = (entry_price - current_price) * position_size
                            capital += pnl
                            trades.append({
                                'type': 'exit',
                                'position': position,
                                'time': current_time,
                                'price': current_price,
                                'reason': 'take_profit',
                                'pnl': pnl
                            })
                            position = None
                            logger.debug(f"Closed short position at {current_price} (take profit)")
                
                # Only generate new signals if we're not in a position
                if not position:
                    # Analyze the current data window
                    signal = self.analyzer.analyze(current_data, pair)
                    action = signal.get('action', 'HOLD')
                    
                    # Open a new position based on the signal
                    if action == 'BUY':
                        position = 'long'
                        # Use 2% of capital per trade
                        position_size = (capital * 0.02) / current_price
                        entry_price = current_price
                        trades.append({
                            'type': 'entry',
                            'position': position,
                            'time': current_time,
                            'price': current_price,
                            'size': position_size,
                            'reason': signal.get('reason', 'N/A')
                        })
                        logger.debug(f"Opened long position at {current_price}")
                    
                    elif action == 'SELL':
                        position = 'short'
                        # Use 2% of capital per trade
                        position_size = (capital * 0.02) / current_price
                        entry_price = current_price
                        trades.append({
                            'type': 'entry',
                            'position': position,
                            'time': current_time,
                            'price': current_price,
                            'size': position_size,
                            'reason': signal.get('reason', 'N/A')
                        })
                        logger.debug(f"Opened short position at {current_price}")
            
            # Close any open position at the end of the backtest
            if position:
                current_price = data['close'].iloc[-1]
                current_time = data.index[-1]
                
                if position == 'long':
                    pnl = (current_price - entry_price) * position_size
                else:  # short
                    pnl = (entry_price - current_price) * position_size
                
                capital += pnl
                trades.append({
                    'type': 'exit',
                    'position': position,
                    'time': current_time,
                    'price': current_price,
                    'reason': 'end_of_test',
                    'pnl': pnl
                })
            
            # Calculate backtest statistics
            stats = self._calculate_statistics(trades, equity_curve, self.initial_capital)
            
            # Store results
            self.results[pair] = {
                'trades': trades,
                'equity_curve': equity_curve,
                'stats': stats,
                'final_capital': capital
            }
            
            logger.info(f"Backtest completed for {pair}. Final capital: {capital:.2f}")
            return self.results[pair]
            
        except Exception as e:
            logger.error(f"Error in backtest for {pair}: {e}")
            return None
    
    def _calculate_statistics(self, trades, equity_curve, initial_capital):
        """Calculate performance statistics from backtest results
        
        Args:
            trades (list): List of trade dictionaries
            equity_curve (list): List of equity points over time
            initial_capital (float): Initial capital
            
        Returns:
            dict: Performance statistics
        """
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Filter to only include exit trades (which have PnL)
        exit_trades = trades_df[trades_df['type'] == 'exit']
        
        if len(exit_trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return_pct': 0
            }
        
        # Basic statistics
        total_trades = len(exit_trades)
        winning_trades = exit_trades[exit_trades['pnl'] > 0]
        losing_trades = exit_trades[exit_trades['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].max() * 100  # as percentage
        
        # Calculate returns
        equity_df['return'] = equity_df['equity'].pct_change()
        sharpe_ratio = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252) if equity_df['return'].std() > 0 else 0
        
        # Total return
        final_capital = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # as percentage
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return_pct': total_return_pct
        }
    
    def plot_results(self, pair):
        """Plot backtest results for a specific pair
        
        Args:
            pair (str): Currency pair to plot results for
        """
        if pair not in self.results:
            logger.error(f"No backtest results found for {pair}")
            return
        
        results = self.results[pair]
        equity_curve = pd.DataFrame(results['equity_curve'])
        trades = pd.DataFrame(results['trades'])
        stats = results['stats']
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(equity_curve['time'], equity_curve['equity'], label='Equity')
        ax1.set_title(f'Backtest Results for {pair}')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot entry and exit points
        if not trades.empty:
            # Plot entries
            entries = trades[trades['type'] == 'entry']
            long_entries = entries[entries['position'] == 'long']
            short_entries = entries[entries['position'] == 'short']
            
            if not long_entries.empty:
                ax1.scatter(long_entries['time'], long_entries['price'], 
                           color='green', marker='^', s=100, label='Long Entry')
            
            if not short_entries.empty:
                ax1.scatter(short_entries['time'], short_entries['price'], 
                           color='red', marker='v', s=100, label='Short Entry')
            
            # Plot exits
            exits = trades[trades['type'] == 'exit']
            profit_exits = exits[exits['pnl'] > 0]
            loss_exits = exits[exits['pnl'] <= 0]
            
            if not profit_exits.empty:
                ax1.scatter(profit_exits['time'], profit_exits['price'], 
                           color='blue', marker='o', s=100, label='Profit Exit')
            
            if not loss_exits.empty:
                ax1.scatter(loss_exits['time'], loss_exits['price'], 
                           color='orange', marker='o', s=100, label='Loss Exit')
        
        ax1.legend()
        
        # Plot drawdown
        if 'peak' in equity_curve.columns and 'drawdown' in equity_curve.columns:
            ax2.fill_between(equity_curve['time'], 0, equity_curve['drawdown'] * 100, 
                            color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Date')
            ax2.grid(True)
        
        # Add statistics as text
        stats_text = f"Total Trades: {stats['total_trades']}\n"
        stats_text += f"Win Rate: {stats['win_rate']:.2f}%\n"
        stats_text += f"Profit Factor: {stats['profit_factor']:.2f}\n"
        stats_text += f"Avg Win: ${stats['avg_win']:.2f}\n"
        stats_text += f"Avg Loss: ${stats['avg_loss']:.2f}\n"
        stats_text += f"Max Drawdown: {stats['max_drawdown']:.2f}%\n"
        stats_text += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
        stats_text += f"Total Return: {stats['total_return_pct']:.2f}%"
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"backtest_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        logger.info(f"Backtest plot saved as {filename}")
        
        # Show the plot
        plt.show()
    
    def generate_report(self, pair):
        """Generate a detailed report of backtest results
        
        Args:
            pair (str): Currency pair to generate report for
            
        Returns:
            str: Report text
        """
        if pair not in self.results:
            logger.error(f"No backtest results found for {pair}")
            return "No backtest results found."
        
        results = self.results[pair]
        trades = results['trades']
        stats = results['stats']
        
        report = f"===== BACKTEST REPORT FOR {pair} =====\n\n"
        
        # Summary statistics
        report += "PERFORMANCE SUMMARY:\n"
        report += f"Initial Capital: ${self.initial_capital:.2f}\n"
        report += f"Final Capital: ${results['final_capital']:.2f}\n"
        report += f"Total Return: {stats['total_return_pct']:.2f}%\n"
        report += f"Total Trades: {stats['total_trades']}\n"
        report += f"Win Rate: {stats['win_rate']:.2f}%\n"
        report += f"Profit Factor: {stats['profit_factor']:.2f}\n"
        report += f"Average Win: ${stats['avg_win']:.2f}\n"
        report += f"Average Loss: ${stats['avg_loss']:.2f}\n"
        report += f"Max Drawdown: {stats['max_drawdown']:.2f}%\n"
        report += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n\n"
        
        # Trade list
        report += "TRADE LIST:\n"
        report += "Type | Position | Time | Price | Reason | P&L\n"
        report += "-" * 70 + "\n"
        
        for trade in trades:
            trade_type = trade['type']
            position = trade.get('position', 'N/A')
            time = trade['time'].strftime('%Y-%m-%d %H:%M')
            price = f"${trade['price']:.5f}"
            reason = trade.get('reason', 'N/A')
            pnl = f"${trade.get('pnl', 0):.2f}" if 'pnl' in trade else 'N/A'
            
            report += f"{trade_type.ljust(6)} | {position.ljust(8)} | {time} | {price.ljust(12)} | {reason.ljust(15)} | {pnl}\n"
        
        report += "\n===== END OF REPORT =====\n"
        
        return report
    
    def save_report(self, pair, filename=None):
        """Save backtest report to a file
        
        Args:
            pair (str): Currency pair to save report for
            filename (str, optional): Filename to save report to
        """
        report = self.generate_report(pair)
        
        if not filename:
            filename = f"backtest_report_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Backtest report saved as {filename}")
        return filename


if __name__ == "__main__":
    # Example usage
    backtester = Backtester(initial_capital=10000)
    
    # Run backtest for EURUSD over the past 3 months
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    results = backtester.run_backtest('EURUSD', start_date, end_date, interval='1h')
    
    if results:
        # Plot the results
        backtester.plot_results('EURUSD')
        
        # Save a report
        backtester.save_report('EURUSD')
