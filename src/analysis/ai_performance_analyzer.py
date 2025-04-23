#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Performance Analyzer for MCP Trader

This module uses OpenAI's GPT-4 to analyze trading performance data
and provide insights and recommendations for improvement.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from database import TradeDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AIPerformanceAnalyzer:
    """Uses AI to analyze trading performance and provide recommendations"""
    
    def __init__(self, config=None):
        """Initialize the AI Performance Analyzer
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.trade_database = TradeDatabase(config)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. AI analysis will not be available.")
    
    def analyze_performance(self, days=30, symbol=None):
        """Analyze trading performance using AI
        
        Args:
            days (int, optional): Number of days to analyze. Defaults to 30.
            symbol (str, optional): Filter by symbol. Defaults to None.
            
        Returns:
            dict: AI analysis and recommendations
        """
        if not self.openai_api_key:
            return {
                'error': 'OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.',
                'recommendations': ['Set up OpenAI API key to enable AI analysis.']
            }
        
        try:
            # Get performance data from database
            performance_data = self.trade_database.get_performance_metrics(days)
            trade_statistics = self.trade_database.get_trade_statistics(symbol, days)
            
            # If no data, return early
            if not performance_data or not trade_statistics:
                return {
                    'error': 'Insufficient trading data for analysis',
                    'recommendations': ['Generate more trading data before requesting AI analysis.']
                }
            
            # Prepare data for OpenAI
            analysis_data = {
                'performance_metrics': performance_data,
                'trade_statistics': trade_statistics,
                'analysis_period_days': days,
                'symbol_filter': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get AI analysis
            ai_analysis = self._get_ai_analysis(analysis_data)
            
            return ai_analysis
        except Exception as e:
            logger.error(f"Error analyzing performance with AI: {e}")
            return {
                'error': f"Error analyzing performance: {str(e)}",
                'recommendations': ['Check logs for more details on the error.']
            }
    
    def _get_ai_analysis(self, data):
        """Get AI analysis from OpenAI API
        
        Args:
            data (dict): Trading performance data
            
        Returns:
            dict: AI analysis and recommendations
        """
        try:
            # Prepare the prompt
            prompt = self._create_analysis_prompt(data)
            
            # Call OpenAI API
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.openai_api_key}'
            }
            
            payload = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert trading performance analyst specializing in forex trading and the Inner Circle Trader (ICT) methodology. Your task is to analyze trading performance data and provide actionable insights and recommendations.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1500
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return {
                    'error': f"OpenAI API error: {response.status_code}",
                    'recommendations': ['Check OpenAI API key and quota.']
                }
            
            # Parse response
            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content']
            
            # Try to parse structured data from the response
            try:
                # Look for JSON-like structure in the response
                if '{' in ai_response and '}' in ai_response:
                    start_idx = ai_response.find('{')
                    end_idx = ai_response.rfind('}') + 1
                    json_str = ai_response[start_idx:end_idx]
                    structured_analysis = json.loads(json_str)
                else:
                    # Parse the text response into structured format
                    structured_analysis = self._parse_text_response(ai_response)
            except Exception as e:
                logger.warning(f"Could not parse structured data from AI response: {e}")
                structured_analysis = {
                    'summary': 'AI analysis completed but could not be structured.',
                    'strengths': [],
                    'weaknesses': [],
                    'recommendations': []
                }
            
            # Add the full text response
            structured_analysis['full_analysis'] = ai_response
            structured_analysis['generated_at'] = datetime.now().isoformat()
            
            return structured_analysis
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return {
                'error': f"Error getting AI analysis: {str(e)}",
                'recommendations': ['Check internet connection and OpenAI API status.']
            }
    
    def _create_analysis_prompt(self, data):
        """Create a prompt for the AI analysis
        
        Args:
            data (dict): Trading performance data
            
        Returns:
            str: Prompt for OpenAI
        """
        # Extract key metrics
        metrics = data['performance_metrics']
        stats = data['trade_statistics']
        period = data['analysis_period_days']
        symbol = data['symbol_filter'] or 'all currency pairs'
        
        prompt = f"""
        Analyze the following forex trading performance data for {symbol} over the past {period} days using the ICT (Inner Circle Trader) methodology:
        
        PERFORMANCE METRICS:
        - Total Profit/Loss: ${metrics.get('total_profit', 0):.2f}
        - Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%
        - Total Trades: {metrics.get('total_trades', 0)}
        - Maximum Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        
        TRADE STATISTICS:
        - Winning Trades: {stats.get('winning_trades', 0)}
        - Losing Trades: {stats.get('losing_trades', 0)}
        - Average Win: ${stats.get('avg_win', 0):.2f}
        - Average Loss: ${stats.get('avg_loss', 0):.2f}
        - Profit Factor: {stats.get('profit_factor', 0):.2f}
        
        Based on this data, please provide:
        1. A brief summary of overall performance
        2. Key strengths identified in the trading approach
        3. Key weaknesses or areas for improvement
        4. Specific recommendations to improve performance based on ICT methodology
        5. Suggested parameter adjustments (if applicable)
        
        Format your response as a JSON object with the following structure:
        {{
            "summary": "Overall performance summary",
            "strengths": ["Strength 1", "Strength 2", ...],
            "weaknesses": ["Weakness 1", "Weakness 2", ...],
            "recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "parameter_adjustments": {{
                "param_name": "suggested_value",
                ...
            }}
        }}
        """
        
        return prompt
    
    def _parse_text_response(self, text):
        """Parse a text response into structured format
        
        Args:
            text (str): AI response text
            
        Returns:
            dict: Structured analysis
        """
        lines = text.split('\n')
        
        analysis = {
            'summary': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'parameter_adjustments': {}
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify sections
            lower_line = line.lower()
            if 'summary' in lower_line or 'overall' in lower_line:
                current_section = 'summary'
                continue
            elif 'strength' in lower_line:
                current_section = 'strengths'
                continue
            elif 'weakness' in lower_line or 'improvement' in lower_line:
                current_section = 'weaknesses'
                continue
            elif 'recommendation' in lower_line or 'suggest' in lower_line:
                current_section = 'recommendations'
                continue
            elif 'parameter' in lower_line or 'adjustment' in lower_line:
                current_section = 'parameter_adjustments'
                continue
                
            # Process line based on current section
            if current_section == 'summary':
                analysis['summary'] += line + ' '
            elif current_section == 'strengths' and line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                analysis['strengths'].append(line.lstrip('-*•123456789. '))
            elif current_section == 'weaknesses' and line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                analysis['weaknesses'].append(line.lstrip('-*•123456789. '))
            elif current_section == 'recommendations' and line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                analysis['recommendations'].append(line.lstrip('-*•123456789. '))
            elif current_section == 'parameter_adjustments' and ':' in line:
                parts = line.split(':', 1)
                param = parts[0].strip().lstrip('-*•123456789. ')
                value = parts[1].strip()
                analysis['parameter_adjustments'][param] = value
        
        return analysis
    
    def save_analysis_to_file(self, analysis, filename=None):
        """Save AI analysis to a file
        
        Args:
            analysis (dict): AI analysis data
            filename (str, optional): Output filename. Defaults to None.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ai_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"AI analysis saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving AI analysis to file: {e}")
            return None


def main():
    """Run the AI Performance Analyzer as a standalone script"""
    import argparse
    from config_loader import get_config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Performance Analyzer for MCP Trader')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--symbol', type=str, help='Filter by symbol (e.g., EURUSD)')
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Create analyzer
    analyzer = AIPerformanceAnalyzer(config)
    
    # Run analysis
    analysis = analyzer.analyze_performance(days=args.days, symbol=args.symbol)
    
    # Save to file if requested
    if args.output:
        analyzer.save_analysis_to_file(analysis, args.output)
    else:
        # Print analysis
        print("\n==== AI PERFORMANCE ANALYSIS ====\n")
        
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

if __name__ == "__main__":
    main()
