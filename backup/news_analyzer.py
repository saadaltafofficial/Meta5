#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for analyzing forex news using OpenAI API"""

import logging
import os
import json
import requests
from datetime import datetime, timedelta
import openai

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Class for analyzing forex news using OpenAI API"""
    
    def __init__(self, api_key=None):
        """Initialize the news analyzer
        
        Args:
            api_key (str, optional): OpenAI API key
        """
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
            self.enabled = True
        else:
            self.enabled = False
            logger.warning("OpenAI API key not provided. News analysis will be disabled.")
        
        # Cache to store news and reduce API calls
        self.news_cache = {}
        self.last_update = {}
    
    def get_forex_news(self, pair):
        """Get forex news for a specific currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            
        Returns:
            list: List of news items or None if an error occurs
        """
        if not self.enabled:
            return None
        
        try:
            # Check if we need to update the cache
            current_time = datetime.now()
            if pair in self.last_update:
                time_diff = (current_time - self.last_update[pair]).total_seconds()
                # Only update if more than 30 minutes have passed
                if time_diff < 1800:
                    return self.news_cache.get(pair, [])
            
            # Extract currencies from the pair
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            # Get news for both currencies
            news_items = self._search_forex_news(from_currency, to_currency)
            
            # Analyze the news impact
            analyzed_news = self._analyze_news_impact(news_items, from_currency, to_currency)
            
            # Update cache
            self.news_cache[pair] = analyzed_news
            self.last_update[pair] = current_time
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error fetching forex news for {pair}: {e}")
            return None
    
    def _search_forex_news(self, currency1, currency2):
        """Search for forex news related to the given currencies
        
        Args:
            currency1 (str): First currency code
            currency2 (str): Second currency code
            
        Returns:
            list: List of news items
        """
        try:
            # Use OpenAI to search for recent forex news
            prompt = f"""Find the latest forex news for {currency1} and {currency2} currencies. 
            Focus on news that might impact the {currency1}/{currency2} exchange rate.
            Include only factual information from the last 24 hours.
            Format the response as a JSON array with objects containing: 
            {{"title": "News title", "summary": "Brief summary", "source": "News source", "url": "URL if available", "date": "Publication date"}}"""
            
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides forex market news in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract the JSON response
            content = response.choices[0].message.content
            
            # Find JSON in the response (it might be wrapped in markdown code blocks)
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                news_items = json.loads(json_content)
                return news_items
            else:
                # Try to parse the entire content as JSON
                try:
                    news_items = json.loads(content)
                    return news_items
                except:
                    logger.error(f"Failed to parse JSON from OpenAI response: {content}")
                    return []
            
        except Exception as e:
            logger.error(f"Error searching for forex news: {e}")
            return []
    
    def _analyze_news_impact(self, news_items, currency1, currency2):
        """Analyze the impact of news items on the currency pair
        
        Args:
            news_items (list): List of news items
            currency1 (str): First currency code
            currency2 (str): Second currency code
            
        Returns:
            list: List of news items with impact analysis
        """
        if not news_items:
            return []
        
        try:
            # Prepare the prompt for impact analysis
            news_json = json.dumps(news_items)
            prompt = f"""Analyze the following forex news items and determine their potential impact on the {currency1}/{currency2} exchange rate.
            For each news item, add an 'impact' field with value 'High', 'Medium', or 'Low' and an 'impact_direction' field with value 'Positive' (for {currency1} strengthening against {currency2}), 
            'Negative' (for {currency1} weakening against {currency2}), or 'Neutral'.
            Also add a brief 'impact_explanation' field explaining why.
            
            News items: {news_json}
            
            Return the enriched news items as a JSON array."""
            
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are a forex analyst that analyzes news impact on currency pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Extract the JSON response
            content = response.choices[0].message.content
            
            # Find JSON in the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                analyzed_news = json.loads(json_content)
                return analyzed_news
            else:
                # Try to parse the entire content as JSON
                try:
                    analyzed_news = json.loads(content)
                    return analyzed_news
                except:
                    logger.error(f"Failed to parse JSON from OpenAI response: {content}")
                    return news_items  # Return original news items without analysis
            
        except Exception as e:
            logger.error(f"Error analyzing news impact: {e}")
            return news_items  # Return original news items without analysis
