#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Economic Calendar Module for Forex Trading Bot

This module fetches economic calendar data directly from MetaTrader 5
"""

import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

class EconomicCalendar:
    """Economic Calendar integration for forex trading using MetaTrader 5"""
    
    def __init__(self):
        # List of high-impact economic events to watch for
        self.high_impact_events = [
            'Interest Rate Decision',
            'Non-Farm Payrolls',
            'CPI',
            'GDP',
            'Unemployment Rate',
            'Retail Sales',
            'PMI',
            'Trade Balance',
            'FOMC Statement',
            'ECB Press Conference',
            'Fed Chair Powell Speaks',
            'ECB President Lagarde Speaks',
            'BOE Gov Bailey Speaks',
            'BOJ Policy Rate',
            'Employment Change',
            'ISM Manufacturing PMI',
            'ISM Services PMI'
        ]
        
        # Cache for economic events to avoid excessive API calls
        self.events_cache = {}
        self.cache_expiry = datetime.now()
        
        # Ensure MT5 is initialized
        if not mt5.initialize():
            logger.error(f"Failed to initialize MetaTrader 5: {mt5.last_error()}")
            # We'll continue and use sample data as fallback if MT5 isn't available
    
    def get_economic_events(self, base_currency, quote_currency, hours_window=24):
        """Get economic events for the specified currencies within the time window
        
        Args:
            base_currency (str): Base currency code (e.g., 'EUR')
            quote_currency (str): Quote currency code (e.g., 'USD')
            hours_window (int): Hours to look ahead for events
            
        Returns:
            list: List of economic events affecting the currency pair
        """
        try:
            # Check if we need to refresh the cache
            if datetime.now() > self.cache_expiry:
                self._refresh_events_cache()
            
            # Get countries for the currencies
            base_country = self._get_country_for_currency(base_currency)
            quote_country = self._get_country_for_currency(quote_currency)
            
            # Filter events for the specified currencies
            currency_events = []
            for event in self.events_cache.get('events', []):
                # Check if event is for one of our currencies
                if event['country'] in [base_country, quote_country]:
                    # Check if the event is within the time window
                    event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S')
                    if event_time <= datetime.now() + timedelta(hours=hours_window):
                        # Add currency pair info to the event
                        event['pair'] = base_currency + quote_currency
                        currency_events.append(event)
            
            return currency_events
        except Exception as e:
            logger.error(f"Error getting economic events: {e}")
            return []
    
    def _refresh_events_cache(self):
        """Refresh the economic events cache from MetaTrader 5"""
        try:
            # Check if MT5 is initialized
            if not mt5.initialize():
                logger.warning("MetaTrader 5 not initialized, using sample data")
                self.events_cache = self._get_sample_events()
                self.cache_expiry = datetime.now() + timedelta(hours=1)
                return
            
            # Get current time and time window (7 days)
            now = datetime.now()
            from_date = now.date()
            to_date = (now + timedelta(days=7)).date()
            
            # Fetch economic calendar from MT5
            logger.info(f"Fetching economic calendar from MT5 from {from_date} to {to_date}")
            # Try to use economic_calendar_get if available, otherwise use sample data
            try:
                calendar = mt5.economic_calendar_get(from_date, to_date)
            except AttributeError:
                logger.warning("MetaTrader5 module does not have economic_calendar_get function, using sample data")
                self.events_cache = self._get_sample_events()
                self.cache_expiry = datetime.now() + timedelta(hours=1)
                return
            
            if calendar is None:
                logger.warning(f"Failed to get economic calendar from MT5: {mt5.last_error()}")
                self.events_cache = self._get_sample_events()
                self.cache_expiry = datetime.now() + timedelta(minutes=30)
                return
            
            # Convert MT5 calendar to our format
            events = []
            for event in calendar:
                # Convert impact (0-3) to text
                impact = "Low"
                if event['importance'] == 2:
                    impact = "Medium"
                elif event['importance'] == 3:
                    impact = "High"
                
                # Format event data
                formatted_event = {
                    'country': event['country'],
                    'category': event['sector'],
                    'event': event['event_name'],
                    'importance': impact,
                    'date': datetime.fromtimestamp(event['time']).strftime('%Y-%m-%dT%H:%M:%S'),
                    'previous': str(event['previous_value']),
                    'forecast': str(event['forecast_value']),
                    'impact_numeric': event['importance']  # Store the numeric value too
                }
                events.append(formatted_event)
            
            # Store in cache
            self.events_cache = {
                'events': events,
                'timestamp': now
            }
            
            # Set cache expiry (refresh every 6 hours)
            self.cache_expiry = now + timedelta(hours=6)
            logger.info(f"Economic calendar refreshed with {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error refreshing economic events from MT5: {e}")
            self.events_cache = self._get_sample_events()
            self.cache_expiry = datetime.now() + timedelta(minutes=30)
    
    def _get_country_for_currency(self, currency_code):
        """Get the country name for a currency code"""
        currency_map = {
            'USD': 'United States',
            'EUR': 'Euro Area',
            'GBP': 'United Kingdom',
            'JPY': 'Japan',
            'AUD': 'Australia',
            'CAD': 'Canada',
            'CHF': 'Switzerland',
            'NZD': 'New Zealand'
        }
        return currency_map.get(currency_code, 'Unknown')
    
    def _get_sample_events(self):
        """Get sample economic events when API is not available"""
        # Create events for the next few days
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        day_after = now + timedelta(days=2)
        
        sample_events = {
            'events': [
                {
                    'country': 'United States',
                    'category': 'Central Banks',
                    'event': 'Interest Rate Decision',
                    'importance': 'High',
                    'date': tomorrow.strftime('%Y-%m-%dT%H:%M:%S'),
                    'previous': '5.50%',
                    'forecast': '5.25%'
                },
                {
                    'country': 'United States',
                    'category': 'Labor',
                    'event': 'Non-Farm Payrolls',
                    'importance': 'High',
                    'date': day_after.strftime('%Y-%m-%dT%H:%M:%S'),
                    'previous': '275K',
                    'forecast': '240K'
                },
                {
                    'country': 'Euro Area',
                    'category': 'Central Banks',
                    'event': 'ECB Press Conference',
                    'importance': 'High',
                    'date': tomorrow.strftime('%Y-%m-%dT%H:%M:%S'),
                    'previous': '',
                    'forecast': ''
                },
                {
                    'country': 'United Kingdom',
                    'category': 'Inflation',
                    'event': 'CPI YoY',
                    'importance': 'High',
                    'date': (now + timedelta(hours=18)).strftime('%Y-%m-%dT%H:%M:%S'),
                    'previous': '3.4%',
                    'forecast': '3.2%'
                }
            ],
            'timestamp': now
        }
        return sample_events
    
    def is_high_impact_event_soon(self, pair, hours_window=24):
        """Check if there's a high impact event soon for the currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            hours_window (int): Hours to look ahead
            
        Returns:
            tuple: (bool, event_dict) - True and event details if high impact event is upcoming, False and None otherwise
        """
        try:
            base_currency = pair[:3]  # EUR from EURUSD
            quote_currency = pair[3:]  # USD from EURUSD
            
            events = self.get_economic_events(base_currency, quote_currency, hours_window)
            
            # Check for high impact events
            for event in events:
                # Check if this is a high impact event (either marked as High or in our list of important events)
                is_high_impact = (
                    event.get('importance') == 'High' or 
                    event.get('impact_numeric', 0) >= 2 or  # Medium or High in MT5
                    any(important_event.lower() in event.get('event', '').lower() for important_event in self.high_impact_events)
                )
                
                if is_high_impact:
                    # Calculate hours until event
                    event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S')
                    hours_until = (event_time - datetime.now()).total_seconds() / 3600
                    
                    # If event is within the window
                    if -0.25 <= hours_until <= hours_window:  # Include events that started up to 15 min ago
                        logger.info(f"High impact economic event detected for {pair}: {event['event']} at {event['date']}")
                        return True, event
            
            return False, None
        except Exception as e:
            logger.error(f"Error checking for high impact events: {e}")
            return False, None

# For testing
if __name__ == "__main__":
    calendar = EconomicCalendar()
    has_event, event = calendar.is_high_impact_event_soon('EURUSD', 48)
    if has_event:
        print(f"High impact event found: {event['event']} in {event['country']} at {event['date']}")
    else:
        print("No high impact events found in the next 48 hours")
