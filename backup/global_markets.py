#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for tracking global forex market centers"""

import logging
from datetime import datetime, time, timedelta
import pytz

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class GlobalMarkets:
    """Class for tracking global forex market centers"""
    
    def __init__(self):
        """Initialize the global markets tracker"""
        # Define major forex trading centers with their timezones and trading hours
        self.trading_centers = {
            'Sydney': {
                'timezone': pytz.timezone('Australia/Sydney'),
                'open_hour': 7,  # 7:00 AM Sydney time
                'close_hour': 16,  # 4:00 PM Sydney time
                'weekend_closed': True  # Closed on weekends
            },
            'Tokyo': {
                'timezone': pytz.timezone('Asia/Tokyo'),
                'open_hour': 9,  # 9:00 AM Tokyo time
                'close_hour': 18,  # 6:00 PM Tokyo time
                'weekend_closed': True  # Closed on weekends
            },
            'London': {
                'timezone': pytz.timezone('Europe/London'),
                'open_hour': 8,  # 8:00 AM London time
                'close_hour': 16,  # 4:00 PM London time
                'weekend_closed': True  # Closed on weekends
            },
            'New York': {
                'timezone': pytz.timezone('America/New_York'),
                'open_hour': 8,  # 8:00 AM New York time
                'close_hour': 17,  # 5:00 PM New York time
                'weekend_closed': True  # Closed on weekends
            }
        }
        
        # Define forex market holidays (major US holidays when forex trading is limited)
        self.holidays = {
            'New York': [
                # 2025 holidays - update yearly
                datetime(2025, 1, 1),   # New Year's Day
                datetime(2025, 4, 18),  # Good Friday
                datetime(2025, 5, 26),  # Memorial Day
                datetime(2025, 7, 4),   # Independence Day
                datetime(2025, 9, 1),   # Labor Day
                datetime(2025, 11, 27), # Thanksgiving Day
                datetime(2025, 12, 25), # Christmas Day
            ],
            'London': [
                # 2025 holidays - update yearly
                datetime(2025, 1, 1),   # New Year's Day
                datetime(2025, 4, 18),  # Good Friday
                datetime(2025, 4, 21),  # Easter Monday
                datetime(2025, 5, 5),   # Early May Bank Holiday
                datetime(2025, 5, 26),  # Spring Bank Holiday
                datetime(2025, 8, 25),  # Summer Bank Holiday
                datetime(2025, 12, 25), # Christmas Day
                datetime(2025, 12, 26), # Boxing Day
            ],
            'Tokyo': [
                # 2025 holidays - update yearly
                datetime(2025, 1, 1),   # New Year's Day
                datetime(2025, 1, 13),  # Coming of Age Day
                datetime(2025, 2, 11),  # National Foundation Day
                datetime(2025, 3, 21),  # Vernal Equinox Day
                datetime(2025, 4, 29),  # Showa Day
                datetime(2025, 5, 3),   # Constitution Memorial Day
                datetime(2025, 5, 4),   # Greenery Day
                datetime(2025, 5, 5),   # Children's Day
                datetime(2025, 7, 21),  # Marine Day
                datetime(2025, 8, 11),  # Mountain Day
                datetime(2025, 9, 15),  # Respect for the Aged Day
                datetime(2025, 9, 23),  # Autumnal Equinox Day
                datetime(2025, 10, 13), # Sports Day
                datetime(2025, 11, 3),  # Culture Day
                datetime(2025, 11, 23), # Labor Thanksgiving Day
                datetime(2025, 12, 23), # Emperor's Birthday
            ],
            'Sydney': [
                # 2025 holidays - update yearly
                datetime(2025, 1, 1),   # New Year's Day
                datetime(2025, 1, 27),  # Australia Day (observed)
                datetime(2025, 4, 18),  # Good Friday
                datetime(2025, 4, 21),  # Easter Monday
                datetime(2025, 4, 25),  # Anzac Day
                datetime(2025, 6, 9),   # King's Birthday
                datetime(2025, 10, 6),  # Labour Day
                datetime(2025, 12, 25), # Christmas Day
                datetime(2025, 12, 26), # Boxing Day
            ]
        }
    
    def is_center_open(self, center_name):
        """Check if a specific trading center is currently open
        
        Args:
            center_name (str): Name of the trading center
            
        Returns:
            bool: True if the center is open, False otherwise
        """
        if center_name not in self.trading_centers:
            logger.warning(f"Unknown trading center: {center_name}")
            return False
        
        center = self.trading_centers[center_name]
        center_tz = center['timezone']
        
        # Get current time in the center's timezone
        now_utc = datetime.now(pytz.UTC)
        now_center = now_utc.astimezone(center_tz)
        
        # Check if today is a weekend and the center is closed on weekends
        if center['weekend_closed']:
            weekday = now_center.weekday()
            if weekday >= 5:  # Saturday or Sunday
                return False
        
        # Check if today is a holiday for this center
        today = now_center.replace(hour=0, minute=0, second=0, microsecond=0)
        if center_name in self.holidays:
            if any(today.date() == holiday.date() for holiday in self.holidays[center_name]):
                return False
        
        # Check if current time is within trading hours
        current_hour = now_center.hour
        current_minute = now_center.minute
        
        # Convert to decimal hours for easier comparison
        current_time_decimal = current_hour + (current_minute / 60)
        
        return center['open_hour'] <= current_time_decimal < center['close_hour']
    
    def get_all_centers_status(self):
        """Get the status of all trading centers
        
        Returns:
            dict: Dictionary with center names as keys and their status as values
        """
        status = {}
        for center_name in self.trading_centers:
            status[center_name] = self.is_center_open(center_name)
        return status
    
    def get_open_centers(self):
        """Get a list of currently open trading centers
        
        Returns:
            list: List of open trading center names
        """
        open_centers = []
        for center_name in self.trading_centers:
            if self.is_center_open(center_name):
                open_centers.append(center_name)
        return open_centers
    
    def get_closed_centers(self):
        """Get a list of currently closed trading centers
        
        Returns:
            list: List of closed trading center names
        """
        closed_centers = []
        for center_name in self.trading_centers:
            if not self.is_center_open(center_name):
                closed_centers.append(center_name)
        return closed_centers
    
    def get_center_hours(self, center_name):
        """Get the trading hours for a specific center
        
        Args:
            center_name (str): Name of the trading center
            
        Returns:
            dict: Dictionary with open and close hours, or None if center not found
        """
        if center_name not in self.trading_centers:
            logger.warning(f"Unknown trading center: {center_name}")
            return None
        
        center = self.trading_centers[center_name]
        return {
            'open_hour': center['open_hour'],
            'close_hour': center['close_hour']
        }
    
    def get_center_local_time(self, center_name):
        """Get the current local time for a specific center
        
        Args:
            center_name (str): Name of the trading center
            
        Returns:
            datetime: Current local time for the center, or None if center not found
        """
        if center_name not in self.trading_centers:
            logger.warning(f"Unknown trading center: {center_name}")
            return None
        
        center = self.trading_centers[center_name]
        center_tz = center['timezone']
        
        # Get current time in the center's timezone
        now_utc = datetime.now(pytz.UTC)
        now_center = now_utc.astimezone(center_tz)
        
        return now_center
    
    def get_centers_status_text(self):
        """Get a human-readable text about all trading centers' status
        
        Returns:
            str: Text describing the status of all trading centers
        """
        status = self.get_all_centers_status()
        
        text = "🌎 *GLOBAL FOREX MARKETS STATUS*\n\n"
        
        for center_name, is_open in status.items():
            local_time = self.get_center_local_time(center_name)
            local_time_str = local_time.strftime("%H:%M %Z") if local_time else "Unknown"
            
            if is_open:
                text += f"✅ *{center_name}*: OPEN (Local time: {local_time_str})\n"
            else:
                text += f"❌ *{center_name}*: CLOSED (Local time: {local_time_str})\n"
        
        # Add overall market status
        if any(status.values()):
            text += "\n🟢 At least one major market is open. Forex trading is available."
        else:
            text += "\n🔴 All major markets are closed. Forex trading may be limited."
        
        return text


if __name__ == "__main__":
    # Example usage
    markets = GlobalMarkets()
    print(markets.get_centers_status_text())
