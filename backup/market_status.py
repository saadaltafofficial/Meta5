#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for checking forex market open/close status"""

import logging
from datetime import datetime, time, timedelta
import pytz

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MarketStatus:
    """Class for checking forex market open/close status"""
    
    def __init__(self):
        """Initialize the market status checker"""
        # Define forex market trading hours
        # Forex market is open 24 hours a day, 5 days a week
        # It opens on Sunday 5:00 PM EST and closes on Friday 5:00 PM EST
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Define market holidays (major US holidays when forex trading is limited)
        self.holidays = [
            # 2025 holidays - update yearly
            datetime(2025, 1, 1),   # New Year's Day
            datetime(2025, 4, 18),  # Good Friday
            datetime(2025, 5, 26),  # Memorial Day
            datetime(2025, 7, 4),   # Independence Day
            datetime(2025, 9, 1),   # Labor Day
            datetime(2025, 11, 27), # Thanksgiving Day
            datetime(2025, 12, 25), # Christmas Day
        ]
    
    def is_market_open(self):
        """Check if the forex market is currently open
        
        Returns:
            bool: True if the market is open, False otherwise
        """
        # Get current time in EST
        now_utc = datetime.now(pytz.UTC)
        now_est = now_utc.astimezone(self.est_tz)
        
        # Check if today is a weekend
        # Forex market is closed from Friday 5 PM to Sunday 5 PM EST
        weekday = now_est.weekday()
        current_time = now_est.time()
        
        # Friday after 5 PM
        if weekday == 4 and current_time >= time(17, 0):
            return False
        
        # Saturday
        if weekday == 5:
            return False
        
        # Sunday before 5 PM
        if weekday == 6 and current_time < time(17, 0):
            return False
        
        # Check if today is a holiday
        today = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
        if any(today.date() == holiday.date() for holiday in self.holidays):
            return False
        
        # If none of the above conditions are met, the market is open
        return True
    
    def get_next_market_open(self):
        """Get the next market open time
        
        Returns:
            datetime: Next market open time in EST
        """
        # Get current time in EST
        now_utc = datetime.now(pytz.UTC)
        now_est = now_utc.astimezone(self.est_tz)
        
        # If market is already open, return current time
        if self.is_market_open():
            return now_est
        
        # Calculate next open time
        weekday = now_est.weekday()
        current_time = now_est.time()
        
        # Friday after 5 PM or Saturday - market opens on Sunday 5 PM
        if (weekday == 4 and current_time >= time(17, 0)) or weekday == 5:
            days_to_add = 6 - weekday
            next_open = now_est.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=days_to_add)
            return next_open
        
        # Sunday before 5 PM - market opens at 5 PM
        if weekday == 6 and current_time < time(17, 0):
            next_open = now_est.replace(hour=17, minute=0, second=0, microsecond=0)
            return next_open
        
        # If it's a holiday, find the next non-holiday weekday
        today = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
        if any(today.date() == holiday.date() for holiday in self.holidays):
            next_day = today + timedelta(days=1)
            while next_day.weekday() > 4 or any(next_day.date() == holiday.date() for holiday in self.holidays):
                next_day += timedelta(days=1)
            return next_day.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # This should not happen, but return current time as fallback
        return now_est
    
    def get_next_market_close(self):
        """Get the next market close time
        
        Returns:
            datetime: Next market close time in EST
        """
        # Get current time in EST
        now_utc = datetime.now(pytz.UTC)
        now_est = now_utc.astimezone(self.est_tz)
        
        # If market is already closed, return None
        if not self.is_market_open():
            return None
        
        # Calculate next close time (Friday 5 PM)
        weekday = now_est.weekday()
        days_to_add = 4 - weekday if weekday < 4 else 0
        next_close = now_est.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=days_to_add)
        
        # Check if there's a holiday before the calculated close time
        check_date = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
        while check_date.date() <= next_close.date():
            if any(check_date.date() == holiday.date() for holiday in self.holidays):
                return check_date.replace(hour=17, minute=0, second=0, microsecond=0) - timedelta(days=1)
            check_date += timedelta(days=1)
        
        return next_close
    
    def get_market_hours_text(self):
        """Get a human-readable text about market hours
        
        Returns:
            str: Text describing market hours
        """
        is_open = self.is_market_open()
        
        # Define Pakistan timezone
        pk_tz = pytz.timezone('Asia/Karachi')
        
        if is_open:
            next_close = self.get_next_market_close()
            next_close_str = next_close.strftime('%A, %B %d at %I:%M %p EST') if next_close else 'Unknown'
            
            # Convert to Pakistan time
            if next_close:
                next_close_pk = next_close.astimezone(pk_tz)
                next_close_pk_str = next_close_pk.strftime('%A, %B %d at %I:%M %p PKT')
                return f"The forex market is currently OPEN. It will close on {next_close_str} ({next_close_pk_str} in Pakistan)."
            else:
                return f"The forex market is currently OPEN. It will close on {next_close_str}."
        else:
            next_open = self.get_next_market_open()
            next_open_str = next_open.strftime('%A, %B %d at %I:%M %p EST')
            
            # Convert to Pakistan time
            next_open_pk = next_open.astimezone(pk_tz)
            next_open_pk_str = next_open_pk.strftime('%A, %B %d at %I:%M %p PKT')
            
            return f"The forex market is currently CLOSED. It will open on {next_open_str} ({next_open_pk_str} in Pakistan)."
