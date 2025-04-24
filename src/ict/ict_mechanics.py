#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICT (Inner Circle Trader) Mechanics Module

This module implements advanced ICT trading concepts including:
- Market Structure Shifts (MSS)
- Breaker Blocks
- Fair Value Gaps (FVG)
- Optimal Trade Entry (OTE)
- Killzones
- Liquidity Pools
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import pytz

# ======================== Market Structure Analysis ========================

def identify_market_structure(df):
    """Identify market structure based on ICT methodology
    
    Identifies higher highs (HH), higher lows (HL), lower highs (LH),
    and lower lows (LL) to determine market structure shifts.
    
    Args:
        df (pd.DataFrame): Price data with OHLC
        
    Returns:
        pd.DataFrame: DataFrame with market structure columns added
    """
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Initialize columns
        df['swing_high'] = False
        df['swing_low'] = False
        df['higher_high'] = False
        df['higher_low'] = False
        df['lower_high'] = False
        df['lower_low'] = False
        df['bullish_mss'] = False  # Market structure shift to bullish
        df['bearish_mss'] = False  # Market structure shift to bearish
        
        # Check for NaN values and fill them
        if df['high'].isna().any() or df['low'].isna().any():
            df['high'] = df['high'].fillna(method='ffill').fillna(method='bfill')
            df['low'] = df['low'].fillna(method='ffill').fillna(method='bfill')
        
        # Need at least 3 candles to identify market structure
        if len(df) < 3:
            return df
        
        # Identify swing points (placeholder — you may want to add logic here)
        swing_high_mask = pd.Series(False, index=df.index)
        swing_low_mask = pd.Series(False, index=df.index)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error initializing market structure analysis: {e}")
        return df

    # The rest of the logic goes here (outside of try-except)
    last_swing_high_idx = None
    last_swing_high_val = None
    last_swing_low_idx = None
    last_swing_low_val = None

    higher_high_mask = pd.Series(False, index=df.index)
    higher_low_mask = pd.Series(False, index=df.index)
    lower_high_mask = pd.Series(False, index=df.index)
    lower_low_mask = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        idx = df.index[i]
        if df['swing_high'].iloc[i]:
            if last_swing_high_idx is not None:
                if df['high'].iloc[i] > last_swing_high_val:
                    higher_high_mask.loc[idx] = True
                else:
                    lower_high_mask.loc[idx] = True
            last_swing_high_idx = i
            last_swing_high_val = df['high'].iloc[i]

        if df['swing_low'].iloc[i]:
            if last_swing_low_idx is not None:
                if df['low'].iloc[i] > last_swing_low_val:
                    higher_low_mask.loc[idx] = True
                else:
                    lower_low_mask.loc[idx] = True
            last_swing_low_idx = i
            last_swing_low_val = df['low'].iloc[i]

    df['higher_high'] = higher_high_mask
    df['higher_low'] = higher_low_mask
    df['lower_high'] = lower_high_mask
    df['lower_low'] = lower_low_mask

    # Initialize market structure shift masks
    bullish_mss_mask = pd.Series(False, index=df.index)
    bearish_mss_mask = pd.Series(False, index=df.index)
    
    # Initialize the bullish_mss and bearish_mss columns first to avoid access errors
    df['bullish_mss'] = bullish_mss_mask
    df['bearish_mss'] = bearish_mss_mask

    for i in range(4, len(df)):
        idx = df.index[i]
        if df['higher_low'].iloc[i-2]:
            recent_bullish_mss = False
            try:
                for k in range(i-3, i):
                    if 0 <= k < len(df) and df['bullish_mss'].iloc[k]:
                        recent_bullish_mss = True
                        break
            except Exception:
                recent_bullish_mss = False
            if not recent_bullish_mss:
                for j in range(i-3, 0, -1):
                    if df['swing_high'].iloc[j]:
                        if df['high'].iloc[i] > df['high'].iloc[j]:
                            bullish_mss_mask.loc[idx] = True
                        break

        if df['lower_high'].iloc[i-2]:
            recent_bearish_mss = False
            try:
                for k in range(i-3, i):
                    if 0 <= k < len(df) and df['bearish_mss'].iloc[k]:
                        recent_bearish_mss = True
                        break
            except Exception:
                recent_bearish_mss = False
            if not recent_bearish_mss:
                for j in range(i-3, 0, -1):
                    if df['swing_low'].iloc[j]:
                        if df['low'].iloc[i] < df['low'].iloc[j]:
                            bearish_mss_mask.loc[idx] = True
                        break

    df['bullish_mss'] = bullish_mss_mask
    df['bearish_mss'] = bearish_mss_mask

    return df


# ======================== Order Blocks and Breaker Blocks ========================

def identify_order_blocks(data):
    """
    Identify Order Blocks and Breaker Blocks according to ICT methodology
    
    Args:
        data (pd.DataFrame): Price data with OHLC values and market structure
        
    Returns:
        pd.DataFrame: Data with order blocks and breaker blocks identified
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize order block columns
    df['bullish_ob'] = False  # Bullish Order Block
    df['bearish_ob'] = False  # Bearish Order Block
    df['bullish_bb'] = False  # Bullish Breaker Block
    df['bearish_bb'] = False  # Bearish Breaker Block
    
    # Order block levels
    df['bullish_ob_top'] = np.nan
    df['bullish_ob_bottom'] = np.nan
    df['bearish_ob_top'] = np.nan
    df['bearish_ob_bottom'] = np.nan
    
    # Breaker block levels
    df['bullish_bb_top'] = np.nan
    df['bullish_bb_bottom'] = np.nan
    df['bearish_bb_top'] = np.nan
    df['bearish_bb_bottom'] = np.nan
    
    # Identify candle colors
    df['red_candle'] = df['close'] < df['open']
    df['green_candle'] = df['close'] > df['open']
    
    # Identify strong moves (using Market Structure Shifts)
    # A strong bullish move is indicated by a bullish MSS
    # A strong bearish move is indicated by a bearish MSS
    
    # Look for bullish order blocks (last red candle before a bullish MSS)
    for i in range(3, len(df)):
        if df['bullish_mss'].iloc[i]:
            # Look back for the last red candle
            for j in range(i-1, max(0, i-10), -1):  # Look back up to 10 candles
                if df['red_candle'].iloc[j]:
                    df.loc[j, 'bullish_ob'] = True
                    df.loc[j, 'bullish_ob_top'] = df['open'].iloc[j]  # Top of bullish OB is the open
                    df.loc[j, 'bullish_ob_bottom'] = df['close'].iloc[j]  # Bottom is the close
                    break
    
    # Look for bearish order blocks (last green candle before a bearish MSS)
    for i in range(3, len(df)):
        if df['bearish_mss'].iloc[i]:
            # Look back for the last green candle
            for j in range(i-1, max(0, i-10), -1):  # Look back up to 10 candles
                if df['green_candle'].iloc[j]:
                    df.loc[j, 'bearish_ob'] = True
                    df.loc[j, 'bearish_ob_top'] = df['close'].iloc[j]  # Top of bearish OB is the close
                    df.loc[j, 'bearish_ob_bottom'] = df['open'].iloc[j]  # Bottom is the open
                    break
    
    # Identify Breaker Blocks (Order Blocks that have been broken)
    for i in range(1, len(df)):
        # Check if any previous bullish order block has been broken
        if df['bullish_ob'].iloc[:i].any():
            # Get the most recent bullish order block
            for j in range(i-1, -1, -1):
                if df['bullish_ob'].iloc[j]:
                    # If price closes below the bottom of the bullish order block, it's broken
                    if df['close'].iloc[i] < df['bullish_ob_bottom'].iloc[j]:
                        # This becomes a bearish breaker block
                        df.loc[j, 'bearish_bb'] = True
                        df.loc[j, 'bearish_bb_top'] = df['bullish_ob_top'].iloc[j]
                        df.loc[j, 'bearish_bb_bottom'] = df['bullish_ob_bottom'].iloc[j]
                    break
        
        # Check if any previous bearish order block has been broken
        if df['bearish_ob'].iloc[:i].any():
            # Get the most recent bearish order block
            for j in range(i-1, -1, -1):
                if df['bearish_ob'].iloc[j]:
                    # If price closes above the top of the bearish order block, it's broken
                    if df['close'].iloc[i] > df['bearish_ob_top'].iloc[j]:
                        # This becomes a bullish breaker block
                        df.loc[j, 'bullish_bb'] = True
                        df.loc[j, 'bullish_bb_top'] = df['bearish_ob_top'].iloc[j]
                        df.loc[j, 'bullish_bb_bottom'] = df['bearish_ob_bottom'].iloc[j]
                    break
    
    return df

# ======================== Fair Value Gaps (FVG) ========================

def identify_fair_value_gaps(data):
    """
    Identify Fair Value Gaps (FVG) according to ICT methodology
    
    Args:
        data (pd.DataFrame): Price data with OHLC values

    Returns:
        pd.DataFrame: Data with FVGs identified
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize FVG columns
    df['bullish_fvg'] = False  # Bullish Fair Value Gap
    df['bearish_fvg'] = False  # Bearish Fair Value Gap
    
    # FVG levels
    df['bullish_fvg_top'] = np.nan
    df['bullish_fvg_bottom'] = np.nan
    df['bearish_fvg_top'] = np.nan
    df['bearish_fvg_bottom'] = np.nan  # ✅ This was missing in your original code
    
    # Create masks for assignments to avoid pandas warnings
    bullish_fvg_mask = pd.Series(False, index=df.index)
    bearish_fvg_mask = pd.Series(False, index=df.index)
    bullish_fvg_top = pd.Series(np.nan, index=df.index)
    bullish_fvg_bottom = pd.Series(np.nan, index=df.index)
    bearish_fvg_top = pd.Series(np.nan, index=df.index)
    bearish_fvg_bottom = pd.Series(np.nan, index=df.index)
    
    # Identify Fair Value Gaps
    for i in range(2, len(df)):
        # Bullish FVG: Current candle's low is higher than previous candle's high
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            middle_idx = df.index[i-1]
            bullish_fvg_mask.loc[middle_idx] = True
            bullish_fvg_top.loc[middle_idx] = df['low'].iloc[i]
            bullish_fvg_bottom.loc[middle_idx] = df['high'].iloc[i-2]
        
        # Bearish FVG: Current candle's high is lower than previous candle's low
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            middle_idx = df.index[i-1]
            bearish_fvg_mask.loc[middle_idx] = True
            bearish_fvg_top.loc[middle_idx] = df['low'].iloc[i-2]
            bearish_fvg_bottom.loc[middle_idx] = df['high'].iloc[i]
    
    # Assign the masks and values to the dataframe
    df['bullish_fvg'] = bullish_fvg_mask
    df['bullish_fvg_top'] = bullish_fvg_top
    df['bullish_fvg_bottom'] = bullish_fvg_bottom
    df['bearish_fvg'] = bearish_fvg_mask
    df['bearish_fvg_top'] = bearish_fvg_top
    df['bearish_fvg_bottom'] = bearish_fvg_bottom
    
    return df

# ======================== Liquidity Pools ========================

def identify_liquidity_pools(data):
    """
    Identify Liquidity Pools according to ICT methodology
    
    Args:
        data (pd.DataFrame): Price data with OHLC values and market structure
        
    Returns:
        pd.DataFrame: Data with liquidity pools identified
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize columns
    df['buy_liquidity'] = False
    df['sell_liquidity'] = False
    df['buy_liquidity_level'] = np.nan
    df['sell_liquidity_level'] = np.nan
    
    # Need at least 10 candles to identify liquidity pools
    if len(df) < 10:
        return df
    
    # Create masks for assignments
    buy_liquidity_mask = pd.Series(False, index=df.index)
    sell_liquidity_mask = pd.Series(False, index=df.index)
    buy_liquidity_levels = pd.Series(np.nan, index=df.index)
    sell_liquidity_levels = pd.Series(np.nan, index=df.index)
    
    # Ensure swing_low and swing_high columns exist
    if 'swing_low' not in df.columns:
        # Create a simple swing low detection if the column doesn't exist
        df['swing_low'] = False
        for i in range(2, len(df)-2):
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
               df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
                df.loc[df.index[i], 'swing_low'] = True
                
    if 'swing_high' not in df.columns:
        # Create a simple swing high detection if the column doesn't exist
        df['swing_high'] = False
        for i in range(2, len(df)-2):
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
               df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
                df.loc[df.index[i], 'swing_high'] = True
    
    # Identify Buy Liquidity (clusters of swing lows)
    for i in range(5, len(df) - 5):
        idx = df.index[i]  # Get the actual index for this position
        # Check if this is a swing low
        if df['swing_low'].iloc[i]:
            # Look for other swing lows within 0.1% price range
            current_low = df['low'].iloc[i]
            price_range = current_low * 0.001  # 0.1% range
            
            # Count nearby swing lows
            nearby_lows = 0
            for j in range(max(0, i-20), min(len(df), i+20)):
                if j != i and df['swing_low'].iloc[j]:
                    if abs(df['low'].iloc[j] - current_low) <= price_range:
                        nearby_lows += 1
            
            # If we have at least 2 nearby swing lows, it's a liquidity pool
            if nearby_lows >= 2:
                buy_liquidity_mask.loc[idx] = True
                buy_liquidity_levels.loc[idx] = current_low
    
    # Identify Sell Liquidity (clusters of swing highs)
    for i in range(5, len(df) - 5):
        idx = df.index[i]  # Get the actual index for this position
        # Check if this is a swing high
        if df['swing_high'].iloc[i]:
            # Look for other swing highs within 0.1% price range
            current_high = df['high'].iloc[i]
            price_range = current_high * 0.001  # 0.1% range
            
            # Count nearby swing highs
            nearby_highs = 0
            for j in range(max(0, i-20), min(len(df), i+20)):
                if j != i and df['swing_high'].iloc[j]:
                    if abs(df['high'].iloc[j] - current_high) <= price_range:
                        nearby_highs += 1
            
            # If we have at least 2 nearby swing highs, it's a liquidity pool
            if nearby_highs >= 2:
                sell_liquidity_mask.loc[idx] = True
                sell_liquidity_levels.loc[idx] = current_high
    
    # Assign the masks and levels to the dataframe
    df['buy_liquidity'] = buy_liquidity_mask
    df['buy_liquidity_level'] = buy_liquidity_levels
    df['sell_liquidity'] = sell_liquidity_mask
    df['sell_liquidity_level'] = sell_liquidity_levels
    
    return df

# ======================== Killzones ========================

def is_in_killzone(dt, timezone='UTC'):
    """
    Check if the given datetime is within an ICT killzone
    
    Args:
        dt (datetime): Datetime to check
        timezone (str): Timezone for the killzones
        
    Returns:
        tuple: (bool, str) - Whether in a killzone and which one
    """
    # Convert to the specified timezone
    tz = pytz.timezone(timezone)
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(tz)
    else:
        dt = dt.astimezone(tz)
    
    # Extract time
    t = dt.time()
    
    # Also convert to Pakistan time (UTC+5) for Pakistan-specific killzones
    pak_tz = pytz.timezone('Asia/Karachi')
    pak_dt = dt.astimezone(pak_tz)
    pak_t = pak_dt.time()
    
    # Define killzones in the specified timezone
    # London Open: 8:00-10:00 GMT/UTC
    london_open_start = time(8, 0)
    london_open_end = time(10, 0)
    
    # New York Open: 13:30-15:30 GMT/UTC
    ny_open_start = time(13, 30)
    ny_open_end = time(15, 30)
    
    # Asian Range: 22:00-2:00 GMT/UTC
    asian_range_start = time(22, 0)
    asian_range_end = time(2, 0)
    
    # Pakistan Prime Trading Hours (London-NY overlap): 17:00-22:00 Pakistan time
    # This captures the most active period when both London and NY markets are open
    pak_prime_start = time(17, 0)
    pak_prime_end = time(22, 0)
    
    # Pakistan Super Prime (peak of London-NY overlap): 19:00-21:00 Pakistan time
    # This is the absolute peak trading period with maximum liquidity
    pak_super_prime_start = time(19, 0)
    pak_super_prime_end = time(21, 0)
    
    # Pakistan Night Session (highest profitability): 19:00-3:00 Pakistan time
    # Extended to capture more of the NY session and early Asian session
    pak_night_start = time(19, 0)
    pak_night_end = time(3, 0)
    
    # Check if in Pakistan Super Prime (absolute highest priority)
    if pak_super_prime_start <= pak_t <= pak_super_prime_end:
        return True, "Pakistan Super Prime (Peak Liquidity)"
    
    # Check if in Pakistan Prime Trading Hours (high priority)
    if pak_prime_start <= pak_t <= pak_prime_end:
        return True, "Pakistan Prime Hours (London-NY Overlap)"
    
    # Check if in Pakistan Night Session (spans midnight)
    if pak_night_start <= pak_t or pak_t <= pak_night_end:
        return True, "Pakistan Night Session"
    
    # Check if in London Open killzone
    if london_open_start <= t <= london_open_end:
        return True, "London Open"
    
    # Check if in New York Open killzone
    if ny_open_start <= t <= ny_open_end:
        return True, "New York Open"
    
    # Check if in Asian Range killzone (spans midnight)
    if asian_range_start <= t or t <= asian_range_end:
        return True, "Asian Range"
    
    return False, None

# ======================== Optimal Trade Entry (OTE) ========================

def calculate_ote_levels(data):
    """
    Calculate Optimal Trade Entry (OTE) levels according to ICT methodology
    
    Args:
        data (pd.DataFrame): Price data with OHLC values, order blocks, and FVGs
        
    Returns:
        pd.DataFrame: Data with OTE levels calculated
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize columns
    df['bullish_ote'] = False
    df['bearish_ote'] = False
    df['bullish_ote_level'] = np.nan
    df['bearish_ote_level'] = np.nan
    
    # Create masks for assignments
    bullish_ote_mask = pd.Series(False, index=df.index)
    bearish_ote_mask = pd.Series(False, index=df.index)
    bullish_ote_levels = pd.Series(np.nan, index=df.index)
    bearish_ote_levels = pd.Series(np.nan, index=df.index)
    
    # Ensure required columns exist
    required_columns = ['bullish_ob', 'bearish_ob', 'bullish_ob_top', 'bullish_ob_bottom', 'bearish_ob_top', 'bearish_ob_bottom']
    for col in required_columns:
        if col not in df.columns:
            df[col] = False if col in ['bullish_ob', 'bearish_ob'] else np.nan
    
    # Calculate OTE levels based on order blocks and FVGs
    for i in range(len(df)):
        idx = df.index[i]  # Get the actual index for this position
        
        # Bullish OTE: 70% retracement into a bullish order block
        if df['bullish_ob'].iloc[i]:
            ob_top = df['bullish_ob_top'].iloc[i]
            ob_bottom = df['bullish_ob_bottom'].iloc[i]
            ob_height = ob_top - ob_bottom
            
            # OTE is 70% from bottom of the order block
            ote_level = ob_bottom + (ob_height * 0.3)  # 30% from bottom = 70% retracement
            
            # Mark this candle as having a bullish OTE level
            bullish_ote_mask.loc[idx] = True
            bullish_ote_levels.loc[idx] = ote_level
        
        # Bearish OTE: 70% retracement into a bearish order block
        if df['bearish_ob'].iloc[i]:
            ob_top = df['bearish_ob_top'].iloc[i]
            ob_bottom = df['bearish_ob_bottom'].iloc[i]
            ob_height = ob_top - ob_bottom
            
            # OTE is 70% from top of the order block
            ote_level = ob_top - (ob_height * 0.3)  # 30% from top = 70% retracement
            
            # Mark this candle as having a bearish OTE level
            bearish_ote_mask.loc[idx] = True
            bearish_ote_levels.loc[idx] = ote_level
    
    # Assign the masks and levels to the dataframe
    df['bullish_ote'] = bullish_ote_mask
    df['bullish_ote_level'] = bullish_ote_levels
    df['bearish_ote'] = bearish_ote_mask
    df['bearish_ote_level'] = bearish_ote_levels
    
    return df

# ======================== Daily Bias ========================

def determine_daily_bias(data):
    """
    Determine the daily bias based on higher timeframe analysis
    
    Args:
        data (pd.DataFrame): Price data with market structure
        
    Returns:
        str: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Check the most recent market structure shifts
    recent_rows = df.iloc[-20:]  # Look at the last 20 candles
    
    # Check for recent MSS
    if recent_rows['bullish_mss'].any():
        last_bullish_mss = recent_rows[recent_rows['bullish_mss']].index[-1]
        
        # If there's a more recent bearish MSS, use that instead
        if recent_rows['bearish_mss'].any():
            last_bearish_mss = recent_rows[recent_rows['bearish_mss']].index[-1]
            if last_bearish_mss > last_bullish_mss:
                return 'BEARISH'
        
        return 'BULLISH'
    
    elif recent_rows['bearish_mss'].any():
        return 'BEARISH'
    
    # If no clear MSS, check the overall trend using moving averages
    if 'ma_50' in df.columns and 'ma_200' in df.columns:
        last_row = df.iloc[-1]
        if last_row['ma_50'] > last_row['ma_200']:
            return 'BULLISH'
        elif last_row['ma_50'] < last_row['ma_200']:
            return 'BEARISH'
    
    # If no clear bias, check the recent price action
    if df['close'].iloc[-1] > df['open'].iloc[-1] and df['close'].iloc[-2] > df['open'].iloc[-2]:
        return 'BULLISH'
    elif df['close'].iloc[-1] < df['open'].iloc[-1] and df['close'].iloc[-2] < df['open'].iloc[-2]:
        return 'BEARISH'
    
    return 'NEUTRAL'

# ======================== ICT 2024 Concepts ========================

def identify_relative_equal_levels(data, lookback=10, tolerance=0.0002):
    """Identify relative equal highs and lows according to ICT 2024 methodology
    
    This function identifies areas where price forms relatively equal highs or lows,
    which are key components of the ICT 2024 trading model.
    
    Args:
        data (pd.DataFrame): Price data with OHLC values
        lookback (int): Number of candles to look back for identifying relative equal levels
        tolerance (float): Percentage tolerance for considering levels as 'equal'
        
    Returns:
        pd.DataFrame: Data with relative equal levels identified
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize columns
    df['rel_equal_high'] = False
    df['rel_equal_low'] = False
    df['rel_equal_high_level'] = np.nan
    df['rel_equal_low_level'] = np.nan
    df['liquidity_grabbed_high'] = False
    df['liquidity_grabbed_low'] = False
    
    # Need at least lookback+2 candles to identify relative equal levels
    if len(df) < lookback + 2:
        return df
    
    # Create masks for assignments to avoid pandas warnings
    rel_equal_high_mask = pd.Series(False, index=df.index)
    rel_equal_low_mask = pd.Series(False, index=df.index)
    rel_equal_high_level = pd.Series(np.nan, index=df.index)
    rel_equal_low_level = pd.Series(np.nan, index=df.index)
    liquidity_grabbed_high_mask = pd.Series(False, index=df.index)
    liquidity_grabbed_low_mask = pd.Series(False, index=df.index)
    
    # Loop through the data to identify relative equal levels
    for i in range(lookback, len(df) - 2):
        # Get current index
        current_idx = df.index[i]
        
        # Get the high/low values in the lookback window
        highs = df['high'].iloc[i-lookback:i]
        lows = df['low'].iloc[i-lookback:i]
        
        # Find the maximum high and minimum low in the lookback window
        max_high = highs.max()
        min_low = lows.min()
        
        # Check for relative equal highs (within tolerance)
        high_count = sum(abs(h - max_high) / max_high < tolerance for h in highs if not pd.isna(h))
        if high_count >= 2:  # At least 2 highs within tolerance range
            rel_equal_high_mask.loc[current_idx] = True
            rel_equal_high_level.loc[current_idx] = max_high
            
            # Check if liquidity was grabbed (price exceeded the relative equal high)
            if i + 1 < len(df) and df['high'].iloc[i+1] > max_high * (1 + tolerance/2):
                liquidity_grabbed_high_mask.loc[df.index[i+1]] = True
        
        # Check for relative equal lows (within tolerance)
        low_count = sum(abs(l - min_low) / min_low < tolerance for l in lows if not pd.isna(l))
        if low_count >= 2:  # At least 2 lows within tolerance range
            rel_equal_low_mask.loc[current_idx] = True
            rel_equal_low_level.loc[current_idx] = min_low
            
            # Check if liquidity was grabbed (price dropped below the relative equal low)
            if i + 1 < len(df) and df['low'].iloc[i+1] < min_low * (1 - tolerance/2):
                liquidity_grabbed_low_mask.loc[df.index[i+1]] = True
    
    # Assign the masks and values to the dataframe
    df['rel_equal_high'] = rel_equal_high_mask
    df['rel_equal_low'] = rel_equal_low_mask
    df['rel_equal_high_level'] = rel_equal_high_level
    df['rel_equal_low_level'] = rel_equal_low_level
    df['liquidity_grabbed_high'] = liquidity_grabbed_high_mask
    df['liquidity_grabbed_low'] = liquidity_grabbed_low_mask
    
    return df

# ======================== ICT 2024 PD-Array ========================

def create_pd_array(data, bias='BULLISH', lookback=20, current_time=None):
    """Create a Potential Delivery Array (PD-Array) according to ICT 2024 methodology
    
    This function identifies multiple potential entry points (order blocks, breaker blocks,
    and relative equal levels) and combines them into a PD-array for high-probability entries.
    
    Args:
        data (pd.DataFrame): Price data with all ICT components identified
        bias (str): Trading bias ('BULLISH' or 'BEARISH')
        lookback (int): Number of candles to look back for identifying entry points
        current_time (datetime, optional): Current time for checking killzones
        
    Returns:
        dict: PD-array with entry points and their confidence levels
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure all required columns exist before processing
    required_columns = ['bullish_fvg', 'bearish_fvg', 'bullish_ob', 'bearish_ob', 'rel_equal_high', 'rel_equal_low']
    for col in required_columns:
        if col not in df.columns:
            df[col] = False
            
    # Ensure all required level columns exist
    level_columns = ['bullish_fvg_top', 'bullish_fvg_bottom', 'bearish_fvg_top', 'bearish_fvg_bottom', 
                     'bullish_ob_top', 'bullish_ob_bottom', 'bearish_ob_top', 'bearish_ob_bottom']
    for col in level_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    # Initialize PD-array dictionary
    pd_array = {
        'bias': bias,
        'entries': [],
        'best_entry': None,
        'stop_loss': None,
        'take_profit': None,
        'in_pakistan_night': False,
        'in_pakistan_super_prime': False
    }
    
    # Check if current time is in Pakistan night hours (for confidence boost)
    if current_time is not None:
        in_killzone, killzone_name = is_in_killzone(current_time)
        pd_array['in_killzone'] = in_killzone
        pd_array['killzone_name'] = killzone_name
        
        # Check specifically for Pakistan night hours
        if killzone_name in ["Pakistan Prime Hours (London-NY Overlap)", "Pakistan Night Session"]:
            pd_array['in_pakistan_night'] = True
            
        # Check specifically for Pakistan Super Prime hours (peak liquidity)
        if killzone_name == "Pakistan Super Prime (Peak Liquidity)":
            pd_array['in_pakistan_super_prime'] = True
            pd_array['in_pakistan_night'] = True  # Super Prime is also part of night hours
    
    # Check if we have enough data
    if len(df) < lookback:
        return pd_array
    
    # Get recent data for analysis
    recent_df = df.iloc[-lookback:]
    
    # Identify entry points based on bias
    if bias == 'BULLISH':
        # 1. Check for bullish order blocks
        if 'bullish_ob' in df.columns and recent_df['bullish_ob'].any():
            ob_indices = recent_df[recent_df['bullish_ob']].index
            for idx in ob_indices:
                # Base confidence level
                confidence = 0.7
                
                # Boost confidence if in Pakistan Super Prime hours (highest priority)
                if pd_array['in_pakistan_super_prime']:
                    confidence += 0.25  # +25% confidence boost during Pakistan Super Prime hours
                # Boost confidence if in Pakistan night hours
                elif pd_array['in_pakistan_night']:
                    confidence += 0.15  # +15% confidence boost during Pakistan night hours
                
                entry = {
                    'type': 'Bullish Order Block',
                    'level': df.loc[idx, 'bullish_ob_top'],
                    'confidence': confidence,
                    'index': idx
                }
                pd_array['entries'].append(entry)
        
        # 2. Check for bullish breaker blocks (higher confidence)
        if 'bullish_bb' in df.columns and recent_df['bullish_bb'].any():
            bb_indices = recent_df[recent_df['bullish_bb']].index
            for idx in bb_indices:
                # Base confidence level
                confidence = 0.85
                
                # Boost confidence if in Pakistan Super Prime hours (highest priority)
                if pd_array['in_pakistan_super_prime']:
                    confidence += 0.14  # +14% confidence boost during Pakistan Super Prime hours
                # Boost confidence if in Pakistan night hours
                elif pd_array['in_pakistan_night']:
                    confidence += 0.1  # +10% confidence boost during Pakistan night hours
                
                entry = {
                    'type': 'Bullish Breaker Block',
                    'level': df.loc[idx, 'bullish_bb_top'],
                    'confidence': confidence,
                    'index': idx
                }
                pd_array['entries'].append(entry)
        
        # 3. Check for relative equal lows with liquidity grabbed (ICT 2024 concept)
        if 'rel_equal_low' in df.columns and recent_df['rel_equal_low'].any():
            rel_indices = recent_df[recent_df['rel_equal_low']].index
            for idx in rel_indices:
                # Check if this level had its liquidity grabbed
                liquidity_grabbed = False
                for i in range(df.index.get_loc(idx) + 1, min(df.index.get_loc(idx) + 5, len(df))):
                    if df['liquidity_grabbed_low'].iloc[i]:
                        liquidity_grabbed = True
                        break
                
                if liquidity_grabbed:
                    # Base confidence level
                    confidence = 0.9  # Higher confidence for ICT 2024 concept
                    
                    # Boost confidence if in Pakistan Super Prime hours (highest priority)
                    if pd_array['in_pakistan_super_prime']:
                        confidence = min(confidence + 0.09, 0.99)  # +9% confidence boost during Pakistan Super Prime hours, capped at 0.99
                    # Boost confidence if in Pakistan night hours
                    elif pd_array['in_pakistan_night']:
                        confidence = min(confidence + 0.08, 0.99)  # +8% confidence boost during Pakistan night hours, capped at 0.99
                    
                    entry = {
                        'type': 'Equal Lows (Liquidity Grabbed)',
                        'level': df.loc[idx, 'rel_equal_low_level'],
                        'confidence': confidence,
                        'index': idx
                    }
                    pd_array['entries'].append(entry)
    
    elif bias == 'BEARISH':
        # 1. Check for bearish order blocks
        if 'bearish_ob' in df.columns and recent_df['bearish_ob'].any():
            ob_indices = recent_df[recent_df['bearish_ob']].index
            for idx in ob_indices:
                # Base confidence level
                confidence = 0.7
                
                # Boost confidence if in Pakistan Super Prime hours (highest priority)
                if pd_array['in_pakistan_super_prime']:
                    confidence += 0.25  # +25% confidence boost during Pakistan Super Prime hours
                # Boost confidence if in Pakistan night hours
                elif pd_array['in_pakistan_night']:
                    confidence += 0.15  # +15% confidence boost during Pakistan night hours
                
                entry = {
                    'type': 'Bearish Order Block',
                    'level': df.loc[idx, 'bearish_ob_bottom'],
                    'confidence': confidence,
                    'index': idx
                }
                pd_array['entries'].append(entry)
        
        # 2. Check for bearish breaker blocks (higher confidence)
        if 'bearish_bb' in df.columns and recent_df['bearish_bb'].any():
            bb_indices = recent_df[recent_df['bearish_bb']].index
            for idx in bb_indices:
                # Base confidence level
                confidence = 0.85
                
                # Boost confidence if in Pakistan Super Prime hours (highest priority)
                if pd_array['in_pakistan_super_prime']:
                    confidence += 0.14  # +14% confidence boost during Pakistan Super Prime hours
                # Boost confidence if in Pakistan night hours
                elif pd_array['in_pakistan_night']:
                    confidence += 0.1  # +10% confidence boost during Pakistan night hours
                
                entry = {
                    'type': 'Bearish Breaker Block',
                    'level': df.loc[idx, 'bearish_bb_bottom'],
                    'confidence': confidence,
                    'index': idx
                }
                pd_array['entries'].append(entry)
        
        # 3. Check for relative equal highs with liquidity grabbed (ICT 2024 concept)
        if 'rel_equal_high' in df.columns and recent_df['rel_equal_high'].any():
            rel_indices = recent_df[recent_df['rel_equal_high']].index
            for idx in rel_indices:
                # Check if this level had its liquidity grabbed
                liquidity_grabbed = False
                for i in range(df.index.get_loc(idx) + 1, min(df.index.get_loc(idx) + 5, len(df))):
                    if df['liquidity_grabbed_high'].iloc[i]:
                        liquidity_grabbed = True
                        break
                
                if liquidity_grabbed:
                    # Base confidence level
                    confidence = 0.9  # Higher confidence for ICT 2024 concept
                    
                    # Boost confidence if in Pakistan Super Prime hours (highest priority)
                    if pd_array['in_pakistan_super_prime']:
                        confidence = min(confidence + 0.09, 0.99)  # +9% confidence boost during Pakistan Super Prime hours, capped at 0.99
                    # Boost confidence if in Pakistan night hours
                    elif pd_array['in_pakistan_night']:
                        confidence = min(confidence + 0.08, 0.99)  # +8% confidence boost during Pakistan night hours, capped at 0.99
                    
                    entry = {
                        'type': 'Equal Highs (Liquidity Grabbed)',
                        'level': df.loc[idx, 'rel_equal_high_level'],
                        'confidence': confidence,
                        'index': idx
                    }
                    pd_array['entries'].append(entry)
    
    # Find the best entry point (highest confidence)
    if pd_array['entries']:
        best_entry = max(pd_array['entries'], key=lambda x: x['confidence'])
        pd_array['best_entry'] = best_entry
        
        # Set stop loss and take profit based on the best entry
        if bias == 'BULLISH':
            # For bullish trades, stop loss is below the entry level
            # Find the lowest low in the last 5 candles or use a percentage-based stop
            if len(df) >= 5:
                pd_array['stop_loss'] = df['low'].iloc[-5:].min() * 0.998  # 0.2% below the lowest low
            else:
                pd_array['stop_loss'] = best_entry['level'] * 0.995  # 0.5% below entry
            
            # Take profit is based on risk:reward ratio (1.5R, 2.5R, 4.0R)
            risk = best_entry['level'] - pd_array['stop_loss']
            pd_array['take_profit_1'] = best_entry['level'] + (risk * 1.5)
            pd_array['take_profit_2'] = best_entry['level'] + (risk * 2.5)
            pd_array['take_profit_3'] = best_entry['level'] + (risk * 4.0)
            pd_array['take_profit'] = pd_array['take_profit_1']  # Default to first TP
        
        elif bias == 'BEARISH':
            # For bearish trades, stop loss is above the entry level
            # Find the highest high in the last 5 candles or use a percentage-based stop
            if len(df) >= 5:
                pd_array['stop_loss'] = df['high'].iloc[-5:].max() * 1.002  # 0.2% above the highest high
            else:
                pd_array['stop_loss'] = best_entry['level'] * 1.005  # 0.5% above entry
            
            # Take profit is based on risk:reward ratio (1.5R, 2.5R, 4.0R)
            risk = pd_array['stop_loss'] - best_entry['level']
            pd_array['take_profit_1'] = best_entry['level'] - (risk * 1.5)
            pd_array['take_profit_2'] = best_entry['level'] - (risk * 2.5)
            pd_array['take_profit_3'] = best_entry['level'] - (risk * 4.0)
            pd_array['take_profit'] = pd_array['take_profit_1']  # Default to first TP
    
    return pd_array

# ======================== ICT Trade Setup ========================

def evaluate_flexible_duration_setup(df, timeframe='H4', daily_bias='NEUTRAL', current_time=None):
    """Evaluate price data for ICT trade setups with flexible duration
    
    This function identifies high-probability ICT setups that can be either short-term
    or long-term trades, depending on market conditions. It strictly follows ICT rules
    and prioritizes setups with multiple confluence factors.
    
    Args:
        df (pd.DataFrame): Price data with ICT indicators
        timeframe (str): Timeframe of the data (e.g., 'M5', 'M15', 'H1', 'H4', 'D1')
        daily_bias (str): Daily bias from higher timeframe
        current_time (datetime, optional): Current time for killzone analysis
        
    Returns:
        dict: Trade setup details with expected duration and exit strategy
    """
    # Initialize setup dictionary with a baseline confidence
    setup = {
        'action': 'HOLD',
        'confidence': 0.1,  # Added baseline confidence instead of 0
        'setup_type': 'Flexible ICT Analysis',
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'expected_duration': 'unknown',
        'exit_strategy': 'fixed',  # 'fixed', 'trailing', or 'dynamic'
        'details': {
            'daily_bias': daily_bias,
            'in_killzone': False,
            'killzone_name': None,
            'timeframe': timeframe,
            'confluence_factors': []
        }
    }
    
    # Check if we're in a killzone
    in_killzone, killzone_name = False, None
    if current_time is not None:
        in_killzone, killzone_name = is_in_killzone(current_time)
    
    setup['details']['in_killzone'] = in_killzone
    setup['details']['killzone_name'] = killzone_name
    
    # Determine expected duration based on timeframe
    if timeframe in ['M5', 'M15']:
        setup['expected_duration'] = 'hours'
        setup['exit_strategy'] = 'dynamic'  # More aggressive exits for short-term trades
    elif timeframe in ['H1', 'H4']:
        setup['expected_duration'] = 'days'
        setup['exit_strategy'] = 'trailing'  # Trailing stops for medium-term trades
    elif timeframe in ['D1']:
        setup['expected_duration'] = 'weeks'
        setup['exit_strategy'] = 'fixed'  # Fixed targets for long-term trades
    
    # Get recent data for analysis (last 50 candles)
    recent_df = df.iloc[-50:].copy() if len(df) > 50 else df.copy()
    
    # Check for ICT setups based on the timeframe
    # 1. Check for order blocks and breaker blocks
    has_bullish_ob = recent_df['bullish_ob'].iloc[-15:].any() if 'bullish_ob' in recent_df.columns else False
    has_bearish_ob = recent_df['bearish_ob'].iloc[-15:].any() if 'bearish_ob' in recent_df.columns else False
    has_bullish_breaker = recent_df['bullish_breaker'].iloc[-15:].any() if 'bullish_breaker' in recent_df.columns else False
    has_bearish_breaker = recent_df['bearish_breaker'].iloc[-15:].any() if 'bearish_breaker' in recent_df.columns else False
    
    # 2. Check for fair value gaps (FVG)
    has_bullish_fvg = recent_df['bullish_fvg'].iloc[-15:].any() if 'bullish_fvg' in recent_df.columns else False
    has_bearish_fvg = recent_df['bearish_fvg'].iloc[-15:].any() if 'bearish_fvg' in recent_df.columns else False
    
    # 3. Check for liquidity grabs (ICT 2024 concept)
    has_liquidity_grabbed_high = recent_df['liquidity_grabbed_high'].iloc[-10:].any() if 'liquidity_grabbed_high' in recent_df.columns else False
    has_liquidity_grabbed_low = recent_df['liquidity_grabbed_low'].iloc[-10:].any() if 'liquidity_grabbed_low' in recent_df.columns else False
    
    # 4. Check for relative equal highs/lows (ICT 2024 concept)
    has_rel_equal_high = recent_df['rel_equal_high'].iloc[-10:].any() if 'rel_equal_high' in recent_df.columns else False
    has_rel_equal_low = recent_df['rel_equal_low'].iloc[-10:].any() if 'rel_equal_low' in recent_df.columns else False
    
    # 5. Check for market structure shifts
    # Ensure all required market structure columns exist
    required_columns = ['bullish_mss', 'bearish_mss', 'swing_high', 'swing_low', 'higher_high', 'higher_low', 'lower_high', 'lower_low']
    for col in required_columns:
        if col not in recent_df.columns:
            recent_df[col] = False
    
    # Safely check for market structure shifts
    try:
        has_bullish_mss = recent_df['bullish_mss'].iloc[-15:].any()
        has_bearish_mss = recent_df['bearish_mss'].iloc[-15:].any()
    except Exception:
        # Default to False if any error occurs
        has_bullish_mss = False
        has_bearish_mss = False
    
    # Evaluate BULLISH setups (strict ICT rules)
    if daily_bias in ['BULLISH', 'NEUTRAL']:
        # Base confidence level
        base_confidence = 0.0
        premium_confidence = 0.0
        
        # Confluence factors for bullish setup
        confluence_factors = []
        
        # ICT 2024 Premium Setup: Relative Equal Lows with Liquidity Grab
        if has_rel_equal_low and has_liquidity_grabbed_low:
            premium_confidence += 0.7  # High confidence for this premium setup
            confluence_factors.append("Relative Equal Lows with Liquidity Grab")
            setup['action'] = 'BUY'
            setup['setup_type'] = 'ICT 2024 Premium Buy'
            
            # Find the entry level (above the relative equal low)
            if 'rel_equal_low_level' in recent_df.columns:
                rel_equal_idx = recent_df[recent_df['rel_equal_low']].iloc[-1].name
                setup['entry'] = recent_df.loc[rel_equal_idx, 'rel_equal_low_level'] * 1.001  # Slightly above
                
                # Set tight stop loss below the liquidity grab
                if 'liquidity_grabbed_low_level' in recent_df.columns:
                    liq_grab_idx = recent_df[recent_df['liquidity_grabbed_low']].iloc[-1].name
                    setup['stop_loss'] = recent_df.loc[liq_grab_idx, 'liquidity_grabbed_low_level'] * 0.999  # Slightly below
        
        # ICT Premium Setup: Bullish Breaker Block
        elif has_bullish_breaker:
            premium_confidence += 0.65  # High confidence for breaker block
            confluence_factors.append("Bullish Breaker Block")
            setup['action'] = 'BUY'
            setup['setup_type'] = 'Bullish Breaker'
            
            # Find the entry level (above the breaker block)
            breaker_idx = recent_df[recent_df['bullish_breaker']].iloc[-1].name
            setup['entry'] = recent_df.loc[breaker_idx, 'high'] * 1.001  # Slightly above high
            setup['stop_loss'] = recent_df.loc[breaker_idx, 'low'] * 0.998  # Below low with buffer
        
        # ICT Setup: Bullish Order Block with FVG
        elif has_bullish_ob and has_bullish_fvg:
            premium_confidence += 0.6  # Good confidence for OB+FVG combo
            confluence_factors.append("Bullish Order Block with FVG")
            setup['action'] = 'BUY'
            setup['setup_type'] = 'Bullish OB+FVG'
            
            # Find the entry level (at the order block)
            ob_idx = recent_df[recent_df['bullish_ob']].iloc[-1].name
            setup['entry'] = recent_df.loc[ob_idx, 'high'] * 1.001  # Slightly above high
            setup['stop_loss'] = recent_df.loc[ob_idx, 'low'] * 0.998  # Below low with buffer
        
        # Add confluence factors to boost confidence
        if has_bullish_mss:
            premium_confidence += 0.1  # Bonus for bullish market structure shift
            confluence_factors.append("Bullish Market Structure Shift")
        
        # Bonus for trading in a killzone (high institutional activity)
        if in_killzone:
            # Give highest bonus for Pakistan Super Prime hours (absolute peak liquidity)
            if killzone_name == "Pakistan Super Prime (Peak Liquidity)":
                premium_confidence += 0.3  # Triple bonus for Pakistan Super Prime hours
            # Give higher bonus for Pakistan Prime/Night hours (when the bot is most profitable)
            elif killzone_name in ["Pakistan Prime Hours (London-NY Overlap)", "Pakistan Night Session"]:
                premium_confidence += 0.2  # Double bonus for Pakistan night hours
            else:
                premium_confidence += 0.1
        
        # Only proceed if we have a valid action and entry
        if setup['action'] == 'BUY' and setup['entry'] is not None and setup['stop_loss'] is not None:
            # Set final confidence level
            setup['confidence'] = min(premium_confidence, 0.99)  # Cap at 0.99
            
            # Calculate risk and set take profit levels
            risk = abs(setup['entry'] - setup['stop_loss'])
            
            # Set take profit based on expected duration
            if setup['expected_duration'] == 'hours':
                # Shorter targets for intraday trades
                setup['take_profit_1'] = setup['entry'] + (risk * 1.0)
                setup['take_profit_2'] = setup['entry'] + (risk * 1.5)
                setup['take_profit_3'] = setup['entry'] + (risk * 2.5)
            elif setup['expected_duration'] == 'days':
                # Medium targets for swing trades
                setup['take_profit_1'] = setup['entry'] + (risk * 1.5)
                setup['take_profit_2'] = setup['entry'] + (risk * 2.75)
                setup['take_profit_3'] = setup['entry'] + (risk * 4.0)
            else:  # 'weeks'
                # Larger targets for position trades
                setup['take_profit_1'] = setup['entry'] + (risk * 2.0)
                setup['take_profit_2'] = setup['entry'] + (risk * 3.5)
                setup['take_profit_3'] = setup['entry'] + (risk * 5.5)
            
            # Set default take profit to the first level
            setup['take_profit'] = setup['take_profit_1']
            
            # Store confluence factors in details
            setup['details']['confluence_factors'] = confluence_factors
    
    # Evaluate BEARISH setups (strict ICT rules)
    elif daily_bias in ['BEARISH', 'NEUTRAL']:
        # Base confidence level
        base_confidence = 0.0
        premium_confidence = 0.0
        
        # Confluence factors for bearish setup
        confluence_factors = []
        
        # ICT 2024 Premium Setup: Relative Equal Highs with Liquidity Grab
        if has_rel_equal_high and has_liquidity_grabbed_high:
            premium_confidence += 0.7  # High confidence for this premium setup
            confluence_factors.append("Relative Equal Highs with Liquidity Grab")
            setup['action'] = 'SELL'
            setup['setup_type'] = 'ICT 2024 Premium Sell'
            
            # Find the entry level (below the relative equal high)
            if 'rel_equal_high_level' in recent_df.columns:
                rel_equal_idx = recent_df[recent_df['rel_equal_high']].iloc[-1].name
                setup['entry'] = recent_df.loc[rel_equal_idx, 'rel_equal_high_level'] * 0.999  # Slightly below
                
                # Set tight stop loss above the liquidity grab
                if 'liquidity_grabbed_high_level' in recent_df.columns:
                    liq_grab_idx = recent_df[recent_df['liquidity_grabbed_high']].iloc[-1].name
                    setup['stop_loss'] = recent_df.loc[liq_grab_idx, 'liquidity_grabbed_high_level'] * 1.001  # Slightly above
        
        # ICT Premium Setup: Bearish Breaker Block
        elif has_bearish_breaker:
            premium_confidence += 0.65  # High confidence for breaker block
            confluence_factors.append("Bearish Breaker Block")
            setup['action'] = 'SELL'
            setup['setup_type'] = 'Bearish Breaker'
            
            # Find the entry level (below the breaker block)
            breaker_idx = recent_df[recent_df['bearish_breaker']].iloc[-1].name
            setup['entry'] = recent_df.loc[breaker_idx, 'low'] * 0.999  # Slightly below low
            setup['stop_loss'] = recent_df.loc[breaker_idx, 'high'] * 1.002  # Above high with buffer
        
        # ICT Setup: Bearish Order Block with FVG
        elif has_bearish_ob and has_bearish_fvg:
            premium_confidence += 0.6  # Good confidence for OB+FVG combo
            confluence_factors.append("Bearish Order Block with FVG")
            setup['action'] = 'SELL'
            setup['setup_type'] = 'Bearish OB+FVG'
            
            # Find the entry level (at the order block)
            ob_idx = recent_df[recent_df['bearish_ob']].iloc[-1].name
            setup['entry'] = recent_df.loc[ob_idx, 'low'] * 0.999  # Slightly below low
            setup['stop_loss'] = recent_df.loc[ob_idx, 'high'] * 1.002  # Above high with buffer
        
        # Add confluence factors to boost confidence
        if has_bearish_mss:
            premium_confidence += 0.1  # Bonus for bearish market structure shift
            confluence_factors.append("Bearish Market Structure Shift")
        
        # Bonus for trading in a killzone (high institutional activity)
        if in_killzone:
            # Give highest bonus for Pakistan Super Prime hours (absolute peak liquidity)
            if killzone_name == "Pakistan Super Prime (Peak Liquidity)":
                premium_confidence += 0.3  # Triple bonus for Pakistan Super Prime hours
            # Give higher bonus for Pakistan Prime/Night hours (when the bot is most profitable)
            elif killzone_name in ["Pakistan Prime Hours (London-NY Overlap)", "Pakistan Night Session"]:
                premium_confidence += 0.2  # Double bonus for Pakistan night hours
            else:
                premium_confidence += 0.1
        
        # Only proceed if we have a valid action and entry
        if setup['action'] == 'SELL' and setup['entry'] is not None and setup['stop_loss'] is not None:
            # Set final confidence level
            setup['confidence'] = min(premium_confidence, 0.99)  # Cap at 0.99
            
            # Calculate risk and set take profit levels
            risk = abs(setup['entry'] - setup['stop_loss'])
            
            # Set take profit based on expected duration
            if setup['expected_duration'] == 'hours':
                # Shorter targets for intraday trades
                setup['take_profit_1'] = setup['entry'] - (risk * 1.0)
                setup['take_profit_2'] = setup['entry'] - (risk * 1.5)
                setup['take_profit_3'] = setup['entry'] - (risk * 2.5)
            elif setup['expected_duration'] == 'days':
                # Medium targets for swing trades
                setup['take_profit_1'] = setup['entry'] - (risk * 1.5)
                setup['take_profit_2'] = setup['entry'] - (risk * 2.75)
                setup['take_profit_3'] = setup['entry'] - (risk * 4.0)
            else:  # 'weeks'
                # Larger targets for position trades
                setup['take_profit_1'] = setup['entry'] - (risk * 2.0)
                setup['take_profit_2'] = setup['entry'] - (risk * 3.5)
                setup['take_profit_3'] = setup['entry'] - (risk * 5.5)
            
            # Set default take profit to the first level
            setup['take_profit'] = setup['take_profit_1']
            
            # Store confluence factors in details
            setup['details']['confluence_factors'] = confluence_factors
    
    return setup

def evaluate_ict_setup(data, daily_bias=None):
    """
    Evaluate the current chart for ICT trade setups based on the 2022 ICT Mentorship methodology
    
    Args:
        data (pd.DataFrame): Price data with all ICT components identified
        daily_bias (str, optional): Override the daily bias
        
    Returns:
        dict: Trade setup evaluation with action, confidence, and details
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Initialize setup dictionary with a small baseline confidence
    setup = {
        'action': 'HOLD',
        'confidence': 0.1,  # Added baseline confidence instead of 0
        'setup_type': 'Basic ICT Analysis',
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'details': {},
        'risk_reward': None
    }
    
    # If no data or not enough data, return HOLD
    if df is None or len(df) < 20:
        return setup
    
    # Get the most recent data (last 50 candles for more context)
    recent_df = df.iloc[-50:]
    
    # If daily_bias is not provided, determine it from the data
    if daily_bias is None:
        daily_bias = determine_daily_bias(df)
    
    # Check if we're in a killzone (high institutional activity periods)
    in_killzone, killzone_name = is_in_killzone(datetime.now())
    
    # Add killzone information to setup details
    setup['details']['in_killzone'] = in_killzone
    setup['details']['killzone_name'] = killzone_name if in_killzone else None
    setup['details']['daily_bias'] = daily_bias
    
    # === PREMIUM ICT SETUPS (2022 Mentorship) ===
    
    # 1. PREMIUM BUY SETUP: Bullish Order Block + Fair Value Gap + Market Structure Shift
    if daily_bias in ['BULLISH', 'NEUTRAL']:
        # Check for bullish market structure shift (MSS)
        # Ensure bullish_mss column exists
        if 'bullish_mss' not in recent_df.columns:
            recent_df['bullish_mss'] = False
            
        # Safely check for market structure shifts
        try:
            has_bullish_mss = recent_df['bullish_mss'].iloc[-15:].any()
        except Exception:
            # Default to False if any error occurs
            has_bullish_mss = False
        
        # Check for bullish order blocks
        has_bullish_ob = recent_df['bullish_ob'].iloc[-15:].any()
        
        # Check for bullish fair value gaps
        has_bullish_fvg = recent_df['bullish_fvg'].iloc[-15:].any()
        
        # Check for optimal trade entry (OTE)
        has_bullish_ote = recent_df['bullish_ote'].iloc[-10:].any()
        
        # Calculate premium setup confidence
        premium_confidence = 0.0
        
        # Base confidence from daily bias
        if daily_bias == 'BULLISH':
            premium_confidence += 0.3
        elif daily_bias == 'NEUTRAL':
            premium_confidence += 0.1
        
        # Add confidence for each component
        if has_bullish_mss:
            premium_confidence += 0.2
            setup['details']['bullish_mss'] = True
        
        if has_bullish_ob:
            premium_confidence += 0.15
            setup['details']['bullish_ob'] = True
            
            # Get the most recent bullish order block
            if not recent_df[recent_df['bullish_ob']].iloc[-15:].empty:
                ob_idx = recent_df[recent_df['bullish_ob']].iloc[-15:].index[-1]
                ob_row = df.loc[ob_idx]
                setup['details']['ob_level'] = (ob_row['bullish_ob_top'] + ob_row['bullish_ob_bottom']) / 2
        
        if has_bullish_fvg:
            premium_confidence += 0.15
            setup['details']['bullish_fvg'] = True
            
            # Get the most recent bullish FVG
            if not recent_df[recent_df['bullish_fvg']].iloc[-15:].empty:
                fvg_idx = recent_df[recent_df['bullish_fvg']].iloc[-15:].index[-1]
                fvg_row = df.loc[fvg_idx]
                # Calculate the middle of the FVG as the level
                if pd.notna(fvg_row['bullish_fvg_top']) and pd.notna(fvg_row['bullish_fvg_bottom']):
                    setup['details']['fvg_level'] = (fvg_row['bullish_fvg_top'] + fvg_row['bullish_fvg_bottom']) / 2
                else:
                    # Use current price as fallback
                    setup['details']['fvg_level'] = df['close'].iloc[-1]
        
        if has_bullish_ote:
            premium_confidence += 0.2
            setup['details']['bullish_ote'] = True
            
            # Get the most recent bullish OTE
            if not recent_df[recent_df['bullish_ote']].iloc[-10:].empty:
                ote_idx = recent_df[recent_df['bullish_ote']].iloc[-10:].index[-1]
                ote_row = df.loc[ote_idx]
                setup['entry'] = ote_row['bullish_ote_level']
        
        # Check for liquidity (institutional order flow)
        if recent_df['buy_liquidity'].iloc[-20:].any():
            premium_confidence += 0.1
            setup['details']['buy_liquidity'] = True
            
            # Get the most recent buy liquidity level
            if not recent_df[recent_df['buy_liquidity']].iloc[-20:].empty:
                liq_idx = recent_df[recent_df['buy_liquidity']].iloc[-20:].index[-1]
                liq_row = df.loc[liq_idx]
                setup['details']['liquidity_level'] = liq_row['buy_liquidity_level']
        
        # Bonus for trading in a killzone (high institutional activity)
        if in_killzone:
            # Give highest bonus for Pakistan Super Prime hours (absolute peak liquidity)
            if killzone_name == "Pakistan Super Prime (Peak Liquidity)":
                premium_confidence += 0.3  # Triple bonus for Pakistan Super Prime hours
            # Give higher bonus for Pakistan Prime/Night hours (when the bot is most profitable)
            elif killzone_name in ["Pakistan Prime Hours (London-NY Overlap)", "Pakistan Night Session"]:
                premium_confidence += 0.2  # Double bonus for Pakistan night hours
            else:
                premium_confidence += 0.1
        
        # If we have a premium setup with sufficient confidence (lowered threshold)
        if premium_confidence >= 0.3 and has_bullish_ote:  # Lowered from 0.5 to 0.3
            setup['action'] = 'BUY'
            setup['confidence'] = min(premium_confidence, 1.0)  # Cap at 1.0
            setup['setup_type'] = 'Premium ICT Buy'
            
            # If entry not set from OTE, use the current price
            if setup['entry'] is None:
                setup['entry'] = df['close'].iloc[-1]
            
            # Set stop loss below the order block or recent low
            if has_bullish_ob and not recent_df[recent_df['bullish_ob']].iloc[-15:].empty:
                ob_idx = recent_df[recent_df['bullish_ob']].iloc[-15:].index[-1]
                ob_row = df.loc[ob_idx]
                setup['stop_loss'] = ob_row['bullish_ob_bottom'] - (0.05 * (ob_row['bullish_ob_top'] - ob_row['bullish_ob_bottom']))
            else:
                # Use recent swing low with buffer
                setup['stop_loss'] = recent_df['low'].iloc[-20:].min() * 0.998
            
            # Set multiple take profit levels based on risk:reward ratios
            risk = abs(setup['entry'] - setup['stop_loss'])
            
            # Take profit levels at 1.5R, 2.75R, and 4.75R (from config)
            setup['take_profit_1'] = setup['entry'] + (risk * 1.5)
            setup['take_profit_2'] = setup['entry'] + (risk * 2.75)
            setup['take_profit_3'] = setup['entry'] + (risk * 4.75)
            setup['take_profit'] = setup['take_profit_1']  # Default TP
            
            # Calculate risk:reward ratio
            setup['risk_reward'] = 1.5  # Default to first TP level
    
    # 2. PREMIUM SELL SETUP: Bearish Order Block + Fair Value Gap + Market Structure Shift
    if daily_bias in ['BEARISH', 'NEUTRAL']:
        # Check for bearish market structure shift (MSS)
        has_bearish_mss = recent_df['bearish_mss'].iloc[-15:].any()
        
        # Check for bearish order blocks
        has_bearish_ob = recent_df['bearish_ob'].iloc[-15:].any()
        
        # Check for bearish fair value gaps
        has_bearish_fvg = recent_df['bearish_fvg'].iloc[-15:].any()
        
        # Check for optimal trade entry (OTE)
        has_bearish_ote = recent_df['bearish_ote'].iloc[-10:].any()
        
        # Calculate premium setup confidence
        premium_confidence = 0.0
        
        # Base confidence from daily bias
        if daily_bias == 'BEARISH':
            premium_confidence += 0.3
        elif daily_bias == 'NEUTRAL':
            premium_confidence += 0.1
        
        # Add confidence for each component
        if has_bearish_mss:
            premium_confidence += 0.2
            setup['details']['bearish_mss'] = True
        
        if has_bearish_ob:
            premium_confidence += 0.15
            setup['details']['bearish_ob'] = True
            
            # Get the most recent bearish order block
            if not recent_df[recent_df['bearish_ob']].iloc[-15:].empty:
                ob_idx = recent_df[recent_df['bearish_ob']].iloc[-15:].index[-1]
                ob_row = df.loc[ob_idx]
                setup['details']['ob_level'] = (ob_row['bearish_ob_top'] + ob_row['bearish_ob_bottom']) / 2
        
        if has_bearish_fvg:
            premium_confidence += 0.15
            setup['details']['bearish_fvg'] = True
            
            # Get the most recent bearish FVG
            if not recent_df[recent_df['bearish_fvg']].iloc[-15:].empty:
                fvg_idx = recent_df[recent_df['bearish_fvg']].iloc[-15:].index[-1]
                fvg_row = df.loc[fvg_idx]
                # Calculate the middle of the FVG as the level
                if pd.notna(fvg_row['bearish_fvg_top']) and pd.notna(fvg_row['bearish_fvg_bottom']):
                    setup['details']['fvg_level'] = (fvg_row['bearish_fvg_top'] + fvg_row['bearish_fvg_bottom']) / 2
                else:
                    # Use current price as fallback
                    setup['details']['fvg_level'] = df['close'].iloc[-1]
        
        if has_bearish_ote:
            premium_confidence += 0.2
            setup['details']['bearish_ote'] = True
            
            # Get the most recent bearish OTE
            if not recent_df[recent_df['bearish_ote']].iloc[-10:].empty:
                ote_idx = recent_df[recent_df['bearish_ote']].iloc[-10:].index[-1]
                ote_row = df.loc[ote_idx]
                setup['entry'] = ote_row['bearish_ote_level']
        
        # Check for liquidity (institutional order flow)
        if recent_df['sell_liquidity'].iloc[-20:].any():
            premium_confidence += 0.1
            setup['details']['sell_liquidity'] = True
            
            # Get the most recent sell liquidity level
            if not recent_df[recent_df['sell_liquidity']].iloc[-20:].empty:
                liq_idx = recent_df[recent_df['sell_liquidity']].iloc[-20:].index[-1]
                liq_row = df.loc[liq_idx]
                setup['details']['liquidity_level'] = liq_row['sell_liquidity_level']
        
        # Bonus for trading in a killzone (high institutional activity)
        if in_killzone:
            # Give highest bonus for Pakistan Super Prime hours (absolute peak liquidity)
            if killzone_name == "Pakistan Super Prime (Peak Liquidity)":
                premium_confidence += 0.3  # Triple bonus for Pakistan Super Prime hours
            # Give higher bonus for Pakistan Prime/Night hours (when the bot is most profitable)
            elif killzone_name in ["Pakistan Prime Hours (London-NY Overlap)", "Pakistan Night Session"]:
                premium_confidence += 0.2  # Double bonus for Pakistan night hours
            else:
                premium_confidence += 0.1
        
        # If we have a premium setup with sufficient confidence (lowered threshold)
        if premium_confidence >= 0.3 and has_bearish_ote:  # Lowered from 0.5 to 0.3
            setup['action'] = 'SELL'
            setup['confidence'] = min(premium_confidence, 1.0)  # Cap at 1.0
            setup['setup_type'] = 'Premium ICT Sell'
            
            # If entry not set from OTE, use the current price
            if setup['entry'] is None:
                setup['entry'] = df['close'].iloc[-1]
            
            # Set stop loss above the order block or recent high
            if has_bearish_ob and not recent_df[recent_df['bearish_ob']].iloc[-15:].empty:
                ob_idx = recent_df[recent_df['bearish_ob']].iloc[-15:].index[-1]
                ob_row = df.loc[ob_idx]
                setup['stop_loss'] = ob_row['bearish_ob_top'] + (0.05 * (ob_row['bearish_ob_top'] - ob_row['bearish_ob_bottom']))
            else:
                # Use recent swing high with buffer
                setup['stop_loss'] = recent_df['high'].iloc[-20:].max() * 1.002
            
            # Set multiple take profit levels based on risk:reward ratios
            risk = abs(setup['entry'] - setup['stop_loss'])
            
            # Take profit levels at 1.5R, 2.5R, and 4.0R (from config)
            setup['take_profit_1'] = setup['entry'] - (risk * 1.5)
            setup['take_profit_2'] = setup['entry'] - (risk * 2.5)
            setup['take_profit_3'] = setup['entry'] - (risk * 4.0)
            setup['take_profit'] = setup['take_profit_1']  # Default TP
            
            # Calculate risk:reward ratio
            setup['risk_reward'] = 1.5  # Default to first TP level
    
    return setup
