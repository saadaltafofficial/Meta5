#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of ICT (Inner Circle Trader) 2024 model for forex trading"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ICTModel:
    """Class implementing the ICT (Inner Circle Trader) 2024 model"""
    
    def __init__(self):
        """Initialize the ICT model"""
        # ICT model parameters
        self.mtf_timeframes = ['1h', '4h', 'D']  # Multi-timeframe analysis
        self.fair_value_gap_threshold = 0.0015  # 0.15% threshold for FVG
        self.liquidity_threshold = 0.002  # 0.2% threshold for liquidity levels
        self.optimal_trade_entry_window = 4  # Hours for OTE
        self.breaker_block_lookback = 10  # Candles to look back for breaker blocks
        self.premium_discount_threshold = 0.003  # 0.3% for premium/discount zones
    
    def analyze(self, data, pair):
        """Analyze forex data using the ICT model
        
        Args:
            data (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            pair (str): Currency pair being analyzed
            
        Returns:
            dict: ICT analysis results including signals and key levels
        """
        logger.info(f"Performing ICT model analysis for {pair}")
        
        # Ensure we have enough data
        if len(data) < 100:
            logger.warning(f"Not enough data for ICT analysis of {pair}")
            return {'valid': False, 'reason': 'Insufficient data'}
        
        # Store results
        results = {
            'valid': True,
            'pair': pair,
            'timestamp': datetime.now(),
            'key_levels': {},
            'signals': {}
        }
        
        try:
            # 1. Identify market structure
            market_structure = self._identify_market_structure(data)
            results['market_structure'] = market_structure
            
            # 2. Find fair value gaps (FVG)
            fvgs = self._find_fair_value_gaps(data)
            results['key_levels']['fair_value_gaps'] = fvgs
            
            # 3. Identify liquidity pools
            liquidity_pools = self._identify_liquidity_pools(data)
            results['key_levels']['liquidity_pools'] = liquidity_pools
            
            # 4. Find order blocks
            order_blocks = self._find_order_blocks(data, market_structure)
            results['key_levels']['order_blocks'] = order_blocks
            
            # 5. Identify breaker blocks
            breaker_blocks = self._find_breaker_blocks(data, market_structure)
            results['key_levels']['breaker_blocks'] = breaker_blocks
            
            # 6. Determine optimal trade entry (OTE)
            ote = self._find_optimal_trade_entry(data, fvgs, order_blocks)
            results['key_levels']['optimal_trade_entry'] = ote
            
            # 7. Identify premium and discount zones
            premium_discount = self._identify_premium_discount_zones(data)
            results['key_levels']['premium_discount'] = premium_discount
            
            # 8. Generate trading signals based on ICT concepts
            signals = self._generate_signals(
                data, 
                market_structure, 
                fvgs, 
                liquidity_pools, 
                order_blocks, 
                breaker_blocks, 
                ote, 
                premium_discount
            )
            results['signals'] = signals
            
            # 9. Calculate confidence level
            confidence = self._calculate_confidence(signals)
            results['confidence'] = confidence
            
            logger.info(f"ICT analysis completed for {pair} with confidence {confidence:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ICT analysis for {pair}: {e}")
            return {'valid': False, 'reason': f'Analysis error: {str(e)}'}
    
    def _identify_market_structure(self, data):
        """Identify market structure (bullish, bearish, or ranging)
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Market structure information
        """
        # Get recent price action (last 20 candles)
        recent_data = data.tail(20).copy()
        
        # Calculate higher highs (HH), higher lows (HL), lower highs (LH), lower lows (LL)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Count trend components
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0
        
        for i in range(2, len(recent_data)):
            # Higher high
            if highs[i] > highs[i-1] and highs[i-1] > highs[i-2]:
                hh_count += 1
            # Higher low
            if lows[i] > lows[i-1] and lows[i-1] > lows[i-2]:
                hl_count += 1
            # Lower high
            if highs[i] < highs[i-1] and highs[i-1] < highs[i-2]:
                lh_count += 1
            # Lower low
            if lows[i] < lows[i-1] and lows[i-1] < lows[i-2]:
                ll_count += 1
        
        # Determine overall structure
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count
        
        if bullish_score > bearish_score * 1.5:
            structure = 'bullish'
        elif bearish_score > bullish_score * 1.5:
            structure = 'bearish'
        else:
            structure = 'ranging'
        
        # Check for change of character (CHoCH)
        choch = False
        if structure == 'bullish' and lh_count > 0:
            choch = True
        elif structure == 'bearish' and hl_count > 0:
            choch = True
        
        return {
            'structure': structure,
            'change_of_character': choch,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'hh_count': hh_count,
            'hl_count': hl_count,
            'lh_count': lh_count,
            'll_count': ll_count
        }
    
    def _find_fair_value_gaps(self, data):
        """Find fair value gaps (FVG) in the data
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            list: Fair value gaps identified
        """
        fvgs = []
        
        # Need at least 3 candles to identify FVGs
        if len(data) < 3:
            return fvgs
        
        # Look for bullish and bearish FVGs
        for i in range(2, len(data)):
            # Bullish FVG: Current candle's low > Previous candle's high
            if data.iloc[i]['low'] > data.iloc[i-2]['high']:
                gap_size = (data.iloc[i]['low'] - data.iloc[i-2]['high']) / data.iloc[i-2]['high']
                if gap_size >= self.fair_value_gap_threshold:
                    fvgs.append({
                        'type': 'bullish',
                        'position': i,
                        'level': (data.iloc[i]['low'] + data.iloc[i-2]['high']) / 2,
                        'size': gap_size,
                        'timestamp': data.index[i]
                    })
            
            # Bearish FVG: Current candle's high < Previous candle's low
            if data.iloc[i]['high'] < data.iloc[i-2]['low']:
                gap_size = (data.iloc[i-2]['low'] - data.iloc[i]['high']) / data.iloc[i]['high']
                if gap_size >= self.fair_value_gap_threshold:
                    fvgs.append({
                        'type': 'bearish',
                        'position': i,
                        'level': (data.iloc[i]['high'] + data.iloc[i-2]['low']) / 2,
                        'size': gap_size,
                        'timestamp': data.index[i]
                    })
        
        return fvgs
    
    def _identify_liquidity_pools(self, data):
        """Identify liquidity pools (swing highs/lows with clustered stops)
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Identified liquidity pools
        """
        # Find swing highs and lows (potential liquidity areas)
        swing_points = []
        
        # Need at least 5 candles to identify swing points
        if len(data) < 5:
            return {'highs': [], 'lows': []}
        
        # Find swing highs and lows using a 5-candle window
        for i in range(2, len(data) - 2):
            # Swing high
            if (data.iloc[i]['high'] > data.iloc[i-1]['high'] and 
                data.iloc[i]['high'] > data.iloc[i-2]['high'] and 
                data.iloc[i]['high'] > data.iloc[i+1]['high'] and 
                data.iloc[i]['high'] > data.iloc[i+2]['high']):
                swing_points.append({
                    'type': 'high',
                    'position': i,
                    'level': data.iloc[i]['high'],
                    'timestamp': data.index[i]
                })
            
            # Swing low
            if (data.iloc[i]['low'] < data.iloc[i-1]['low'] and 
                data.iloc[i]['low'] < data.iloc[i-2]['low'] and 
                data.iloc[i]['low'] < data.iloc[i+1]['low'] and 
                data.iloc[i]['low'] < data.iloc[i+2]['low']):
                swing_points.append({
                    'type': 'low',
                    'position': i,
                    'level': data.iloc[i]['low'],
                    'timestamp': data.index[i]
                })
        
        # Cluster swing points to identify liquidity pools
        highs = []
        lows = []
        
        for point in swing_points:
            if point['type'] == 'high':
                highs.append(point)
            else:
                lows.append(point)
        
        return {'highs': highs, 'lows': lows}
    
    def _find_order_blocks(self, data, market_structure):
        """Find order blocks (last candle before a strong move)
        
        Args:
            data (pd.DataFrame): OHLCV data
            market_structure (dict): Market structure information
            
        Returns:
            list: Identified order blocks
        """
        order_blocks = []
        
        # Need at least 10 candles to identify order blocks
        if len(data) < 10:
            return order_blocks
        
        # Look for bullish and bearish order blocks
        for i in range(3, len(data) - 3):
            # Bullish order block: Last red candle before a strong bullish move
            if (data.iloc[i]['close'] < data.iloc[i]['open'] and  # Red candle
                data.iloc[i+1]['close'] > data.iloc[i+1]['open'] and  # Green candle
                data.iloc[i+2]['close'] > data.iloc[i+2]['open'] and  # Green candle
                data.iloc[i+3]['close'] > data.iloc[i+3]['open']):  # Green candle
                
                # Calculate move strength
                move_strength = (data.iloc[i+3]['close'] - data.iloc[i]['low']) / data.iloc[i]['low']
                
                if move_strength >= 0.005:  # 0.5% move or greater
                    order_blocks.append({
                        'type': 'bullish',
                        'position': i,
                        'top': data.iloc[i]['open'],
                        'bottom': data.iloc[i]['close'],
                        'strength': move_strength,
                        'timestamp': data.index[i]
                    })
            
            # Bearish order block: Last green candle before a strong bearish move
            if (data.iloc[i]['close'] > data.iloc[i]['open'] and  # Green candle
                data.iloc[i+1]['close'] < data.iloc[i+1]['open'] and  # Red candle
                data.iloc[i+2]['close'] < data.iloc[i+2]['open'] and  # Red candle
                data.iloc[i+3]['close'] < data.iloc[i+3]['open']):  # Red candle
                
                # Calculate move strength
                move_strength = (data.iloc[i]['high'] - data.iloc[i+3]['close']) / data.iloc[i+3]['close']
                
                if move_strength >= 0.005:  # 0.5% move or greater
                    order_blocks.append({
                        'type': 'bearish',
                        'position': i,
                        'top': data.iloc[i]['close'],
                        'bottom': data.iloc[i]['open'],
                        'strength': move_strength,
                        'timestamp': data.index[i]
                    })
        
        return order_blocks
    
    def _find_breaker_blocks(self, data, market_structure):
        """Find breaker blocks (order blocks that have been broken)
        
        Args:
            data (pd.DataFrame): OHLCV data
            market_structure (dict): Market structure information
            
        Returns:
            list: Identified breaker blocks
        """
        breaker_blocks = []
        
        # Need sufficient data
        if len(data) < self.breaker_block_lookback + 5:
            return breaker_blocks
        
        # Get recent order blocks
        recent_data = data.tail(self.breaker_block_lookback + 5).copy()
        order_blocks = self._find_order_blocks(recent_data, market_structure)
        
        # Check if price has returned to and broken through these order blocks
        current_price = data.iloc[-1]['close']
        
        for block in order_blocks:
            if block['type'] == 'bullish':
                # A bullish order block becomes a breaker when price breaks below it and then back above
                if any(data.iloc[block['position']:]['low'] < block['bottom']) and current_price > block['top']:
                    breaker_blocks.append({
                        'type': 'bullish_breaker',
                        'original_block': block,
                        'broken_at': current_price,
                        'timestamp': data.index[-1]
                    })
            else:  # bearish
                # A bearish order block becomes a breaker when price breaks above it and then back below
                if any(data.iloc[block['position']:]['high'] > block['top']) and current_price < block['bottom']:
                    breaker_blocks.append({
                        'type': 'bearish_breaker',
                        'original_block': block,
                        'broken_at': current_price,
                        'timestamp': data.index[-1]
                    })
        
        return breaker_blocks
    
    def _find_optimal_trade_entry(self, data, fvgs, order_blocks):
        """Find optimal trade entry (OTE) zones
        
        Args:
            data (pd.DataFrame): OHLCV data
            fvgs (list): Fair value gaps
            order_blocks (list): Order blocks
            
        Returns:
            list: Optimal trade entry zones
        """
        ote_zones = []
        
        # Current price
        current_price = data.iloc[-1]['close']
        
        # Check FVGs for OTE opportunities
        for fvg in fvgs:
            # Only consider recent FVGs (within last 20 candles)
            if fvg['position'] >= len(data) - 20:
                # Bullish FVG: OTE when price pulls back to the FVG
                if fvg['type'] == 'bullish' and current_price <= fvg['level'] * 1.01:
                    ote_zones.append({
                        'type': 'bullish_fvg_ote',
                        'level': fvg['level'],
                        'source': 'fvg',
                        'timestamp': data.index[-1]
                    })
                # Bearish FVG: OTE when price rallies to the FVG
                elif fvg['type'] == 'bearish' and current_price >= fvg['level'] * 0.99:
                    ote_zones.append({
                        'type': 'bearish_fvg_ote',
                        'level': fvg['level'],
                        'source': 'fvg',
                        'timestamp': data.index[-1]
                    })
        
        # Check order blocks for OTE opportunities
        for block in order_blocks:
            # Only consider recent order blocks (within last 20 candles)
            if block['position'] >= len(data) - 20:
                # Bullish order block: OTE when price pulls back to the block
                if block['type'] == 'bullish' and current_price >= block['bottom'] and current_price <= block['top']:
                    ote_zones.append({
                        'type': 'bullish_ob_ote',
                        'top': block['top'],
                        'bottom': block['bottom'],
                        'source': 'order_block',
                        'timestamp': data.index[-1]
                    })
                # Bearish order block: OTE when price rallies to the block
                elif block['type'] == 'bearish' and current_price >= block['bottom'] and current_price <= block['top']:
                    ote_zones.append({
                        'type': 'bearish_ob_ote',
                        'top': block['top'],
                        'bottom': block['bottom'],
                        'source': 'order_block',
                        'timestamp': data.index[-1]
                    })
        
        return ote_zones
    
    def _identify_premium_discount_zones(self, data):
        """Identify premium and discount zones
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Premium and discount zones
        """
        # Calculate daily average range
        if len(data) < 20:
            return {'premium': None, 'discount': None, 'fair_value': None}
        
        # Use recent data to calculate average daily range
        recent_data = data.tail(20).copy()
        daily_ranges = (recent_data['high'] - recent_data['low']) / recent_data['low']
        avg_daily_range = daily_ranges.mean()
        
        # Current price
        current_price = data.iloc[-1]['close']
        
        # Calculate 20-day moving average as fair value
        fair_value = data['close'].tail(20).mean()
        
        # Define premium and discount zones
        premium_threshold = fair_value * (1 + self.premium_discount_threshold)
        discount_threshold = fair_value * (1 - self.premium_discount_threshold)
        
        # Determine if current price is in premium, discount, or fair value zone
        if current_price > premium_threshold:
            zone = 'premium'
        elif current_price < discount_threshold:
            zone = 'discount'
        else:
            zone = 'fair_value'
        
        return {
            'premium': premium_threshold,
            'discount': discount_threshold,
            'fair_value': fair_value,
            'current_zone': zone
        }
    
    def _generate_signals(self, data, market_structure, fvgs, liquidity_pools, 
                         order_blocks, breaker_blocks, ote, premium_discount):
        """Generate trading signals based on ICT concepts
        
        Args:
            data (pd.DataFrame): OHLCV data
            market_structure (dict): Market structure information
            fvgs (list): Fair value gaps
            liquidity_pools (dict): Liquidity pools
            order_blocks (list): Order blocks
            breaker_blocks (list): Breaker blocks
            ote (list): Optimal trade entry zones
            premium_discount (dict): Premium and discount zones
            
        Returns:
            dict: Trading signals
        """
        # Current price
        current_price = data.iloc[-1]['close']
        
        # Default signal
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': 'No clear ICT setup',
            'price': current_price,
            'timestamp': datetime.now(),
            'ict_factors': []
        }
        
        # List to track ICT factors supporting the signal
        ict_factors = []
        
        # 1. Check for strong market structure
        if market_structure['structure'] == 'bullish' and market_structure['bullish_score'] >= 3:
            ict_factors.append({
                'factor': 'bullish_structure',
                'weight': 0.2,
                'description': f"Strong bullish market structure (score: {market_structure['bullish_score']})"
            })
        elif market_structure['structure'] == 'bearish' and market_structure['bearish_score'] >= 3:
            ict_factors.append({
                'factor': 'bearish_structure',
                'weight': 0.2,
                'description': f"Strong bearish market structure (score: {market_structure['bearish_score']})"
            })
        
        # 2. Check for OTE opportunities
        for zone in ote:
            if zone['type'].startswith('bullish'):
                ict_factors.append({
                    'factor': 'bullish_ote',
                    'weight': 0.25,
                    'description': f"Bullish optimal trade entry from {zone['source']}"
                })
            elif zone['type'].startswith('bearish'):
                ict_factors.append({
                    'factor': 'bearish_ote',
                    'weight': 0.25,
                    'description': f"Bearish optimal trade entry from {zone['source']}"
                })
        
        # 3. Check for breaker blocks (high-probability setups)
        for block in breaker_blocks:
            if block['type'] == 'bullish_breaker':
                ict_factors.append({
                    'factor': 'bullish_breaker',
                    'weight': 0.3,
                    'description': "Bullish breaker block (high-probability setup)"
                })
            elif block['type'] == 'bearish_breaker':
                ict_factors.append({
                    'factor': 'bearish_breaker',
                    'weight': 0.3,
                    'description': "Bearish breaker block (high-probability setup)"
                })
        
        # 4. Check premium/discount zones for value entries
        if premium_discount['current_zone'] == 'discount' and any(f['factor'].startswith('bullish') for f in ict_factors):
            ict_factors.append({
                'factor': 'value_buy',
                'weight': 0.15,
                'description': "Price in discount zone (value buying opportunity)"
            })
        elif premium_discount['current_zone'] == 'premium' and any(f['factor'].startswith('bearish') for f in ict_factors):
            ict_factors.append({
                'factor': 'value_sell',
                'weight': 0.15,
                'description': "Price in premium zone (value selling opportunity)"
            })
        
        # 5. Check for liquidity grabs
        recent_highs = [h for h in liquidity_pools['highs'] if h['position'] >= len(data) - 10]
        recent_lows = [l for l in liquidity_pools['lows'] if l['position'] >= len(data) - 10]
        
        if recent_highs and current_price > max(h['level'] for h in recent_highs):
            ict_factors.append({
                'factor': 'liquidity_sweep_high',
                'weight': 0.1,
                'description': "Recent liquidity sweep above swing highs"
            })
        
        if recent_lows and current_price < min(l['level'] for l in recent_lows):
            ict_factors.append({
                'factor': 'liquidity_sweep_low',
                'weight': 0.1,
                'description': "Recent liquidity sweep below swing lows"
            })
        
        # Generate the final signal based on accumulated factors
        bullish_weight = sum(f['weight'] for f in ict_factors if f['factor'].startswith('bullish') or f['factor'] == 'value_buy')
        bearish_weight = sum(f['weight'] for f in ict_factors if f['factor'].startswith('bearish') or f['factor'] == 'value_sell')
        
        # Determine action and confidence
        if bullish_weight > 0.4 and bullish_weight > bearish_weight * 1.5:
            if bullish_weight > 0.6:
                signal['action'] = 'BUY'
            else:
                signal['action'] = 'WEAK BUY'
            signal['confidence'] = bullish_weight
            signal['reason'] = "ICT bullish setup"
            signal['ict_factors'] = [f for f in ict_factors if f['factor'].startswith('bullish') or f['factor'] == 'value_buy']
        elif bearish_weight > 0.4 and bearish_weight > bullish_weight * 1.5:
            if bearish_weight > 0.6:
                signal['action'] = 'SELL'
            else:
                signal['action'] = 'WEAK SELL'
            signal['confidence'] = bearish_weight
            signal['reason'] = "ICT bearish setup"
            signal['ict_factors'] = [f for f in ict_factors if f['factor'].startswith('bearish') or f['factor'] == 'value_sell']
        else:
            # Not enough conviction for a trade
            signal['ict_factors'] = ict_factors
        
        return signal
    
    def _calculate_confidence(self, signals):
        """Calculate confidence level for the signal
        
        Args:
            signals (dict): Trading signals
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        if 'confidence' in signals:
            return signals['confidence']
        return 0.0
