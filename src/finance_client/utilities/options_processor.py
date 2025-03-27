import logging
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm

logger = logging.getLogger(__name__)

def calculate_option_greeks(S, K, t, r, sigma, option_type='call'):
    """
    Calculate full set of option greeks using Black-Scholes model
    
    Args:
        S: Current stock price
        K: Option strike price
        t: Time to expiration in years
        r: Risk-free interest rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with calculated greeks
    """
    try:
        # Check for invalid inputs
        if sigma <= 0 or t <= 0 or K <= 0 or S <= 0:
            return {
                'delta': 1.0 if S > K and option_type == 'call' else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
            
        # Calculate d1 and d2
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        
        # Calculate greeks
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(t)) - 
                     r * K * math.exp(-r * t) * norm.cdf(d2)) / 365.0
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(t)) + 
                     r * K * math.exp(-r * t) * norm.cdf(-d2)) / 365.0
        
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(t))
        vega = S * math.sqrt(t) * norm.pdf(d1) * 0.01  # For 1% change in IV
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega)
        }
        
    except Exception as e:
        logger.error(f"Error calculating option greeks: {e}")
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'error': str(e)
        }

def calculate_iv_skew(calls_df, puts_df, current_price):
    """
    Calculate implied volatility skew metrics
    
    Args:
        calls_df: DataFrame with call options data
        puts_df: DataFrame with put options data
        current_price: Current stock price
        
    Returns:
        Dictionary with IV skew metrics
    """
    try:
        # Filter for options near the money (within 5% of current price)
        atm_calls = calls_df[abs((calls_df['strike'] - current_price) / current_price) < 0.05]
        atm_puts = puts_df[abs((puts_df['strike'] - current_price) / current_price) < 0.05]
        
        # Filter for OTM options (calls above, puts below current price)
        otm_calls = calls_df[calls_df['strike'] > current_price * 1.05]
        otm_puts = puts_df[puts_df['strike'] < current_price * 0.95]
        
        # Calculate average IVs for each group
        atm_call_iv = atm_calls['impliedVolatility'].mean() if len(atm_calls) > 0 else None
        atm_put_iv = atm_puts['impliedVolatility'].mean() if len(atm_puts) > 0 else None
        
        otm_call_iv = otm_calls['impliedVolatility'].mean() if len(otm_calls) > 0 else None
        otm_put_iv = otm_puts['impliedVolatility'].mean() if len(otm_puts) > 0 else None
        
        # Calculate various skew metrics
        atm_skew = atm_put_iv - atm_call_iv if (atm_put_iv is not None and atm_call_iv is not None) else None
        otm_skew = otm_put_iv - otm_call_iv if (otm_put_iv is not None and otm_call_iv is not None) else None
        
        # Call and put wing skew (how IV changes as strikes move OTM)
        call_wing_skew = otm_call_iv - atm_call_iv if (otm_call_iv is not None and atm_call_iv is not None) else None
        put_wing_skew = otm_put_iv - atm_put_iv if (otm_put_iv is not None and atm_put_iv is not None) else None
        
        # Skew slope (difference between call wing and put wing)
        skew_slope = call_wing_skew - put_wing_skew if (call_wing_skew is not None and put_wing_skew is not None) else None
        
        return {
            'atm_call_iv': float(atm_call_iv) if atm_call_iv is not None else None,
            'atm_put_iv': float(atm_put_iv) if atm_put_iv is not None else None,
            'otm_call_iv': float(otm_call_iv) if otm_call_iv is not None else None,
            'otm_put_iv': float(otm_put_iv) if otm_put_iv is not None else None,
            'atm_skew': float(atm_skew) if atm_skew is not None else None,
            'otm_skew': float(otm_skew) if otm_skew is not None else None,
            'call_wing_skew': float(call_wing_skew) if call_wing_skew is not None else None,
            'put_wing_skew': float(put_wing_skew) if put_wing_skew is not None else None,
            'skew_slope': float(skew_slope) if skew_slope is not None else None
        }
    except Exception as e:
        logger.error(f"Error calculating IV skew: {e}")
        return {
            'error': str(e),
            'atm_skew': None
        }

def identify_spread_opportunities(calls_df, puts_df, current_price, min_credit=0.1, max_width=10):
    """
    Identify potential credit spread opportunities
    
    Args:
        calls_df: DataFrame with call options data
        puts_df: DataFrame with put options data
        current_price: Current stock price
        min_credit: Minimum credit required (percentage of spread width)
        max_width: Maximum width between short and long strikes (percentage)
        
    Returns:
        Dictionary with bull put and bear call spread opportunities
    """
    try:
        # Calculate max strike width in dollars
        max_width_dollars = current_price * max_width / 100
        
        # Initialize lists for spreads
        bull_put_spreads = []
        bear_call_spreads = []
        
        # Process bull put spreads (for bullish outlook)
        # Sort puts by strike descending (higher to lower)
        sorted_puts = puts_df.sort_values('strike', ascending=False)
        
        for i, put in sorted_puts.iterrows():
            short_strike = put['strike']
            
            # Skip if strike is too far ITM or OTM
            if short_strike < current_price * 0.8 or short_strike > current_price * 1.05:
                continue
            
            # Calculate estimated delta if not available
            delta_estimation = put.get('delta', (current_price - short_strike) / current_price)
            
            # Look for a lower strike to buy for protection
            potential_long_puts = sorted_puts[sorted_puts['strike'] < short_strike]
            
            for j, long_put in potential_long_puts.iterrows():
                long_strike = long_put['strike']
                
                # Check if width is within limits
                width = short_strike - long_strike
                if width > max_width_dollars:
                    continue
                
                # Calculate credit and risk metrics
                short_price = put['lastPrice']
                long_price = long_put['lastPrice']
                credit = short_price - long_price
                max_risk = short_strike - long_strike - credit
                
                # Only include if credit is meaningful
                if credit < min_credit:
                    continue
                    
                # Calculate return metrics
                if max_risk > 0:
                    return_on_risk = credit / max_risk
                    # Add to list if it meets criteria
                    bull_put_spreads.append({
                        'short_strike': float(short_strike),
                        'long_strike': float(long_strike),
                        'width': float(width),
                        'credit': float(credit),
                        'max_risk': float(max_risk),
                        'return_on_risk': float(return_on_risk),
                        'short_delta': float(delta_estimation)
                    })
                    # Only need one long strike per short strike
                    break
        
        # Process bear call spreads (for bearish outlook)
        # Sort calls by strike ascending (lower to higher)
        sorted_calls = calls_df.sort_values('strike', ascending=True)
        
        for i, call in sorted_calls.iterrows():
            short_strike = call['strike']
            
            # Skip if strike is too far ITM or OTM
            if short_strike < current_price * 0.95 or short_strike > current_price * 1.2:
                continue
            
            # Calculate estimated delta if not available
            delta_estimation = call.get('delta', (short_strike - current_price) / current_price)
            
            # Look for a higher strike to buy for protection
            potential_long_calls = sorted_calls[sorted_calls['strike'] > short_strike]
            
            for j, long_call in potential_long_calls.iterrows():
                long_strike = long_call['strike']
                
                # Check if width is within limits
                width = long_strike - short_strike
                if width > max_width_dollars:
                    continue
                
                # Calculate credit and risk metrics
                short_price = call['lastPrice']
                long_price = long_call['lastPrice']
                credit = short_price - long_price
                max_risk = long_strike - short_strike - credit
                
                # Only include if credit is meaningful
                if credit < min_credit:
                    continue
                    
                # Calculate return metrics
                if max_risk > 0:
                    return_on_risk = credit / max_risk
                    # Add to list if it meets criteria
                    bear_call_spreads.append({
                        'short_strike': float(short_strike),
                        'long_strike': float(long_strike),
                        'width': float(width),
                        'credit': float(credit),
                        'max_risk': float(max_risk),
                        'return_on_risk': float(return_on_risk),
                        'short_delta': float(delta_estimation)
                    })
                    # Only need one long strike per short strike
                    break
        
        # Sort spreads by return on risk
        bull_put_spreads.sort(key=lambda x: x['return_on_risk'], reverse=True)
        bear_call_spreads.sort(key=lambda x: x['return_on_risk'], reverse=True)
        
        return {
            'bull_put_spreads': bull_put_spreads,
            'bear_call_spreads': bear_call_spreads
        }
    except Exception as e:
        logger.error(f"Error identifying spread opportunities: {e}")
        return {
            'error': str(e),
            'bull_put_spreads': [],
            'bear_call_spreads': []
        }

def enrich_options_chain(options_chain, current_price, risk_free_rate=0.05):
    """
    Enrich options chain data with accurate greeks and additional metrics
    
    Args:
        options_chain: Dictionary with calls and puts DataFrames
        current_price: Current stock price
        risk_free_rate: Risk-free interest rate to use in calculations
        
    Returns:
        Enriched options chain with added metrics
    """
    try:
        if not options_chain or 'calls' not in options_chain or 'puts' not in options_chain:
            logger.error("Invalid options chain data")
            return options_chain
            
        # Get calls and puts DataFrames
        calls_df = options_chain['calls'].copy()
        puts_df = options_chain['puts'].copy()
        
        # Get expiration date and calculate time to expiry
        expiration_date = options_chain.get('expiration_date')
        
        # Calculate days to expiration
        if expiration_date:
            try:
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                today = datetime.now().date()
                days_to_expiration = (exp_date - today).days
                time_to_expiry = days_to_expiration / 365.0  # In years
            except:
                # Default to 30 days if parsing fails
                time_to_expiry = 30 / 365.0
                days_to_expiration = 30
        else:
            time_to_expiry = 30 / 365.0
            days_to_expiration = 30
        
        # Calculate greeks for calls
        for idx, option in calls_df.iterrows():
            strike = option['strike']
            
            # Use implied volatility from data or use a default
            iv = option.get('impliedVolatility', 0.3)
            if pd.isna(iv) or iv <= 0:
                iv = 0.3
                
            # Calculate all greeks
            greeks = calculate_option_greeks(
                current_price, strike, time_to_expiry, risk_free_rate, iv, 'call'
            )
            
            # Update the DataFrame
            for greek, value in greeks.items():
                calls_df.at[idx, greek] = value
                
            # Add distance from current price
            calls_df.at[idx, 'distance_pct'] = ((strike - current_price) / current_price) * 100
            calls_df.at[idx, 'is_atm'] = abs(strike - current_price) < (current_price * 0.05)
            calls_df.at[idx, 'is_itm'] = strike < current_price
            calls_df.at[idx, 'is_otm'] = strike > current_price
        
        # Calculate greeks for puts
        for idx, option in puts_df.iterrows():
            strike = option['strike']
            
            # Use implied volatility from data or use a default
            iv = option.get('impliedVolatility', 0.3)
            if pd.isna(iv) or iv <= 0:
                iv = 0.3
                
            # Calculate all greeks
            greeks = calculate_option_greeks(
                current_price, strike, time_to_expiry, risk_free_rate, iv, 'put'
            )
            
            # Update the DataFrame
            for greek, value in greeks.items():
                puts_df.at[idx, greek] = value
                
            # Add distance from current price
            puts_df.at[idx, 'distance_pct'] = ((strike - current_price) / current_price) * 100
            puts_df.at[idx, 'is_atm'] = abs(strike - current_price) < (current_price * 0.05)
            puts_df.at[idx, 'is_itm'] = strike > current_price
            puts_df.at[idx, 'is_otm'] = strike < current_price
        
        # Calculate IV skew
        iv_skew = calculate_iv_skew(calls_df, puts_df, current_price)
        
        # Find potential credit spread opportunities
        spread_opportunities = identify_spread_opportunities(calls_df, puts_df, current_price)
        
        # Return enriched options chain
        return {
            'symbol': options_chain.get('symbol', 'Unknown'),
            'expiration_date': expiration_date,
            'days_to_expiration': days_to_expiration,
            'current_price': current_price,
            'calls': calls_df,
            'puts': puts_df,
            'iv_skew': iv_skew,
            'bull_put_spreads': spread_opportunities.get('bull_put_spreads', []),
            'bear_call_spreads': spread_opportunities.get('bear_call_spreads', [])
        }
    except Exception as e:
        logger.error(f"Error enriching options chain: {e}")
        return options_chain 