import pandas as pd
import numpy as np
import math
import logging
from scipy.stats import norm
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

def calculate_call_greeks(
    df: pd.DataFrame, 
    current_price: float, 
    time_to_expiry: float, 
    risk_free_rate: float, 
    historical_vol: float = 0.3
) -> pd.DataFrame:
    """
    Calculate accurate Greeks for call options using Black-Scholes
    
    Args:
        df: DataFrame with call options data
        current_price: Current price of the underlying asset
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (decimal)
        historical_vol: Historical volatility (if IV not available)
        
    Returns:
        DataFrame with added Greeks calculations
    """
    for idx, row in df.iterrows():
        strike_price = row['strike']
        
        # Use implied volatility from yfinance if available, otherwise use historical
        sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']) else historical_vol
        
        # Ensure sigma is positive
        sigma = max(0.001, sigma)
        
        try:
            # Calculate d1 and d2
            d1 = (math.log(current_price/strike_price) + (risk_free_rate + 0.5 * sigma**2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
            d2 = d1 - sigma * math.sqrt(time_to_expiry)
            
            # Calculate delta
            delta = norm.cdf(d1)
            
            # Calculate gamma
            gamma = norm.pdf(d1) / (current_price * sigma * math.sqrt(time_to_expiry))
            
            # Calculate theta (daily)
            theta = (-current_price * norm.pdf(d1) * sigma / (2 * math.sqrt(time_to_expiry)) - 
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
            
            # Calculate vega (for 1% change in volatility)
            vega = current_price * math.sqrt(time_to_expiry) * norm.pdf(d1) * 0.01
            
            # Calculate rho (for 1% change in interest rate)
            rho = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) * 0.01
            
            # Update the dataframe
            df.at[idx, 'delta'] = delta
            df.at[idx, 'gamma'] = gamma
            df.at[idx, 'theta'] = theta
            df.at[idx, 'vega'] = vega
            df.at[idx, 'rho'] = rho
            
        except Exception as e:
            logger.warning(f"Error calculating call greeks for strike {strike_price}: {e}")
            df.at[idx, 'delta'] = 1.0 if current_price > strike_price else 0.0
            df.at[idx, 'gamma'] = 0.0
            df.at[idx, 'theta'] = 0.0
            df.at[idx, 'vega'] = 0.0
            df.at[idx, 'rho'] = 0.0
    
    return df

def calculate_put_greeks(
    df: pd.DataFrame, 
    current_price: float, 
    time_to_expiry: float, 
    risk_free_rate: float, 
    historical_vol: float = 0.3
) -> pd.DataFrame:
    """
    Calculate accurate Greeks for put options using Black-Scholes
    
    Args:
        df: DataFrame with put options data
        current_price: Current price of the underlying asset
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (decimal)
        historical_vol: Historical volatility (if IV not available)
        
    Returns:
        DataFrame with added Greeks calculations
    """
    for idx, row in df.iterrows():
        strike_price = row['strike']
        
        # Use implied volatility from yfinance if available, otherwise use historical
        sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']) else historical_vol
        
        # Ensure sigma is positive
        sigma = max(0.001, sigma)
        
        try:
            # Calculate d1 and d2
            d1 = (math.log(current_price/strike_price) + (risk_free_rate + 0.5 * sigma**2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
            d2 = d1 - sigma * math.sqrt(time_to_expiry)
            
            # Calculate delta
            delta = norm.cdf(d1) - 1
            
            # Calculate gamma (same as call)
            gamma = norm.pdf(d1) / (current_price * sigma * math.sqrt(time_to_expiry))
            
            # Calculate theta (daily)
            theta = (-current_price * norm.pdf(d1) * sigma / (2 * math.sqrt(time_to_expiry)) + 
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
            
            # Calculate vega (for 1% change in volatility)
            vega = current_price * math.sqrt(time_to_expiry) * norm.pdf(d1) * 0.01
            
            # Calculate rho (for 1% change in interest rate)
            rho = -strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) * 0.01
            
            # Update the dataframe
            df.at[idx, 'delta'] = delta
            df.at[idx, 'gamma'] = gamma
            df.at[idx, 'theta'] = theta
            df.at[idx, 'vega'] = vega
            df.at[idx, 'rho'] = rho
            
        except Exception as e:
            logger.warning(f"Error calculating put greeks for strike {strike_price}: {e}")
            df.at[idx, 'delta'] = -1.0 if current_price < strike_price else 0.0
            df.at[idx, 'gamma'] = 0.0
            df.at[idx, 'theta'] = 0.0
            df.at[idx, 'vega'] = 0.0
            df.at[idx, 'rho'] = 0.0
    
    return df

def calculate_historical_volatility(data: pd.DataFrame, days: int = 30) -> float:
    """
    Calculate historical volatility from daily returns
    
    Args:
        data: DataFrame with price history
        days: Number of days to use for calculation
        
    Returns:
        Historical volatility (annualized)
    """
    try:
        if data.empty or len(data) < 5:
            logger.warning("Insufficient data to calculate historical volatility")
            return 0.3  # Default value
        
        # Calculate daily returns
        daily_returns = data['Close'].pct_change().dropna()
        
        if len(daily_returns) < 5:
            return 0.3  # Default value
        
        # Calculate rolling standard deviation (annualized)
        historical_vol = daily_returns.std() * math.sqrt(252)
        
        logger.info(f"Historical volatility: {historical_vol:.4f}")
        return float(historical_vol)
        
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {e}")
        return 0.3  # Default value

def get_risk_free_rate() -> float:
    """
    Get current risk-free rate (Treasury yield)
    
    In a real implementation, we would fetch the current Treasury yield.
    For simplicity, we'll use a fixed reasonable value.
    
    Returns:
        Current risk-free rate as a decimal
    """
    # 5% is a reasonable value for mid-2023 to 2024
    return 0.05

def calculate_iv_surface(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate implied volatility surface summary
    
    Args:
        calls_df: DataFrame with call options data
        puts_df: DataFrame with put options data
        
    Returns:
        Dictionary with IV surface data
    """
    try:
        iv_surface = {
            'call_iv_by_strike': {},
            'put_iv_by_strike': {},
            'skew_by_strike': {}
        }
        
        # Process call IVs
        if not calls_df.empty and 'impliedVolatility' in calls_df.columns:
            for _, row in calls_df.iterrows():
                strike = row['strike']
                iv = row['impliedVolatility']
                if not pd.isna(iv):
                    iv_surface['call_iv_by_strike'][strike] = float(iv)
        
        # Process put IVs
        if not puts_df.empty and 'impliedVolatility' in puts_df.columns:
            for _, row in puts_df.iterrows():
                strike = row['strike']
                iv = row['impliedVolatility']
                if not pd.isna(iv):
                    iv_surface['put_iv_by_strike'][strike] = float(iv)
        
        # Calculate skew where we have both call and put IVs
        common_strikes = set(iv_surface['call_iv_by_strike'].keys()) & set(iv_surface['put_iv_by_strike'].keys())
        for strike in common_strikes:
            iv_surface['skew_by_strike'][strike] = iv_surface['put_iv_by_strike'][strike] - iv_surface['call_iv_by_strike'][strike]
        
        # Find the smile shape
        if iv_surface['put_iv_by_strike'] and iv_surface['call_iv_by_strike']:
            sorted_call_strikes = sorted(iv_surface['call_iv_by_strike'].keys())
            sorted_put_strikes = sorted(iv_surface['put_iv_by_strike'].keys())
            
            if len(sorted_call_strikes) >= 3:
                # Check for volatility smile in calls
                lower_call_iv = iv_surface['call_iv_by_strike'][sorted_call_strikes[0]]
                middle_strikes = sorted_call_strikes[len(sorted_call_strikes)//2]
                middle_call_iv = iv_surface['call_iv_by_strike'][middle_strikes]
                upper_call_iv = iv_surface['call_iv_by_strike'][sorted_call_strikes[-1]]
                
                iv_surface['call_smile'] = lower_call_iv > middle_call_iv and upper_call_iv > middle_call_iv
            
            if len(sorted_put_strikes) >= 3:
                # Check for volatility smile in puts
                lower_put_iv = iv_surface['put_iv_by_strike'][sorted_put_strikes[0]]
                middle_strikes = sorted_put_strikes[len(sorted_put_strikes)//2]
                middle_put_iv = iv_surface['put_iv_by_strike'][middle_strikes]
                upper_put_iv = iv_surface['put_iv_by_strike'][sorted_put_strikes[-1]]
                
                iv_surface['put_smile'] = lower_put_iv > middle_put_iv and upper_put_iv > middle_put_iv
        
        return iv_surface
        
    except Exception as e:
        logger.error(f"Error calculating IV surface: {e}")
        return {
            'call_iv_by_strike': {},
            'put_iv_by_strike': {},
            'skew_by_strike': {}
        } 