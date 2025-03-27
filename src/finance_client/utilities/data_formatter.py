import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def format_stock_data_for_analysis(data, ticker, date_str):
    """
    Format stock data into a structure suitable for analysis
    
    Args:
        data: DataFrame with OHLCV data
        ticker: Stock symbol
        date_str: Date string for the analysis
        
    Returns:
        Dictionary with formatted data for analysis
    """
    logger.info(f"Formatting stock data for {ticker} as of {date_str}")
    
    try:
        # Skip if data is empty
        if data is None or len(data) == 0:
            logger.error(f"No historical data available for {ticker} on {date_str}")
            return {
                "ticker": ticker,
                "date": date_str,
                "error": "No data available"
            }
        
        # Handle multi-index columns (from yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            # For standard yfinance format where columns have (Price, Ticker) structure
            if len(data.columns.levels) == 2:
                # Try to get just the data for our ticker
                try:
                    data = data.xs(ticker, level=1, axis=1)
                except (KeyError, ValueError):
                    # If that fails, just take the first ticker in the data
                    first_ticker = data.columns.get_level_values(1)[0]
                    data = data.xs(first_ticker, level=1, axis=1)
                    logger.warning(f"Could not find {ticker} in data, using {first_ticker} instead")
            else:
                # If the structure is different than expected, just flatten the column index
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Extract key metrics with safe extraction logic
        current_price = safe_extract(data, 'Close')
        open_price = safe_extract(data, 'Open')
        high_price = safe_extract(data, 'High')
        low_price = safe_extract(data, 'Low')
        volume = safe_extract(data, 'Volume')
        
        # Get other indicators if they exist
        atr = safe_extract(data, 'atr', None)
        atr_percent = safe_extract(data, 'atr_percent', None)
        ema9 = safe_extract(data, 'ema9', None)
        ema21 = safe_extract(data, 'ema21', None)
        ema50 = safe_extract(data, 'ema50', None)
        sma50 = safe_extract(data, 'sma50', None)
        sma200 = safe_extract(data, 'sma200', None)
        rsi = safe_extract(data, 'rsi', None)
        macd = safe_extract(data, 'macd', None)
        macd_signal = safe_extract(data, 'macd_signal', None)
        vwap = safe_extract(data, 'vwap', None)
        obv = safe_extract(data, 'obv', None)
        
        # Calculate basic metrics if not available
        if atr is None and 'High' in data.columns and 'Low' in data.columns:
            # Calculate ATR
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
        # Calculate ATR percent if we have ATR and current price
        if atr is not None and current_price > 0 and atr_percent is None:
            atr_percent = (atr / current_price) * 100
            
        # Calculate EMA9 and EMA21 if not available
        if ema9 is None and 'Close' in data.columns and len(data) >= 9:
            ema9 = data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            
        if ema21 is None and 'Close' in data.columns and len(data) >= 21:
            ema21 = data['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
            
        # Calculate percent change
        percent_change = None
        if 'Close' in data.columns and len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            if not pd.isna(prev_close) and prev_close > 0:
                percent_change = ((current_price - prev_close) / prev_close) * 100
                
        # Calculate volume metrics
        avg_volume = None
        if 'Volume' in data.columns and len(data) > 0:
            avg_volume = data['Volume'].mean()
            
        volume_change = None
        if 'Volume' in data.columns and len(data) > 1:
            prev_volume = data['Volume'].iloc[-2]
            if not pd.isna(prev_volume) and prev_volume > 0:
                volume_change = ((volume - prev_volume) / prev_volume) * 100
                
        # Get start and end dates
        start_date = data.index[0].strftime("%Y-%m-%d") if hasattr(data.index[0], 'strftime') else str(data.index[0])
        end_date = data.index[-1].strftime("%Y-%m-%d") if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        
        # Get quality score if available
        quality_score = data.attrs.get('quality_score', 100) if hasattr(data, 'attrs') else 100
        
        # Prepare result dictionary with maximum data available
        result = {
            "ticker": ticker,
            "date": date_str,
            "current_price": float_or_none(current_price),
            "open": float_or_none(open_price),
            "high": float_or_none(high_price),
            "low": float_or_none(low_price),
            "volume": int_or_none(volume),
            "avg_volume": int_or_none(avg_volume),
            "percent_change": float_or_none(percent_change),
            "volume_change": float_or_none(volume_change),
            "atr": float_or_none(atr),
            "atr_percent": float_or_none(atr_percent),
            "ema9": float_or_none(ema9),
            "ema21": float_or_none(ema21),
            "ema50": float_or_none(ema50),
            "sma50": float_or_none(sma50),
            "sma200": float_or_none(sma200),
            "data_points": len(data),
            "start_date": start_date,
            "end_date": end_date,
            "quality_score": quality_score
        }
        
        # Add additional indicators if available
        if rsi is not None:
            result["rsi"] = float_or_none(rsi)
        if macd is not None:
            result["macd"] = float_or_none(macd)
        if macd_signal is not None:
            result["macd_signal"] = float_or_none(macd_signal)
        if vwap is not None:
            result["vwap"] = float_or_none(vwap)
        if obv is not None:
            result["obv"] = float_or_none(obv)
            
        return result
    
    except Exception as e:
        logger.error(f"Error formatting stock data for {ticker}: {e}")
        return {
            "ticker": ticker,
            "date": date_str,
            "error": str(e)
        }

def format_market_data(market_data):
    """
    Format market data for analysis
    
    Args:
        market_data: Dictionary with market data for various indices
        
    Returns:
        Dictionary with formatted market data
    """
    try:
        # Skip if no market data
        if not market_data:
            logger.error("No market data to format")
            return {"error": "No market data available"}
        
        result = {}
        
        # Process each index
        for index, data in market_data.items():
            if "info" not in data:
                continue
                
            index_info = data["info"]
            
            # Extract key metrics
            result[index] = {
                "price": float_or_none(index_info.get("regularMarketPrice")),
                "open": float_or_none(index_info.get("open")),
                "high": float_or_none(index_info.get("dayHigh")),
                "low": float_or_none(index_info.get("dayLow")),
                "previous_close": float_or_none(index_info.get("previousClose")),
                "volume": int_or_none(index_info.get("regularMarketVolume")),
                "avg_volume": int_or_none(index_info.get("averageVolume")),
                "fifty_day_avg": float_or_none(index_info.get("fiftyDayAverage")),
                "two_hundred_day_avg": float_or_none(index_info.get("twoHundredDayAverage")),
                "ema9": float_or_none(index_info.get("ema9")),
                "ema21": float_or_none(index_info.get("ema21")),
                "daily_change": float_or_none(index_info.get("dailyChange")),
                "daily_pct_change": float_or_none(index_info.get("dailyPctChange"))
            }
            
        # Add VIX data if available
        if "VIX" in result or "^VIX" in result:
            vix_key = "VIX" if "VIX" in result else "^VIX"
            vix_price = result[vix_key]["price"]
            
            # Determine market stability based on VIX
            market_stability = "normal"
            risk_adjustment = 1.0
            
            if vix_price < 15:
                market_stability = "very_stable"
            elif vix_price < 20:
                market_stability = "stable"
            elif vix_price < 25:
                market_stability = "normal"
            elif vix_price < 35:
                market_stability = "elevated"
                risk_adjustment = 0.5
            else:
                market_stability = "extreme"
                risk_adjustment = 0.0
                
            result["vix_assessment"] = {
                "price": vix_price,
                "stability": market_stability,
                "risk_adjustment": risk_adjustment
            }
            
        return result
    
    except Exception as e:
        logger.error(f"Error formatting market data: {e}")
        return {"error": str(e)}

def format_spread_data(spread_data, ticker):
    """
    Format spread data for analysis
    
    Args:
        spread_data: Dictionary with spread opportunities
        ticker: Stock symbol
        
    Returns:
        Dictionary with formatted spread data
    """
    try:
        # Skip if no spread data
        if not spread_data:
            logger.error(f"No spread data to format for {ticker}")
            return {"ticker": ticker, "error": "No spread data available"}
        
        result = {
            "ticker": ticker,
            "expiration_date": spread_data.get("expiration_date"),
            "current_price": float_or_none(spread_data.get("current_price")),
            "bull_put_spreads": [],
            "bear_call_spreads": []
        }
        
        # Format bull put spreads
        for spread in spread_data.get("bull_put_spreads", []):
            result["bull_put_spreads"].append({
                "short_strike": float_or_none(spread.get("short_strike")),
                "long_strike": float_or_none(spread.get("long_strike")),
                "width": float_or_none(spread.get("width")),
                "credit": float_or_none(spread.get("credit")),
                "max_risk": float_or_none(spread.get("max_risk")),
                "return_on_risk": float_or_none(spread.get("return_on_risk")),
                "short_delta": float_or_none(spread.get("short_delta"))
            })
            
        # Format bear call spreads
        for spread in spread_data.get("bear_call_spreads", []):
            result["bear_call_spreads"].append({
                "short_strike": float_or_none(spread.get("short_strike")),
                "long_strike": float_or_none(spread.get("long_strike")),
                "width": float_or_none(spread.get("width")),
                "credit": float_or_none(spread.get("credit")),
                "max_risk": float_or_none(spread.get("max_risk")),
                "return_on_risk": float_or_none(spread.get("return_on_risk")),
                "short_delta": float_or_none(spread.get("short_delta"))
            })
            
        return result
    
    except Exception as e:
        logger.error(f"Error formatting spread data for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

def safe_extract(data, column, default=None):
    """
    Safely extract a value from a DataFrame
    
    Args:
        data: DataFrame or Series to extract from
        column: Column name to extract
        default: Default value if extraction fails
        
    Returns:
        Extracted value
    """
    try:
        if column not in data.columns:
            return default
            
        value = data[column].iloc[-1]
        if isinstance(value, pd.Series):
            value = value.iloc[0] if len(value) > 0 else default
            
        return value
    except Exception as e:
        logger.debug(f"Error extracting {column} from data: {e}")
        return default

def float_or_none(value):
    """Convert value to float if possible, otherwise return None"""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def int_or_none(value):
    """Convert value to int if possible, otherwise return None"""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None 