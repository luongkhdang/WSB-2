import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# Import utilities
from src.finance_client.utilities.data_validation import (
    validate_historical_data,
    add_data_metadata,
    save_to_cache,
    load_from_cache
)

logger = logging.getLogger(__name__)

def get_historical_data(symbol, start=None, end=None, interval="1d", cache_dir="./data-cache", max_retries=3):
    """
    Get historical data with validation, caching and retry logic
    
    Args:
        symbol: Ticker symbol
        start: Start date in format 'YYYY-MM-DD'
        end: End date in format 'YYYY-MM-DD'
        interval: Data interval ('1d', '1h', '15m', etc.)
        cache_dir: Directory for cache files
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with historical data
    """
    logger.info(f"Getting historical data for {symbol} from {start} to {end} with interval {interval}")
    
    # Try to load from cache first
    cached_data = load_from_cache(symbol, start, end, interval, cache_dir)
    if cached_data is not None:
        # Validate the cached data
        is_valid, message, quality_score = validate_historical_data(cached_data, symbol)
        if is_valid:
            logger.info(f"Using cached data for {symbol}, quality: {quality_score:.1f}%")
            return add_data_metadata(cached_data, quality_score, message)
        else:
            logger.warning(f"Cached data for {symbol} failed validation: {message}")
    
    # Try to retrieve fresh data with retries
    retry_count = 0
    base_delay = 2  # Base delay in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempt {retry_count+1}: Getting historical data for {symbol}")
            data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
            
            # Validate the data
            is_valid, message, quality_score = validate_historical_data(data, symbol)
            
            if is_valid:
                # Add metadata
                result = add_data_metadata(data, quality_score, message)
                
                # Save to cache if successful
                save_to_cache(result, symbol, start, end, interval, cache_dir)
                
                logger.info(f"Successfully retrieved data for {symbol}, shape: {data.shape}, quality: {quality_score:.1f}%")
                return result
            else:
                logger.warning(f"Attempt {retry_count+1}: Retrieved data failed validation: {message}")
                retry_count += 1
                time.sleep(base_delay * (2 ** retry_count))  # Exponential backoff
                
        except Exception as e:
            logger.error(f"Attempt {retry_count+1}: Error retrieving data for {symbol}: {e}")
            retry_count += 1
            time.sleep(base_delay * (2 ** retry_count))
    
    # All retries failed, return None or an empty DataFrame with error metadata
    logger.error(f"All {max_retries} attempts failed for {symbol}")
    empty_data = pd.DataFrame()
    empty_data.attrs['error'] = f"Failed to retrieve valid data after {max_retries} attempts"
    empty_data.attrs['quality_score'] = 0
    return empty_data

def get_market_data(indices=None, cache_dir="./data-cache"):
    """
    Get comprehensive market data for multiple indices
    
    Args:
        indices: List of indices to retrieve (default: key market indices)
        cache_dir: Directory for cache files
        
    Returns:
        Dictionary with market data for each index
    """
    logger.info("Getting market data")
    
    # Default indices if none provided
    if indices is None:
        indices = ["SPY", "QQQ", "IWM", "VTV", "VGLT", "^VIX", "DIA", "BND", "BTC-USD"]
    
    # Get today and yesterday's dates
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Dictionary to store all market data
    market_data = {}
    
    # Get data for each index
    for index in indices:
        try:
            # Get 5 days of hourly data
            index_data = get_historical_data(
                index, 
                start=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                end=today, 
                interval="1h",
                cache_dir=cache_dir
            )
            
            if index_data is None or len(index_data) == 0:
                logger.warning(f"No data retrieved for {index}")
                continue
            
            # Calculate basic metrics
            current_price = index_data['Close'].iloc[-1] if len(index_data) > 0 else None
            open_price = index_data['Open'].iloc[-1] if len(index_data) > 0 else None
            day_high = index_data['High'].iloc[-1] if len(index_data) > 0 else None
            day_low = index_data['Low'].iloc[-1] if len(index_data) > 0 else None
            previous_close = index_data['Close'].iloc[-2] if len(index_data) > 1 else None
            
            # Calculate daily change
            daily_change = current_price - previous_close if current_price and previous_close else None
            daily_pct_change = (daily_change / previous_close * 100) if daily_change and previous_close else None
            
            # Get monthly data for longer-term averages
            monthly_data = get_historical_data(
                index, 
                start=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                end=today, 
                interval="1d",
                cache_dir=cache_dir
            )
            
            # Calculate 50-day and 200-day averages if we have enough data
            fifty_day_avg = None
            two_hundred_day_avg = None
            
            if monthly_data is not None and len(monthly_data) > 0:
                if len(monthly_data) >= 50:
                    fifty_day_avg = monthly_data['Close'].rolling(window=50).mean().iloc[-1]
                
                if len(monthly_data) >= 200:
                    two_hundred_day_avg = monthly_data['Close'].rolling(window=200).mean().iloc[-1]
            
            # Calculate EMAs
            ema9 = index_data['Close'].ewm(span=9, adjust=False).mean().iloc[-1] if len(index_data) > 0 else None
            ema21 = index_data['Close'].ewm(span=21, adjust=False).mean().iloc[-1] if len(index_data) > 0 else None
            
            # Store info and raw data
            market_data[index] = {
                "info": {
                    "regularMarketPrice": current_price,
                    "open": open_price,
                    "dayHigh": day_high,
                    "dayLow": day_low,
                    "previousClose": previous_close,
                    "regularMarketVolume": index_data['Volume'].iloc[-1] if len(index_data) > 0 else None,
                    "averageVolume": index_data['Volume'].mean() if len(index_data) > 0 else None,
                    "fiftyDayAverage": fifty_day_avg,
                    "twoHundredDayAverage": two_hundred_day_avg,
                    "ema9": ema9,
                    "ema21": ema21,
                    "dailyChange": daily_change,
                    "dailyPctChange": daily_pct_change
                },
                "history": index_data
            }
            
            # If monthly data is available, add it
            if monthly_data is not None and len(monthly_data) > 0:
                market_data[index]["monthly_history"] = monthly_data
                
        except Exception as e:
            logger.error(f"Error retrieving data for {index}: {e}")
    
    # Return None if no data for SPY (essential index)
    if "SPY" not in market_data:
        logger.error("Failed to retrieve SPY data, which is required for market analysis")
        return None
        
    return market_data

def get_options_chain(symbol, expiration_date=None, cache_dir="./data-cache"):
    """
    Get options chain data for a symbol
    
    Args:
        symbol: Ticker symbol
        expiration_date: Optional specific expiration date
        cache_dir: Directory for cache files
        
    Returns:
        Dictionary with options chain data
    """
    logger.info(f"Getting options chain for {symbol}")
    
    try:
        # Create a filename for caching
        cache_key = f"{symbol}_options"
        if expiration_date:
            cache_key += f"_{expiration_date}"
            
        # Try to load from cache
        cached_data = load_from_cache(cache_key, cache_dir=cache_dir, max_age_days=0.5)
        if cached_data is not None and 'options_chain' in cached_data.attrs:
            logger.info(f"Using cached options data for {symbol}")
            return cached_data.attrs['options_chain']
        
        # Get ticker data
        ticker = yf.Ticker(symbol)
        
        # Get current price
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        
        # Get expiration dates if none provided
        if not expiration_date:
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return None
                
            # Find shortest expiration date
            expiration_date = min(expirations)
            
        # Get options chain
        opt = ticker.option_chain(expiration_date)
        
        if not opt or not hasattr(opt, 'calls') or not hasattr(opt, 'puts'):
            logger.warning(f"Invalid options data for {symbol}")
            return None
            
        # Create options chain dictionary
        options_chain = {
            'symbol': symbol,
            'expiration_date': expiration_date,
            'current_price': current_price,
            'calls': opt.calls,
            'puts': opt.puts,
            'retrieval_timestamp': datetime.now().isoformat()
        }
        
        # Save to cache
        # Create a dummy DataFrame to use with our caching system
        cache_df = pd.DataFrame({'dummy': [0]})
        cache_df.attrs['options_chain'] = options_chain
        save_to_cache(cache_df, cache_key, cache_dir=cache_dir)
        
        return options_chain
        
    except Exception as e:
        logger.error(f"Error getting options chain for {symbol}: {e}")
        return None

def get_volatility_data(cache_dir="./data-cache"):
    """
    Get VIX data for market volatility analysis
    
    Args:
        cache_dir: Directory for cache files
        
    Returns:
        Dictionary with VIX data and volatility assessment
    """
    logger.info("Getting VIX data")
    
    try:
        # Get VIX data
        vix_data = get_historical_data(
            "^VIX", 
            start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"), 
            interval="1d",
            cache_dir=cache_dir
        )
        
        if vix_data is None or len(vix_data) == 0:
            logger.error("Failed to retrieve VIX data")
            logger.info("Using fallback VIX data")
            # Provide fallback data
            return {
                "price": 19.5,  # Moderate VIX level
                "historical_data": None,
                "stability": "normal",
                "risk_adjustment": 1.0,
                "adjustment_note": "Standard position sizing (fallback data)"
            }
        
        # Extract the last VIX price
        vix_price = vix_data['Close'].iloc[-1]
        
        # Calculate market stability score
        if vix_price < 15:
            stability = "very_stable"
            risk_adjustment = 1.0
            adjustment_note = "Standard position sizing (1-2% of account)"
        elif vix_price < 20:
            stability = "stable"
            risk_adjustment = 1.0
            adjustment_note = "Standard position sizing with +5 to Market Trend score"
        elif vix_price < 25:
            stability = "normal"
            risk_adjustment = 1.0
            adjustment_note = "Standard position sizing"
        elif vix_price < 35:
            stability = "elevated"
            risk_adjustment = 0.5
            adjustment_note = "Reduce position size by 50% (-5 to score unless size halved)"
        else:
            stability = "extreme"
            risk_adjustment = 0.0
            adjustment_note = "Skip unless justified by high Gamble Score (>80)"
        
        # Calculate VIX percentile compared to last 30 days
        vix_30d_low = vix_data['Close'].min()
        vix_30d_high = vix_data['Close'].max()
        vix_range = vix_30d_high - vix_30d_low
        
        if vix_range > 0:
            vix_percentile = (vix_price - vix_30d_low) / vix_range * 100
        else:
            vix_percentile = 50
            
        # Detect if VIX is rising or falling
        vix_5d_ago = vix_data['Close'].iloc[-6] if len(vix_data) >= 6 else vix_price
        vix_trend = "rising" if vix_price > vix_5d_ago else "falling"
        
        return {
            "price": float(vix_price),
            "historical_data": vix_data,
            "stability": stability,
            "risk_adjustment": risk_adjustment,
            "adjustment_note": adjustment_note,
            "vix_30d_low": float(vix_30d_low),
            "vix_30d_high": float(vix_30d_high),
            "vix_percentile": float(vix_percentile),
            "vix_trend": vix_trend
        }
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
        logger.info("Using fallback VIX data")
        return {
            "price": 22.5,
            "historical_data": None,
            "stability": "normal",
            "risk_adjustment": 1.0,
            "adjustment_note": "Standard position sizing (fallback data)"
        } 