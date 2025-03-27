import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_historical_data(data, symbol, required_columns=None, nan_threshold=20):
    """
    Validate that historical data meets quality criteria

    Args:
        data: DataFrame with historical data
        symbol: Ticker symbol for logging
        required_columns: List of columns that must be present
        nan_threshold: Maximum percentage of NaN values allowed

    Returns:
        Tuple of (is_valid, validation_message, quality_score)
    """
    if data is None or len(data) == 0:
        return False, f"Empty dataset for {symbol}", 0

    # Validate required columns
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    missing_columns = [
        col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing columns for {symbol}: {missing_columns}", 0

    # Check for NaN values
    nan_percentage = data.isna().mean().mean() * 100
    if nan_percentage > nan_threshold:
        return False, f"Excessive NaN values ({nan_percentage:.1f}%) for {symbol}", 0

    # Calculate quality score (0-100)
    quality_score = 100 - nan_percentage

    return True, f"Data validation passed for {symbol}, quality: {quality_score:.1f}%", quality_score


def add_data_metadata(data, quality_score, validation_message=None):
    """
    Add metadata to DataFrame for tracking quality and validation

    Args:
        data: DataFrame to add metadata to
        quality_score: Quality score (0-100)
        validation_message: Optional validation message

    Returns:
        DataFrame with added metadata
    """
    # Clone to avoid modifying the original
    df = data.copy()

    # Add metadata to dataframe
    df.attrs['quality_score'] = quality_score
    df.attrs['retrieval_timestamp'] = datetime.now().isoformat()
    if validation_message:
        df.attrs['validation_message'] = validation_message

    return df


def save_to_cache(data, symbol, start=None, end=None, interval="1d", cache_dir="./data-cache"):
    """
    Save data to local cache

    Args:
        data: DataFrame to save
        symbol: Ticker symbol
        start: Start date string
        end: End date string
        interval: Data interval
        cache_dir: Directory to save cache files

    Returns:
        Boolean indicating success
    """
    try:
        # Create cache directory if it doesn't exist
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        # Create a unique filename based on parameters
        filename = f"{symbol}_{start}_{end}_{interval}.parquet"
        cache_path = cache_dir / filename

        # Save to parquet format (efficient storage)
        data.to_parquet(cache_path)
        logger.info(f"Saved data to cache: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
        return False


def load_from_cache(symbol, start=None, end=None, interval="1d", cache_dir="./data-cache", max_age_days=1):
    """
    Load data from local cache if available

    Args:
        symbol: Ticker symbol
        start: Start date string
        end: End date string
        interval: Data interval
        cache_dir: Directory for cache files
        max_age_days: Maximum age of cache in days

    Returns:
        DataFrame if cache exists and is fresh, otherwise None
    """
    try:
        cache_dir = Path(cache_dir)
        filename = f"{symbol}_{start}_{end}_{interval}.parquet"
        cache_path = cache_dir / filename

        if not cache_path.exists():
            return None

        # Check if cache is too old
        file_age = time.time() - cache_path.stat().st_mtime
        max_age = max_age_days * 86400  # Convert days to seconds

        # For intraday data, use shorter max age
        if interval != "1d":
            max_age = min(max_age, 14400)  # 4 hours for intraday

        if file_age > max_age:
            logger.info(
                f"Cache for {symbol} is outdated ({file_age/3600:.1f} hours old)")
            return None

        # Load the cached data
        data = pd.read_parquet(cache_path)
        logger.info(f"Loaded cached data for {symbol}, shape: {data.shape}")

        # Add cache metadata
        data.attrs['loaded_from_cache'] = True
        data.attrs['cache_age_hours'] = file_age / 3600

        return data
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None
