import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def calculate_moving_averages(data, periods=None):
    """
    Calculate moving averages for price data

    Args:
        data: DataFrame with OHLCV data
        periods: List of periods for moving averages [9, 21, 50, 200] by default

    Returns:
        DataFrame with added moving averages
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Handle multi-index columns
        is_multi_index = isinstance(df.columns, pd.MultiIndex)
        close_col = 'Close'

        if periods is None:
            periods = [9, 21, 50, 200]

        # Calculate SMAs
        for period in periods:
            col_name = f'SMA_{period}'
            df[col_name] = df[close_col].rolling(window=period).mean()

        # Calculate EMAs
        for period in periods:
            col_name = f'EMA_{period}'
            df[col_name] = df[close_col].ewm(span=period, adjust=False).mean()

        # Calculate specific EMAs for common use
        df['ema9'] = df[close_col].ewm(span=9, adjust=False).mean()
        df['ema21'] = df[close_col].ewm(span=21, adjust=False).mean()

        return df
    except Exception as e:
        logger.error(f"Error calculating moving averages: {e}")
        return data


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands

    Args:
        data: DataFrame with OHLCV data
        window: Window size for moving average (default: 20)
        num_std: Number of standard deviations (default: 2)

    Returns:
        DataFrame with added Bollinger Bands
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Calculate middle band (SMA)
        df['Bollinger_Middle'] = df['Close'].rolling(window=window).mean()

        # Calculate standard deviation
        std_dev = df['Close'].rolling(window=window).std()

        # Calculate upper and lower bands
        df['Bollinger_Upper'] = df['Bollinger_Middle'] + (std_dev * num_std)
        df['Bollinger_Lower'] = df['Bollinger_Middle'] - (std_dev * num_std)

        # Calculate bandwidth
        df['Bollinger_Width'] = (
            df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['Bollinger_Middle']

        return df
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return data


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)

    Args:
        data: DataFrame with OHLCV data
        window: Window size for RSI calculation (default: 14)

    Returns:
        DataFrame with added RSI
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Calculate price changes
        delta = df['Close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return data


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)

    Args:
        data: DataFrame with OHLCV data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        DataFrame with added MACD
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Verify we have enough data points
        if len(df) < slow:
            logger.warning(
                f"Insufficient data for MACD calculation: {len(df)} points (need {slow})")
            # Add NaN MACD values to avoid errors
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            df['MACD_Histogram'] = np.nan
            return df

        # Check for NaN values in close data
        if df['Close'].isna().any():
            logger.warning(
                "NaN values detected in Close data, this may affect MACD calculation")
            # Fill NaN values with forward fill then backward fill
            df['Close'] = df['Close'].fillna(
                method='ffill').fillna(method='bfill')

        # Calculate fast and slow EMAs
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        df['MACD'] = ema_fast - ema_slow

        # Calculate signal line
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Check if we produced valid values
        if df['MACD'].isna().all():
            logger.error(
                f"MACD calculation returned all NaN values despite sufficient data length {len(df)}")
        elif df['MACD'].isna().any():
            nan_count = df['MACD'].isna().sum()
            logger.warning(
                f"MACD calculation has {nan_count} NaN values out of {len(df)} rows")

        return df
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        # Add NaN MACD values to avoid further errors
        df = data.copy()
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan
        df['MACD_Histogram'] = np.nan
        return df


def calculate_volatility(data, windows=None):
    """
    Calculate volatility indicators

    Args:
        data: DataFrame with OHLCV data
        windows: List of windows for volatility calculation [5, 10, 20] by default

    Returns:
        DataFrame with added volatility indicators
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        if windows is None:
            windows = [5, 10, 20]

        # Calculate daily returns (fix deprecated warning)
        df['Returns'] = df['Close'].pct_change(fill_method=None)

        # Calculate volatility (standard deviation of returns)
        for window in windows:
            df[f'Volatility_{window}'] = df['Returns'].rolling(
                window=window).std() * np.sqrt(252)  # Annualized

        # Calculate ATR
        tr1 = abs(df['High'] - df['Low'])
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()

        return df
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return data


def calculate_fibonacci_levels(data):
    """
    Calculate Fibonacci retracement levels

    Args:
        data: DataFrame with OHLCV data

    Returns:
        Dictionary with Fibonacci levels
    """
    try:
        # Handle multi-index columns
        is_multi_index = isinstance(data.columns, pd.MultiIndex)
        high_col = 'High'
        low_col = 'Low'

        # Find recent high and low
        recent_high = data[high_col].tail(100).max()
        recent_low = data[low_col].tail(100).min()

        # Calculate range - properly handle Series
        if isinstance(recent_high, pd.Series):
            recent_high = recent_high.iloc[0]
        if isinstance(recent_low, pd.Series):
            recent_low = recent_low.iloc[0]

        price_range = recent_high - recent_low

        # Ensure we're working with simple numbers, not Series
        if isinstance(price_range, pd.Series):
            price_range = price_range.iloc[0]

        # Calculate Fibonacci levels - properly handle Series
        fib_levels = {
            'high': float(recent_high),
            'low': float(recent_low),
            'fib_0': float(recent_low),  # 0%
            'fib_236': float(recent_low + 0.236 * price_range),  # 23.6%
            'fib_382': float(recent_low + 0.382 * price_range),  # 38.2%
            'fib_50': float(recent_low + 0.5 * price_range),     # 50%
            'fib_618': float(recent_low + 0.618 * price_range),  # 61.8%
            'fib_786': float(recent_low + 0.786 * price_range),  # 78.6%
            'fib_100': float(recent_high)  # 100%
        }

        return fib_levels
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}


def calculate_volatility_indicators(data):
    """
    Calculate volatility indicators like ATR

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with added volatility indicators
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Check if data has multi-index columns (common with yfinance)
        is_multi_index = isinstance(df.columns, pd.MultiIndex)

        # Get column names accounting for multi-index
        high_col = 'High'
        low_col = 'Low'
        close_col = 'Close'

        # ATR calculation
        tr1 = abs(df[high_col] - df[low_col])
        tr2 = abs(df[high_col] - df[close_col].shift(1))
        tr3 = abs(df[low_col] - df[close_col].shift(1))
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()

        # Calculate ATR as percentage of close
        try:
            if is_multi_index:
                # For multi-index, need to access close differently
                atr_percent_values = (df['atr'] / df[close_col]) * 100
                # Ensure we're assigning a Series, not a DataFrame
                if isinstance(atr_percent_values, pd.DataFrame):
                    atr_percent_values = atr_percent_values.iloc[:, 0]
                df['atr_percent'] = atr_percent_values
            else:
                atr_percent_values = (df['atr'] / df[close_col]) * 100
                # Ensure we're assigning a Series, not a DataFrame
                if isinstance(atr_percent_values, pd.DataFrame):
                    atr_percent_values = atr_percent_values.iloc[:, 0]
                df['atr_percent'] = atr_percent_values
        except Exception as e:
            logger.error(f"Error calculating ATR percentage: {e}")
            # Provide a fallback value
            df['atr_percent'] = 2.0  # Default 2% ATR

        # Bollinger Bands
        df['bollinger_mid'] = df[close_col].rolling(window=20).mean()
        df['stddev'] = df[close_col].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_mid'] + (df['stddev'] * 2)
        df['bollinger_lower'] = df['bollinger_mid'] - (df['stddev'] * 2)
        df['bollinger_width'] = (
            df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_mid']

        return df
    except Exception as e:
        logger.error(f"Error calculating volatility indicators: {e}")
        return data


def calculate_momentum_indicators(data):
    """
    Calculate momentum indicators like RSI, MACD

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with added momentum indicators
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Check if data has multi-index columns (common with yfinance)
        is_multi_index = isinstance(df.columns, pd.MultiIndex)

        # Get column names accounting for multi-index
        close_col = 'Close'

        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df
    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {e}")
        return data


def calculate_volume_indicators(data, ticker=None):
    """
    Calculate volume-based indicators

    Args:
        data: DataFrame with OHLCV data
        ticker: Optional ticker symbol to handle special cases

    Returns:
        DataFrame with added volume indicators
    """
    try:
        # Clone the dataframe to avoid modifying original
        df = data.copy()

        # Skip volume calculations for indices that typically don't have volume data
        indices_without_volume = ["^VIX", "VIX", "^GSPC", "^DJI", "^IXIC"]
        if ticker and ticker in indices_without_volume:
            logger.info(f"Skipping volume calculations for index {ticker}")
            # Add placeholder values to avoid errors
            df['volume_ema20'] = np.nan
            df['volume_ratio'] = 1.0  # Default neutral value
            df['obv'] = np.nan
            return df

        # Check if Volume column exists and has valid data
        if 'Volume' not in df.columns.tolist():
            logger.warning(
                "No Volume column found in data, skipping volume calculations")
            # Add placeholder values
            df['volume_ema20'] = np.nan
            df['volume_ratio'] = 1.0
            df['obv'] = np.nan
            return df

        # Check if all volume values are zero or NaN
        if df['Volume'].isnull().all() or (df['Volume'] == 0).all():
            logger.warning(
                "All Volume values are zero or NaN, skipping volume calculations")
            # Add placeholder values
            df['volume_ema20'] = np.nan
            df['volume_ratio'] = 1.0
            df['obv'] = np.nan
            return df

        # Calculate volume EMAs
        df['volume_ema20'] = df['Volume'].ewm(span=20).mean()

        # Calculate volume to average ratio
        df['volume_ratio'] = df['Volume'] / df['volume_ema20']

        # Avoid division by zero errors
        df['volume_ratio'] = df['volume_ratio'].replace(
            [np.inf, -np.inf], np.nan).fillna(1.0)

        # Calculate on-balance volume (OBV) - fixed with proper vector approach
        obv = pd.Series(0, index=df.index)
        close_diff = df['Close'].diff()

        # Use vectorized operations instead of loop
        obv[close_diff > 0] = df['Volume'][close_diff > 0]
        obv[close_diff < 0] = -df['Volume'][close_diff < 0]
        df['obv'] = obv.cumsum()

        return df
    except Exception as e:
        logger.error(f"Error calculating volume indicators: {e}")
        # Add placeholder values to avoid further errors
        df = data.copy()
        df['volume_ema20'] = np.nan
        df['volume_ratio'] = 1.0
        df['obv'] = np.nan
        return df


def calculate_support_resistance(data, lookback=20):
    """
    Calculate support and resistance levels based on recent highs and lows

    Args:
        data: DataFrame with OHLCV data
        lookback: Number of periods to look back for highs and lows

    Returns:
        Dictionary with support and resistance levels
    """
    try:
        # Extract recent lows and highs
        lows_array = data['Low'].tail(lookback).to_numpy()
        highs_array = data['High'].tail(lookback).to_numpy()

        # Sort for finding clusters
        lows_array = np.sort(lows_array)
        highs_array = -np.sort(-highs_array)

        # Get the average of the 3 highest highs and 3 lowest lows
        support = np.mean(lows_array[:3])
        resistance = np.mean(highs_array[:3])

        # Get the current price
        current_price = data['Close'].iloc[-1]

        # Handle Series objects properly
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]

        # Calculate distance as percentage
        support_distance = ((current_price - support) / current_price) * 100
        resistance_distance = (
            (resistance - current_price) / current_price) * 100

        # Determine if price is near support or resistance (within 2%)
        near_support = bool(support_distance <= 2.0)
        near_resistance = bool(resistance_distance <= 2.0)

        return {
            'support': float(support),
            'resistance': float(resistance),
            'support_distance': float(support_distance),
            'resistance_distance': float(resistance_distance),
            'near_support': near_support,
            'near_resistance': near_resistance
        }
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        # Return default values
        current_price = data['Close'].iloc[-1] if len(data) > 0 else 100
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]

        return {
            'support': float(current_price * 0.95),
            'resistance': float(current_price * 1.05),
            'support_distance': 5.0,
            'resistance_distance': 5.0,
            'near_support': False,
            'near_resistance': False
        }


def calculate_all_indicators(data, ticker=None):
    """
    Calculate all technical indicators at once

    Args:
        data: DataFrame with OHLCV data
        ticker: Optional ticker symbol to handle special cases

    Returns:
        DataFrame with all indicators calculated
    """
    try:
        # Apply all indicator calculations sequentially
        df = data.copy()
        df = calculate_moving_averages(df)
        df = calculate_volatility_indicators(df)
        df = calculate_momentum_indicators(df)
        df = calculate_volume_indicators(df, ticker=ticker)

        # Calculate support/resistance levels
        support_resistance = calculate_support_resistance(df)

        # Add support/resistance to DataFrame attributes
        df.attrs['support_levels'] = support_resistance['support']
        df.attrs['resistance_levels'] = support_resistance['resistance']
        df.attrs['near_support'] = support_resistance['near_support']
        df.attrs['near_resistance'] = support_resistance['near_resistance']

        # Calculate Fibonacci levels
        fibonacci_levels = calculate_fibonacci_levels(df)
        df.attrs['fibonacci_levels'] = fibonacci_levels

        # Add a processed flag to the DataFrame
        df.attrs['indicators_calculated'] = True
        df.attrs['calculation_timestamp'] = pd.Timestamp.now().isoformat()

        return df
    except Exception as e:
        logger.error(f"Error calculating all indicators: {e}")
        return data
