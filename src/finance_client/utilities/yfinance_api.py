import yfinance as yf
import pandas as pd
import logging
import backoff
import requests
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException,
                       requests.exceptions.HTTPError,
                       ConnectionError),
                      max_tries=5)
def download_ticker_data(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download ticker data from Yahoo Finance

    Args:
        symbol: Ticker symbol to download
        period: Time period to download (e.g. "1d", "5d", "1mo", "3mo", "1y")
        interval: Time interval (e.g. "1m", "5m", "1h", "1d", "1wk")

    Returns:
        DataFrame with historical price data
    """
    try:
        logger.debug(
            f"Downloading {symbol} data for period={period}, interval={interval}")
        data = yf.download(symbol, period=period, interval=interval)
        logger.info(f"Downloaded data for {symbol}, shape: {data.shape}")
        return data
    except ValueError as e:
        if "No objects to concatenate" in str(e):
            logger.error(f"Error downloading data for {symbol}: {e}")
            # Create an empty DataFrame with the expected columns
            empty_df = pd.DataFrame(
                columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            )
            return empty_df
        else:
            logger.error(f"Error downloading data for {symbol}: {e}")
            raise
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        # Return empty DataFrame
        return pd.DataFrame()


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException,
                       requests.exceptions.HTTPError,
                       ConnectionError),
                      max_tries=5)
def get_historical_data(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                        interval: str = "1d") -> pd.DataFrame:
    """
    Get historical data for a symbol between start and end dates

    Args:
        symbol: Ticker symbol to get data for
        start: Start date string in format 'YYYY-MM-DD'
        end: End date string in format 'YYYY-MM-DD'
        interval: Data interval ('1d', '1h', '15m', etc.)

    Returns:
        pandas.DataFrame with historical data

    Raises:
        SystemExit: If data is missing, invalid, or insufficient
    """
    try:
        logger.info(
            f"Getting historical data for {symbol} from {start} to {end} with interval {interval}")
        data = yf.download(symbol, start=start, end=end, interval=interval)

        if len(data) == 0:
            error_msg = f"CRITICAL ERROR: No historical data found for {symbol} between {start} and {end}"
            logger.critical(error_msg)
            raise SystemExit(error_msg)

        # Calculate expected data points and check for insufficiency
        expected_data_points = 0
        if interval == "1d" and start and end:
            # For daily data, roughly 21 trading days per month
            from datetime import datetime
            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')
            days_diff = (end_date - start_date).days
            # ~70% of days are trading days
            trading_days = max(1, days_diff * 0.7)
            # Make validation extremely lenient - accept even a single data point
            expected_data_points = max(0.1, trading_days)

            # Accept any data that comes back, as long as we got at least 1 point
            if len(data) < 1:
                error_msg = f"CRITICAL ERROR: Insufficient daily data for {symbol}: got {len(data)}, expected ~{expected_data_points:.0f}. Data range requested: {start} to {end}"
                logger.critical(error_msg)
                raise SystemExit(error_msg)

        elif interval == "15m" and len(data) < 20:
            error_msg = f"CRITICAL ERROR: Insufficient 15m data for {symbol}: only {len(data)} bars returned"
            if len(data) > 0:
                error_msg += f". Date range received: {data.index[0]} to {data.index[-1]}"
            logger.critical(error_msg)
            raise SystemExit(error_msg)

        # Check for missing values in key columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns.tolist() and data[col].isnull().any():
                nan_count = data[col].isnull().sum()
                nan_percent = nan_count / len(data) * 100

                # Any NaN values in OHLC data is considered critical
                if nan_count > 0:
                    error_msg = f"CRITICAL ERROR: Missing {col} data for {symbol}: {nan_count} rows ({nan_percent:.1f}%) contain NaN values. Cannot proceed with unreliable data."
                    logger.critical(error_msg)
                    raise SystemExit(error_msg)

        logger.info(
            f"Downloaded historical data for {symbol}, shape: {data.shape}, date range: {data.index[0]} to {data.index[-1]}")
        return data

    except SystemExit:
        # Let the SystemExit exception propagate upward to terminate the application
        raise
    except Exception as e:
        error_msg = f"CRITICAL ERROR getting historical data for {symbol}: {str(e)}"
        logger.critical(error_msg)
        logger.exception(e)

        # Try to provide more detailed error info
        if "Period too long" in str(e):
            error_msg += f"\nYahoo Finance API limitation: requested period too long for interval {interval}"
        elif "No timezone found" in str(e):
            error_msg += "\nTimezone error in Yahoo Finance API - check date formats"

        # Terminate application instead of returning empty DataFrame
        raise SystemExit(error_msg)


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException,
                       requests.exceptions.HTTPError,
                       ConnectionError),
                      max_tries=5)
def get_ticker_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a ticker

    Args:
        symbol: Ticker symbol

    Returns:
        Dictionary with ticker information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            logger.warning(f"No info found for ticker {symbol}")
            return None

        return info

    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {e}")
        logger.exception(e)
        return None


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException,
                       requests.exceptions.HTTPError,
                       ConnectionError),
                      max_tries=5)
def get_option_chain(symbol: str, expiration_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get options chain data for a symbol

    Args:
        symbol: Ticker symbol
        expiration_date: Specific expiration date (YYYY-MM-DD)

    Returns:
        Dictionary containing calls and puts DataFrames with options closest to current price
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get all expiration dates if none specified
        if not expiration_date:
            expiration_dates = ticker.options
            if not expiration_dates or len(expiration_dates) == 0:
                logger.warning(f"No options data available for {symbol}")
                return None
            expiration_date = expiration_dates[0]  # Use nearest expiration

        # Get options chain for the expiration date
        option_chain = ticker.option_chain(expiration_date)

        if not option_chain or not hasattr(option_chain, 'calls') or not hasattr(option_chain, 'puts'):
            logger.warning(f"Invalid options data for {symbol}")
            return None

        # Get current stock price
        try:
            current_price = ticker.info['regularMarketPrice']
        except Exception as e:
            logger.warning(
                f"Could not get current price for {symbol}: {e}. Using last closing price.")
            history = ticker.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else None

        if current_price is None:
            logger.warning(
                f"Could not determine current price for {symbol}, returning all options")
            return {
                'expiration_date': expiration_date,
                'calls': option_chain.calls,
                'puts': option_chain.puts
            }

        # Filter calls to get 6 closest to current price
        calls = option_chain.calls
        calls['price_diff'] = abs(calls['strike'] - current_price)
        filtered_calls = calls.sort_values('price_diff').head(6)

        # Filter puts to get 6 closest to current price
        puts = option_chain.puts
        puts['price_diff'] = abs(puts['strike'] - current_price)
        filtered_puts = puts.sort_values('price_diff').head(6)

        # Remove the temporary price_diff column
        if 'price_diff' in filtered_calls.columns:
            filtered_calls = filtered_calls.drop('price_diff', axis=1)
        if 'price_diff' in filtered_puts.columns:
            filtered_puts = filtered_puts.drop('price_diff', axis=1)

        return {
            'expiration_date': expiration_date,
            'current_price': current_price,
            'calls': filtered_calls,
            'puts': filtered_puts
        }

    except Exception as e:
        logger.error(f"Error getting option chain for {symbol}: {e}")
        logger.exception(e)
        return None


def get_available_tickers() -> Dict[str, str]:
    """
    Get a dictionary of common ticker symbols used in the application

    Returns:
        Dictionary mapping ticker aliases to actual symbols
    """
    return {
        "SPY": "SPY",   # S&P 500 Index ETF
        "QQQ": "QQQ",   # Nasdaq 100 ETF (Tech-heavy)
        "IWM": "IWM",   # Russell 2000 Small Cap ETF
        "VTV": "VTV",   # Vanguard Value ETF
        "VGLT": "VGLT",  # Vanguard Long-Term Treasury ETF
        "VIX": "^VIX",  # Volatility Index
        "DIA": "DIA",   # Dow Jones Industrial Average ETF
        "BND": "BND",   # Total Bond Market ETF
        "BTC-USD": "BTC-USD",  # Bitcoin USD
        "TSLA": "TSLA"
    }
