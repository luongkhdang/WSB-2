#!/usr/bin/env python3
"""
File Operations Utility (src/main_utilities/file_operations.py)
-------------------------------------------------------------
Handles file I/O operations for watchlist, pretraining data, and market context.

Functions:
  - get_watchlist_symbols - Retrieves ticker symbols from watchlist file
  - update_watchlist - Updates watchlist based on screener data
  - save_pretraining_results - Saves pretraining data to disk
  - load_pretraining_results - Loads pretraining data from disk
  - get_historical_market_context - Retrieves historical market context

Dependencies:
  - pandas - For CSV handling in update_watchlist
  - pathlib.Path - For file path operations
  - json - For reading/writing JSON data
  - hashlib - For creating unique file names

Used by:
  - main.py and various main_hooks modules to read/write data files
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def get_watchlist_symbols(watchlist_file):
    """
    Get list of ticker symbols from watchlist file.

    Args:
        watchlist_file: Path to the watchlist file

    Returns:
        List of ticker symbols
    """
    try:
        symbols = []

        # Check if file exists first
        if not os.path.exists(watchlist_file):
            logger.warning(f"Watchlist file not found at {watchlist_file}")
            return ['SPY']  # Return default symbol

        with open(watchlist_file, 'r') as f:
            for line in f:
                # Skip any comment lines or empty lines
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue

                # Skip obvious file names or malformed ticker symbols
                if '.' in line and (line.endswith('.CSV') or line.endswith('.csv')):
                    logger.warning(
                        f"Skipping file name found in watchlist: {line}")
                    continue

                # Skip any tokens that are likely not valid tickers
                if len(line) > 10 or '-' in line or line.isupper() == False:
                    if not line.endswith('-USD'):  # Allow crypto pairs like BTC-USD
                        logger.warning(
                            f"Skipping likely invalid ticker: {line}")
                        continue

                symbols.append(line)

        # Add SPY as a default if the list is empty
        if not symbols:
            logger.warning(
                "No valid symbols found in watchlist, using default")
            symbols = ['SPY']

        return symbols
    except Exception as e:
        logger.error(f"Error reading watchlist file: {e}")
        return ['SPY']  # Return default symbol on error


def update_watchlist(screener_file, watchlist_file, key_indices, constant_players):
    """Update the watchlist based on the options screener data

    Parameters:
    - screener_file: Path to options screener file
    - watchlist_file: Path to watchlist file to update
    - key_indices: List of key market indices to always include
    - constant_players: List of constant big players to always include

    Returns:
    - Boolean indicating success
    """
    logger.info("Updating watchlist from options screener data...")

    try:
        # Read options screener data
        if not Path(screener_file).exists():
            logger.error(f"Screener file not found: {screener_file}")
            return False

        # Load the CSV file
        df = pd.read_csv(screener_file)

        # Filter out non-ticker symbols
        valid_symbols = []
        for sym in df['Symbol'].unique():
            # Skip obvious file names or non-ticker-like strings
            if isinstance(sym, str):
                # Skip file names or malformed symbols
                if '.' in sym and (sym.endswith('.CSV') or sym.endswith('.csv')):
                    logger.warning(
                        f"Skipping file name in screener data: {sym}")
                    continue

                # Skip tokens that are likely not valid tickers
                if len(sym) > 10 or sym.upper() != sym:
                    if not sym.endswith('-USD'):  # Allow crypto pairs like BTC-USD
                        logger.warning(
                            f"Skipping likely invalid ticker: {sym}")
                        continue

                valid_symbols.append(sym)

        symbols = valid_symbols

        # Sort by volume if available
        if 'Volume' in df.columns:
            # Create a new dataframe with only valid symbols
            valid_df = df[df['Symbol'].isin(valid_symbols)]
            # Group by symbol and sum the volume
            volume_by_symbol = valid_df.groupby(
                'Symbol')['Volume'].sum().reset_index()
            # Sort by volume in descending order
            volume_by_symbol = volume_by_symbol.sort_values(
                'Volume', ascending=False)
            # Get the top 15 symbols
            symbols = list(volume_by_symbol['Symbol'])[:15]

        # Ensure key indices are included
        for index in key_indices:
            if index not in symbols:
                symbols.append(index)

        # Add constant big players to the watchlist
        for player in constant_players:
            if player not in symbols:
                symbols.append(player)

        # Update the watchlist file
        with open(watchlist_file, 'w') as f:
            f.write("# WSB Trading - Credit Spread Watchlist\n")
            f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("# Stocks and key indices for analysis\n\n")

            f.write("# KEY INDICES (Always included)\n")

            # Write key indices first
            for index in key_indices:
                f.write(f"{index}\n")

            f.write("\n# CONSTANT BIG PLAYERS (Always included)\n")

            # Write constant big players
            for player in constant_players:
                if player not in key_indices:  # Avoid duplicates
                    f.write(f"{player}\n")

            f.write("\n# STOCKS FOR CREDIT SPREADS\n\n")

            # Write other symbols
            for symbol in symbols:
                if symbol not in key_indices and symbol not in constant_players:  # Avoid duplicates
                    f.write(f"{symbol}\n")

        logger.info(
            f"Watchlist updated with {len(symbols)} symbols including key indices and constant big players")
        return True

    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        return False


def save_pretraining_results(pretraining_dir, ticker, results, context):
    """
    Save pretraining results to disk

    Parameters:
    - pretraining_dir: Base directory for pretraining data
    - ticker: Stock symbol
    - results: Pretraining results
    - context: Context information

    Returns:
    - Path to saved file
    """
    logger.info(f"Saving pretraining results for {ticker}")

    try:
        # Create ticker directory if needed
        ticker_dir = Path(pretraining_dir) / ticker
        ticker_dir.mkdir(exist_ok=True)

        # Create unique filename based on date and hash
        date_str = datetime.now().strftime("%Y%m%d")
        hash_str = hashlib.md5(str(results).encode()).hexdigest()[:8]
        filename = f"{ticker}_pretraining_{date_str}_{hash_str}.json"

        # Save results to file
        result_file = ticker_dir / filename
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save latest context file
        context_file = ticker_dir / "latest_context.txt"
        with open(context_file, 'w') as f:
            f.write(json.dumps(context, indent=2))

        logger.info(f"Saved pretraining results to {result_file}")
        return str(result_file)

    except Exception as e:
        logger.error(f"Error saving pretraining results: {e}")
        return None


def load_pretraining_results(pretraining_dir, ticker):
    """
    Load the latest pretraining results for a ticker

    Parameters:
    - pretraining_dir: Base directory for pretraining data
    - ticker: Stock symbol

    Returns:
    - Dictionary with pretraining results and context
    """
    logger.info(f"Loading pretraining results for {ticker}")

    try:
        # Get ticker directory
        ticker_dir = Path(pretraining_dir) / ticker

        if not ticker_dir.exists():
            logger.warning(f"No pretraining directory found for {ticker}")
            return None

        # Check for latest context file
        context_file = ticker_dir / "latest_context.txt"

        if not context_file.exists():
            logger.warning(f"No context file found for {ticker}")
            return None

        # Load context
        with open(context_file, 'r') as f:
            context = json.load(f)

        # Find the latest results file
        result_files = list(ticker_dir.glob(f"{ticker}_pretraining_*.json"))

        if not result_files:
            logger.warning(f"No result files found for {ticker}")
            return None

        # Sort by modified time (most recent first)
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = result_files[0]

        # Load results
        with open(latest_file, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded pretraining results from {latest_file}")

        return {
            "results": results,
            "context": context,
            "file_path": str(latest_file),
            "timestamp": datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Error loading pretraining results: {e}")
        return None


def get_historical_market_context(market_data_dir, date_str, yfinance_client=None):
    """
    Get historical market context for a specific date

    Parameters:
    - market_data_dir: Directory containing historical market data
    - date_str: Date string in YYYY-MM-DD format
    - yfinance_client: Optional YFinance client to fetch data if not cached

    Returns:
    - Dictionary with market context
    """
    logger.info(f"Getting historical market context for {date_str}")

    try:
        # Format date for filename
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year_month = date_obj.strftime("%Y-%m")

        # Look for month-specific file first
        month_file = Path(market_data_dir) / \
            f"market_context_{year_month}.json"

        if month_file.exists():
            with open(month_file, 'r') as f:
                month_data = json.load(f)

            # Check if date exists in month data
            if date_str in month_data:
                logger.info(
                    f"Found market context for {date_str} in month file")
                return month_data[date_str]

        # If not found, try the specific date file
        date_file = Path(market_data_dir) / f"market_context_{date_str}.json"

        if date_file.exists():
            with open(date_file, 'r') as f:
                logger.info(f"Found market context in date-specific file")
                return json.load(f)

        # If we have a yfinance client and still haven't found data, fetch it
        if yfinance_client:
            logger.info(
                f"No cached market context found for {date_str}, fetching from yfinance")

            # Get SPY and VIX data
            start_date = (date_obj - timedelta(days=5)).strftime("%Y-%m-%d")
            end_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

            spy_data = yfinance_client.get_historical_data(
                "SPY", start=start_date, end=end_date)
            vix_data = yfinance_client.get_historical_data(
                "^VIX", start=start_date, end=end_date)

            # Extract date-specific data
            target_date_data = None
            try:
                if not spy_data.empty:
                    # Find closest date to target date
                    closest_date = min(spy_data.index, key=lambda x: abs(
                        (x.to_pydatetime().date() - date_obj.date()).days))
                    logger.info(
                        f"Found closest date to {date_str}: {closest_date.strftime('%Y-%m-%d')}")

                    # Get SPY trend
                    spy_close = spy_data.loc[closest_date]['Close']
                    # Ensure spy_close is a scalar value, not a Series
                    if isinstance(spy_close, pd.Series):
                        spy_close = spy_close.iloc[0]

                    # Calculate change from previous day if possible
                    spy_change = 0.0
                    if len(spy_data) >= 2:
                        # Find previous trading day
                        prev_dates = spy_data.index[spy_data.index <
                                                    closest_date]
                        if len(prev_dates) > 0:
                            prev_date = prev_dates[-1]
                            spy_prev_close = spy_data.loc[prev_date]['Close']
                            # Ensure spy_prev_close is a scalar value
                            if isinstance(spy_prev_close, pd.Series):
                                spy_prev_close = spy_prev_close.iloc[0]
                            spy_change = (
                                (spy_close - spy_prev_close) / spy_prev_close) * 100

                    # Determine trend
                    # Ensure spy_change is a scalar value before comparing
                    if isinstance(spy_change, pd.Series):
                        spy_change = spy_change.iloc[0]

                    if spy_change > 0.5:
                        spy_trend = "bullish"
                    elif spy_change < -0.5:
                        spy_trend = "bearish"
                    else:
                        spy_trend = "neutral"

                    # Get VIX level
                    vix_level = None
                    if not vix_data.empty:
                        # Find closest VIX date
                        closest_vix_date = min(vix_data.index, key=lambda x: abs(
                            (x.to_pydatetime().date() - date_obj.date()).days))
                        vix_level = vix_data.loc[closest_vix_date]['Close']

                    if vix_level is None:
                        logger.warning(
                            f"No VIX data found for {date_str}, using default")
                        vix_level = 20.0  # Default moderate volatility

                    # Create market context
                    market_context = {
                        "date": date_str,
                        "spy_trend": spy_trend,
                        "market_trend": spy_trend,  # Use SPY as proxy for market
                        # Scale -10 to +10
                        "market_trend_score": 50 + (spy_change * 5),
                        "spy_change_percent": round(float(spy_change), 2),
                        "vix_level": round(float(vix_level), 2),
                        "vix_assessment": f"VIX is at {round(float(vix_level), 1)} indicating " +
                        ("high volatility" if vix_level > 25 else
                         "low volatility" if vix_level < 15 else "moderate volatility"),
                        "risk_adjustment": "reduced" if vix_level > 25 else "standard",
                        "market_indices": {
                            "SPY": {"trend": spy_trend, "change": round(float(spy_change), 2)},
                        },
                        "data_source": "yfinance_realtime",
                        "closest_date_used": closest_date.strftime('%Y-%m-%d')
                    }

                    # Save this context
                    try:
                        # Ensure the market data directory exists
                        Path(market_data_dir).mkdir(
                            exist_ok=True, parents=True)

                        with open(date_file, 'w') as f:
                            json.dump(market_context, f, indent=2)
                        logger.info(
                            f"Saved fetched market context for {date_str}")

                        return market_context
                    except Exception as e:
                        logger.warning(
                            f"Could not save fetched market context: {e}")
                        return market_context

            except Exception as e:
                logger.error(f"Error processing fetched market data: {e}")
                logger.exception(e)

        # If we get here, we couldn't find or fetch market context
        logger.warning(
            f"No market context found for {date_str}, generating default context")

        # Create default market context - but log it as a default
        default_context = {
            "date": date_str,
            "spy_trend": "neutral",
            "market_trend": "neutral",
            "market_trend_score": 50,
            "vix_assessment": "VIX is at moderate levels indicating balanced market sentiment",
            "risk_adjustment": "standard",
            "sector_rotation": "No specific sector rotation data available for this date",
            "market_indices": {
                "SPY": {"trend": "neutral", "change": 0.0},
                "QQQ": {"trend": "neutral", "change": 0.0},
                "IWM": {"trend": "neutral", "change": 0.0}
            },
            "data_source": "default",
            "note": "This is a generated default context as no historical data was found or could be fetched"
        }

        # Ensure the market data directory exists
        Path(market_data_dir).mkdir(exist_ok=True, parents=True)

        # Save the default context for future use
        try:
            with open(date_file, 'w') as f:
                json.dump(default_context, f, indent=2)
            logger.info(f"Saved default market context for {date_str}")
        except Exception as e:
            logger.warning(f"Could not save default market context: {e}")

        return default_context

    except Exception as e:
        logger.error(f"Error getting historical market context: {e}")
        logger.exception(e)
        return {
            "date": date_str,
            "spy_trend": "neutral",
            "market_trend": "unknown",
            "error": str(e),
            "data_source": "error_fallback"
        }
