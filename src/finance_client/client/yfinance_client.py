#!/usr/bin/env python3
"""
YFinance Client Module (src/finance_client/client/yfinance_client.py)
---------------------------------------------------------------------
Client for retrieving financial market data via YFinance API with caching.

Class:
  - YFinanceClient - Wrapper around YFinance with specialized hooks and caching

Methods:
  - get_historical_data - Gets OHLCV data for a symbol
  - get_market_data - Gets data for major market indices
  - get_volatility_filter - Gets VIX data for risk assessment
  - get_options_chain - Gets options data for a symbol
  - get_stock_analysis - Gets enhanced stock analysis with technical indicators
  - plus various other specialized data retrieval methods

Dependencies:
  - yfinance - For Yahoo Finance API access
  - pandas - For data manipulation
  - matplotlib - For visualization (if used)
  - src.finance_client.utilities.* - For API and caching utilities
  - src.finance_client.hooks.* - For specialized data processors

Used by:
  - main.py for all financial data retrieval
"""

from src.finance_client.utilities.yfinance_api import (
    download_ticker_data,
    get_historical_data,
    get_ticker_info,
    get_option_chain
)
from src.finance_client.utilities.data_cache import DataCache
from src.finance_client.hooks.stock_analysis_hook import StockAnalysisHook
from src.finance_client.hooks.options_analysis_hook import OptionsAnalysisHook
from src.finance_client.hooks.market_data_hook import MarketDataHook
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import copy
import logging
import numpy as np
import pytz
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import requests
from functools import lru_cache
from pathlib import Path
import pickle
import hashlib
import backoff

# Set up logging for yfinance and urllib3 (used by yfinance)
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Set up logging for our client
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yfinance_client')

# Import hooks and utilities


class YFinanceClient:
    """
    YFinance client with various finance data hooks for market analysis
    """

    def __init__(self, use_cache=True, cache_dir="./data_cache"):
        logger.info("Initializing YFinanceClient")
        try:
            # Set up caching
            self.use_cache = use_cache
            self.cache_dir = cache_dir

            if use_cache:
                self.cache = DataCache(cache_dir)

            # Initialize hooks
            self.market_data_hook = MarketDataHook(
                use_cache=use_cache, cache_dir=cache_dir)
            self.options_hook = OptionsAnalysisHook(
                use_cache=use_cache, cache_dir=cache_dir)
            self.stock_hook = StockAnalysisHook(
                use_cache=use_cache, cache_dir=cache_dir)

            # Common ticker symbols - hardcoded list of common stock symbols
            self.tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT',
                            'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

            logger.info("YFinanceClient initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing YFinanceClient: {e}")
            raise

    def _get_ticker_data(self, symbol, period="1d", interval="1h"):
        """
        Get ticker data using the API utility

        Args:
            symbol: Ticker symbol
            period: Time period
            interval: Data interval

        Returns:
            DataFrame with price data
        """
        return download_ticker_data(symbol, period, interval)

    def get_historical_data(self, symbol, start=None, end=None, interval="1d"):
        """
        Get historical data for a symbol between start and end dates

        Args:
            symbol: Ticker symbol
            start: Start date string in format 'YYYY-MM-DD'
            end: End date string in format 'YYYY-MM-DD'
            interval: Data interval

        Returns:
            DataFrame with historical data

        Raises:
            SystemExit: If data is missing or invalid, terminating the application
        """
        # If start date is not provided, default to 60 days before end date
        # (or today if end is not provided) to ensure sufficient data for technical indicators
        if start is None:
            from datetime import datetime, timedelta
            if end is None:
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(end, "%Y-%m-%d")

            start_date = end_date - timedelta(days=60)
            start = start_date.strftime("%Y-%m-%d")

            if end is None:
                end = datetime.now().strftime("%Y-%m-%d")

            logger.info(
                f"Using default 60-day lookback period for {symbol}: {start} to {end}")

        # Fix VIX ticker symbol if needed
        if symbol == "VIX":
            symbol = "^VIX"  # Ensure correct Yahoo Finance symbol for VIX

        # Get data from YFinance API - will raise SystemExit if data is missing or invalid
        data = get_historical_data(symbol, start, end, interval)

        # Special handling for VIX - try alternative tickers if needed
        if data is None or len(data) == 0:
            if symbol == "^VIX":
                logger.warning(
                    f"Attempting to get VIX data using alternative ticker 'VIXY'")
                data = get_historical_data("VIXY", start, end, interval)

                if data is None or len(data) == 0:
                    error_msg = f"CRITICAL ERROR: Failed to retrieve VIX data using both ^VIX and VIXY alternative"
                    logger.critical(error_msg)
                    raise SystemExit(error_msg)
                else:
                    logger.info(
                        f"Successfully retrieved alternative VIX data using VIXY")
            else:
                # This should never happen as get_historical_data should have already raised SystemExit
                error_msg = f"CRITICAL ERROR: No data available for {symbol} and no fallback options"
                logger.critical(error_msg)
                raise SystemExit(error_msg)

        return data

    def get_market_data(self):
        """
        Get comprehensive market data including major indices

        Returns:
            Dictionary with market data for various indices
        """
        return self.market_data_hook.get_market_data()

    def get_volatility_filter(self):
        """
        Get VIX data for market volatility filtering

        Returns:
            Dictionary with VIX analysis
        """
        return self.market_data_hook.get_volatility_filter()

    def get_options_sentiment(self, symbol="SPY", expiration_date=None):
        """
        Get enhanced options sentiment data

        Args:
            symbol: Ticker symbol
            expiration_date: Options expiration date

        Returns:
            Dictionary with options sentiment data
        """
        return self.options_hook.get_options_sentiment(symbol, expiration_date)

    def get_stock_analysis(self, ticker_symbol="TSLA"):
        """
        Get enhanced stock analysis with advanced technical indicators

        Args:
            ticker_symbol: Stock ticker symbol

        Returns:
            Dictionary with stock analysis data
        """
        return self.stock_hook.get_stock_analysis(ticker_symbol)

    def get_options_for_spread(self, symbol, expiration_date=None):
        """
        Get enhanced options data formatted for credit spread analysis

        Args:
            symbol: Ticker symbol
            expiration_date: Options expiration date

        Returns:
            Dictionary with credit spread analysis
        """
        return self.options_hook.get_options_for_spread(symbol, expiration_date)

    def get_complete_analysis(self, ticker_symbol="TSLA"):
        """
        Get comprehensive analysis for trading decision following the Rule Book sequence:
        1. Analyze SPY: Determine general market trend
        2. Analyze Options of SPY: Assess general market direction
        3. Analyze Underlying Stock: Evaluate fundamental data 
        4. Analyze Credit Spreads: Identify profiting opportunities

        This returns a structured dataset for AI analysis rather than performing the analysis itself.

        Args:
            ticker_symbol: Stock ticker symbol

        Returns:
            Dictionary with complete analysis data
        """
        logger.info(
            f"Gathering comprehensive data for {ticker_symbol} analysis")

        # Step a: Get SPY Market Data & VIX for Market Trend Analysis
        market_data = self.get_market_data()
        volatility_filter = self.get_volatility_filter()
        options_sentiment = self.get_options_sentiment()

        # Step b: Get stock data
        stock_data = self.get_stock_analysis(ticker_symbol)

        # Step c: Get options data for spreads
        options_spread_data = self.get_options_for_spread(ticker_symbol)

        # Prepare market context
        market_context = {
            "spy_price": market_data.get("SPY", {}).get("info", {}).get("regularMarketPrice") if market_data else None,
            "spy_ema_data": {
                "50d_avg": market_data.get("SPY", {}).get("info", {}).get("fiftyDayAverage") if market_data else None,
                "200d_avg": market_data.get("SPY", {}).get("info", {}).get("twoHundredDayAverage") if market_data else None
            },
            "vix": {
                "price": volatility_filter.get("price") if volatility_filter else None,
                "stability": volatility_filter.get("stability") if volatility_filter else None,
                "risk_adjustment": volatility_filter.get("risk_adjustment") if volatility_filter else None
            },
            "options_sentiment": {
                "call_put_ratio": options_sentiment.get("call_put_volume_ratio") if options_sentiment else None,
                "iv_skew": options_sentiment.get("iv_skew") if options_sentiment else None
            }
        }

        # Prepare stock analysis data
        stock_analysis_data = None
        if stock_data:
            stock_analysis_data = {
                "price": stock_data.get("info", {}).get("regularMarketPrice"),
                "support": stock_data.get("technical", {}).get("support"),
                "resistance": stock_data.get("technical", {}).get("resistance"),
                "near_support": stock_data.get("technical", {}).get("near_support"),
                "near_resistance": stock_data.get("technical", {}).get("near_resistance"),
                "ema9": stock_data.get("technical", {}).get("ema9"),
                "ema21": stock_data.get("technical", {}).get("ema21"),
                "trend": stock_data.get("technical", {}).get("trend"),
                "atr": stock_data.get("technical", {}).get("atr"),
                "atr_percent": stock_data.get("technical", {}).get("atr_percent"),
                "volatility": stock_data.get("technical", {}).get("volatility")
            }

        # Prepare options data
        options_data = None
        if options_spread_data:
            # Extract just what's needed for the analysis
            bull_put_spreads = options_spread_data.get("bull_put_spreads", [])
            bear_call_spreads = options_spread_data.get(
                "bear_call_spreads", [])

            options_data = {
                "current_price": options_spread_data.get("current_price"),
                "expiration_date": options_spread_data.get("expiration_date"),
                # Just keep top 3
                "bull_put_spreads": bull_put_spreads[:3] if bull_put_spreads else [],
                # Just keep top 3
                "bear_call_spreads": bear_call_spreads[:3] if bear_call_spreads else []
            }

        # Assemble structured dataset for AI analysis
        complete_analysis = {
            "ticker": ticker_symbol,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "market_context": market_context,
            "stock_analysis": stock_analysis_data,
            "options_data": options_data,

            # Keep the full detailed data for reference
            "full_data": {
                "market_data": market_data,
                "volatility_filter": volatility_filter,
                "options_sentiment": options_sentiment,
                "stock_data": stock_data,
                "options_spread_data": options_spread_data
            }
        }

        logger.info(
            f"Comprehensive data gathering complete for {ticker_symbol}")
        return complete_analysis

    def get_option_chain_with_greeks(self, ticker_symbol, weeks=4):
        """
        Get enhanced option chain data with accurate Greeks calculations

        Args:
            ticker_symbol: Stock ticker symbol
            weeks: Number of weeks to look ahead

        Returns:
            Dictionary with option chain data by expiration date
        """
        # This is a complex operation that requires direct interaction with option chains
        # Rather than refactoring it, we'll pass through to a future implementation
        logger.warning(
            "get_option_chain_with_greeks not yet implemented in hooks")
        return {}

    def get_options_chain(self, symbol, expiration_date=None):
        """
        Get options chain data for a symbol

        Args:
            symbol: Ticker symbol
            expiration_date: Specific expiration date

        Returns:
            Dictionary with options chain data (6 calls and 6 puts closest to current price)
        """
        options_data = get_option_chain(symbol, expiration_date)

        if not options_data:
            return None

        result = {
            'expiration_date': options_data.get('expiration_date'),
            'current_price': options_data.get('current_price'),
            'calls': options_data.get('calls').to_dict() if 'calls' in options_data else {},
            'puts': options_data.get('puts').to_dict() if 'puts' in options_data else {}
        }

        return result

    def export_to_excel(self, data, filename="yfinance_analysis.xlsx"):
        """
        Export analysis data to Excel for further processing

        Args:
            data: Analysis data to export
            filename: Output Excel filename

        Returns:
            Status message
        """
        try:
            with pd.ExcelWriter(filename) as writer:
                # Create summary sheet
                summary_data = data.get("summary", {})
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name="Summary")

                # Create SPY sheet
                if data.get("market_data") and "SPY" in data["market_data"] and data["market_data"]["SPY"].get("info"):
                    spy_df = pd.DataFrame([data["market_data"]["SPY"]["info"]])
                    spy_df.to_excel(writer, sheet_name="SPY")

                # Create VIX sheet
                if data.get("volatility_filter"):
                    vix_df = pd.DataFrame([data["volatility_filter"]])
                    vix_df.to_excel(writer, sheet_name="VIX")

                # Create stock sheet
                if data.get("stock_analysis") and data["stock_analysis"].get("info"):
                    stock_df = pd.DataFrame([data["stock_analysis"]["info"]])
                    stock_tech_df = pd.DataFrame(
                        [data["stock_analysis"]["technical"]])
                    stock_df.to_excel(writer, sheet_name="Stock")
                    stock_tech_df.to_excel(
                        writer, sheet_name="Stock_Technical")

                # Create Options sheet if available
                if data.get("options_spread_data"):
                    options_df = pd.DataFrame([{
                        "ticker": data["options_spread_data"]["ticker"],
                        "expiration": data["options_spread_data"]["expiration_date"]
                    }])
                    options_df.to_excel(writer, sheet_name="Options")

            logger.info(f"Data exported to {filename}")
            return f"Data exported to {filename}"
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return None

    def export_to_json(self, data, filename="yfinance_analysis.json"):
        """
        Export analysis data to JSON

        Args:
            data: Analysis data to export
            filename: Output JSON filename

        Returns:
            Status message
        """
        try:
            # Deep copy to avoid modifying original data
            data_copy = copy.deepcopy(data)

            # Convert pandas dataframes and timestamps for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                if isinstance(obj, pd.Timestamp):
                    return str(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return obj

            data_json = convert_for_json(data_copy)

            with open(filename, 'w') as f:
                json.dump(data_json, f, indent=4, default=str)

            logger.info(f"Data exported to {filename}")
            return f"Data exported to {filename}"
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return None


def run_sample_tests():
    """
    Run sample tests for YFinanceClient
    """
    client = YFinanceClient(use_cache=True)

    print("Testing market data...")
    market_data = client.get_market_data()

    print("Testing volatility filter...")
    vix_data = client.get_volatility_filter()

    print("Testing stock analysis...")
    stock_data = client.get_stock_analysis("AAPL")

    print("Testing options sentiment...")
    options_sentiment = client.get_options_sentiment("SPY")

    print("Testing complete analysis...")
    complete_analysis = client.get_complete_analysis("MSFT")

    return {
        "success": True,
        "market_data": market_data is not None,
        "vix_data": vix_data is not None,
        "stock_data": stock_data is not None,
        "options_sentiment": options_sentiment is not None,
        "complete_analysis": complete_analysis is not None
    }


if __name__ == "__main__":
    print("Testing improved YFinance client...")
    test_results = run_sample_tests()
    print("\nTest complete!")
