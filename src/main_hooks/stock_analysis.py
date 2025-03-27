#!/usr/bin/env python3
"""
Stock Analysis Module (src/main_hooks/stock_analysis.py)
------------------------------------------------------
Analyzes individual stocks based on technical, fundamental, and market conditions.

Functions:
  - analyze_stocks - Processes multiple stocks in parallel with AI analysis

Dependencies:
  - src.main_utilities.analysis_parser - For parsing AI analysis results
  - src.main_utilities.data_processor - For formatting stock data
  - YFinance client - For stock price and options data
  - Gemini client - For AI analysis
  - concurrent.futures - For parallel processing

Used by:
  - main.py for analyzing individual stocks in the watchlist
"""

import logging
import concurrent.futures
from datetime import datetime, timedelta

# Import utilities
from src.main_utilities.analysis_parser import parse_stock_analysis
from src.main_utilities.data_processor import format_stock_data_for_analysis

logger = logging.getLogger(__name__)


def analyze_stocks(yfinance_client, gemini_client, market_analysis, symbols, stock_analysis_prompt_hook):
    """
    Analyze individual stocks based on market conditions

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - market_analysis: Market analysis results
    - symbols: List of stock symbols to analyze
    - stock_analysis_prompt_hook: Function to generate stock analysis prompt

    Returns:
    - Dictionary of stock analyses by symbol
    """
    logger.info(
        f"Analyzing {len(symbols)} stocks based on market trend: {market_analysis.get('trend', 'neutral')}")

    try:
        # Analyze all stocks in the watchlist
        symbols_to_analyze = symbols
        logger.info(
            f"Will analyze all {len(symbols_to_analyze)} stocks from watchlist")

        # Get all stock data in parallel
        stock_data = {}
        options_data = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # First get stock data
            future_to_symbol = {executor.submit(
                yfinance_client.get_stock_analysis, symbol): symbol for symbol in symbols_to_analyze}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stock_data[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    stock_data[symbol] = None

            # Then get options data
            future_to_symbol = {executor.submit(
                yfinance_client.get_options_chain, symbol): symbol for symbol in symbols_to_analyze}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    options_data[symbol] = future.result()
                except Exception as e:
                    logger.error(
                        f"Error getting options data for {symbol}: {e}")
                    options_data[symbol] = None

        # Analyze each stock
        stock_analyses = {}

        for symbol in symbols_to_analyze:
            try:
                # Skip if no data
                if not stock_data.get(symbol):
                    logger.warning(
                        f"No data available for {symbol}, skipping analysis")
                    continue

                # Get historical data - first try from stock_data, then directly fetch if needed
                historical_data = stock_data[symbol].get("historical", None)

                # If historical data is not available, try to fetch it directly
                if historical_data is None:
                    logger.info(
                        f"Historical data not found in stock_data for {symbol}, fetching directly")
                    try:
                        end_date = datetime.now()
                        # Get 100 days of data instead of 30 to ensure sufficient data for technical indicators
                        start_date = end_date - timedelta(days=100)
                        historical_data = yfinance_client.get_historical_data(
                            symbol,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            interval="1d"
                        )

                        if historical_data is None or historical_data.empty:
                            logger.warning(
                                f"Failed to fetch historical data for {symbol}")
                    except Exception as e:
                        logger.error(
                            f"Error fetching historical data for {symbol}: {e}")

                # Format stock data for analysis
                formatted_data = format_stock_data_for_analysis(
                    historical_data,
                    symbol,
                    datetime.now().strftime("%Y-%m-%d")
                )

                # Add market trend context
                analysis_context = {
                    "stock_data": formatted_data,
                    "market_trend": market_analysis.get("trend", "neutral"),
                    "market_trend_score": market_analysis.get("market_trend_score", 0),
                    "risk_adjustment": market_analysis.get("risk_adjustment", "standard"),
                    "options_data": options_data.get(symbol, None)
                }

                # Generate prompt for stock analysis
                prompt = stock_analysis_prompt_hook(analysis_context)

                # Get analysis from Gemini
                analysis_text = gemini_client.generate_text(
                    prompt, temperature=0.3)

                # Parse the analysis
                analysis = parse_stock_analysis(analysis_text)

                # Add raw data and full analysis text
                analysis["raw_data"] = formatted_data
                analysis["full_analysis"] = analysis_text

                # Process options data if available
                if options_data.get(symbol):
                    from src.main_utilities.data_processor import process_options_data
                    analysis["options_summary"] = process_options_data(
                        options_data[symbol])

                # Add market alignment assessment - compare stock trend with market trend
                if analysis["trend"] == market_analysis.get("trend", "neutral"):
                    analysis["market_alignment"] = "aligned"
                else:
                    analysis["market_alignment"] = "divergent"

                # Add to results
                stock_analyses[symbol] = analysis
                logger.info(
                    f"Completed analysis for {symbol}: {analysis['trend']} with score {analysis['total_score']}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                # Add minimal entry for failed analysis
                stock_analyses[symbol] = {
                    "error": str(e),
                    "trend": "neutral",
                    "technical_score": 0,
                    "fundamental_score": 0,
                    "sentiment_score": 0,
                    "total_score": 0
                }

        return stock_analyses

    except Exception as e:
        logger.error(f"Error in stock analysis: {e}")
        return {}
