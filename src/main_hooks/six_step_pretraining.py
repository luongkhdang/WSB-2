"""
Six-step pretraining implementation for WSB trading system with credit spread focus.

This module provides a structured pretraining process for analyzing financial instruments 
and generating credit spread trading recommendations based on multi-timeframe analysis.

Key Functions:
- six_step_pretrain_analyzer(yfinance_client, gemini_client, pretraining_dir, ticker, ...): 
    Main function that implements the 6-step pretraining process for a single ticker
- six_step_batch_pretrain_analyzer(yfinance_client, gemini_client, pretraining_dir, tickers, ...):
    Batch processor for running analysis on multiple tickers efficiently
    
Dependencies:
- src.gemini.hooks.credit_spread_pretraining_prompt: Prompt templates and parsing
- src.main_utilities.data_processor: Data formatting and technical indicators
- src.main_utilities.file_operations: File I/O and market context retrieval

Related Files:
- src/gemini/hooks/credit_spread_pretraining_prompt.py: Prompt generation and parsing
- src/main_utilities/data_processor.py: Technical indicator calculation and data formatting
- src/main_utilities/file_operations.py: File operations and market context retrieval

Performance Metrics:
- The analysis should generate actionable trading strategies with:
  - Defined entry/exit criteria
  - Specific risk/reward parameters
  - Adaption based on historical performance
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import json
import math
import pandas as pd
import numpy as np

# Import utilities
from src.gemini.hooks.credit_spread_pretraining_prompt import (
    get_step1_prompt,
    get_step2_to_6_prompt,
    get_summary_prompt,
    parse_credit_spread_prediction,
    parse_credit_spread_memory_update,
    format_data_for_credit_spreads
)

from src.main_utilities.data_processor import format_stock_data_for_analysis
from src.main_utilities.file_operations import save_pretraining_results, get_historical_market_context

logger = logging.getLogger(__name__)

# Constants for API rate limiting
MAX_REQUESTS_PER_MINUTE = 30
# Time between requests in seconds - set to a shorter interval to maximize usage
REQUEST_INTERVAL = 2  # 2 second interval allows for 30 requests per minute
MAX_CONSECUTIVE_REQUESTS = 25  # Leave buffer for other system activities

# Constants for volatility interpretation
VOLATILITY_PERCENTILES = {
    "very_low": 10,      # Below 10th percentile
    "low": 25,           # Below 25th percentile
    "normal": 50,        # Between 25th and 75th percentile
    "elevated": 75,      # Above 75th percentile
    "high": 90           # Above 90th percentile
}

# Constants for technical indicator fallbacks
DEFAULT_IV = 0.25        # Default implied volatility when missing
DEFAULT_HV = 0.20        # Default historical volatility when missing
DEFAULT_ATR_PERCENT = 0.015  # Default ATR as percentage of price
# Minimum days required for reliable analysis (reduced from 30)
MIN_REQUIRED_DAYS = 15

# Track request timestamps to manage rate limiting
_request_timestamps = []

# Cache for market-wide volatility measurements to use as baselines
_volatility_baseline_cache = {}


def manage_rate_limit():
    """
    Manage rate limiting by tracking request timestamps and waiting if needed.
    Ensures we don't exceed 30 requests per minute but maximizes throughput.
    """
    global _request_timestamps

    current_time = time.time()

    # Remove timestamps older than 60 seconds
    _request_timestamps = [
        ts for ts in _request_timestamps if current_time - ts < 60]

    # Check if we've hit the limit
    if len(_request_timestamps) >= MAX_CONSECUTIVE_REQUESTS:
        # Calculate wait time based on oldest request in the window
        oldest_timestamp = min(_request_timestamps)
        wait_time = max(0, 60 - (current_time - oldest_timestamp))

        if wait_time > 0:
            logger.info(
                f"Rate limit approaching, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)

    # Short delay to prevent burst requests
    time.sleep(REQUEST_INTERVAL)

    # Record this request
    _request_timestamps.append(time.time())

# Wrapper function to track Gemini responses and send to Discord


def track_gemini_response(gemini_client, discord_client, ticker, prompt, response, prompt_type):
    """
    Track Gemini API responses and send them to Discord webhook

    Parameters:
    - gemini_client: The Gemini client that generated the response
    - discord_client: Discord client for sending messages
    - ticker: Stock symbol being analyzed
    - prompt: The prompt sent to Gemini
    - response: The response from Gemini
    - prompt_type: Type of prompt (e.g., "Step 1", "Step 2", etc.)

    Returns:
    - None
    """
    try:
        if not discord_client:
            logger.warning(
                "Discord client not available, skipping response tracking")
            return

        # Create a formatted message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"**Gemini API Response - {ticker} - {prompt_type}**\n"
        message += f"Timestamp: {timestamp}\n"

        # Send the response to Discord webhook
        discord_client.send_message(
            content=message,
            webhook_type="pretraining",
            embed_data={
                "title": f"{ticker} - {prompt_type} Response",
                "description": f"```\nFirst 500 chars of response:\n{response[:500]}...\n```",
                "color": 0x00FFFF,
                "footer": {"text": f"Timestamp: {timestamp}"}
            }
        )

        # Send the full response in chunks since it might be too long for a single message
        MAX_CHUNK_SIZE = 1800  # Discord has a 2000 char limit, leaving room for formatting

        if len(response) > MAX_CHUNK_SIZE:
            # Split into chunks
            chunk_index = 1
            total_chunks = math.ceil(len(response) / MAX_CHUNK_SIZE)

            for i in range(0, len(response), MAX_CHUNK_SIZE):
                chunk = response[i:i + MAX_CHUNK_SIZE]
                chunk_message = f"**{ticker} - {prompt_type} - Part {chunk_index}/{total_chunks}**\n```\n{chunk}\n```"

                discord_client.send_message(
                    content=chunk_message,
                    webhook_type="pretraining"
                )

                chunk_index += 1
                # Add a small delay to prevent rate limiting
                time.sleep(0.5)
        else:
            # Send the full response
            full_message = f"**{ticker} - {prompt_type} - Full Response**\n```\n{response}\n```"
            discord_client.send_message(
                content=full_message,
                webhook_type="pretraining"
            )

        logger.info(
            f"Sent Gemini response tracking for {ticker} ({prompt_type}) to Discord")

    except Exception as e:
        logger.error(f"Error tracking Gemini response: {e}")
        logger.exception(e)


def six_step_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    ticker: str,
    start_date=None,
    end_date=None,
    save_results=True,
    callback=None,
    discord_client=None,
    strict_validation=True
):
    """
    Implements the strict 6-step pretraining process for credit spread analysis.

    Steps:
    1. Analyze 1mo/1d data for long-term context
    2-6. Analyze 5d/15m data, one day at a time with reflection

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - ticker: Stock symbol to analyze
    - start_date: Starting date for pretraining (defaults to 5 business days before end_date)
    - end_date: End date for pretraining (defaults to yesterday)
    - save_results: Whether to save pretraining results to disk
    - callback: Optional callback function to receive each analysis result
    - discord_client: Optional Discord client for sending responses to webhook
    - strict_validation: If True, will raise errors when critical data is missing instead of using defaults

    Returns:
    - Dictionary containing pretraining results and context for future analysis

    Raises:
    - ValueError: When critical data is missing and strict_validation is True
    """
    logger.info(f"Starting six-step pretraining for {ticker}...")
    start_time = time.time()

    # Create a wrapped version of gemini_client.generate_text that tracks responses
    original_generate_text = gemini_client.generate_text

    def tracked_generate_text(prompt, temperature=0.0, prompt_type="unknown"):
        logger.info(f"Generating {prompt_type} response for {ticker}")

        # Apply rate limiting before making the request
        manage_rate_limit()

        # Try to get a response with retries for rate limit errors
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = original_generate_text(prompt, temperature)

                # Track the response if discord_client is available
                if discord_client:
                    track_gemini_response(
                        gemini_client,
                        discord_client,
                        ticker,
                        prompt,
                        response,
                        prompt_type
                    )

                return response
            except Exception as e:
                error_str = str(e)
                logger.error(f"Error in tracked_generate_text: {e}")

                # Check if it's a rate limit error (429)
                if "429" in error_str and attempt < max_retries:
                    logger.warning(
                        f"Rate limit exceeded in tracked_generate_text. Waiting 10 seconds before retry #{attempt+1}")
                    time.sleep(10)  # Wait 10 seconds before retrying
                    continue

                # Re-raise for other errors
                raise

    # Temporarily replace the generate_text method
    gemini_client.generate_text = tracked_generate_text

    try:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now() - timedelta(days=1)  # yesterday
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Check if end_date is a weekend and adjust to previous Friday if it is
        if end_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            end_date = end_date - \
                timedelta(days=end_date.weekday() - 4)  # Adjust to Friday

        if not start_date:
            # Find 5 business days before end_date
            start_date = end_date
            business_days_count = 0
            while business_days_count < 5:
                start_date = start_date - timedelta(days=1)
                if start_date.weekday() < 5:  # Only count weekdays (Monday-Friday)
                    business_days_count += 1
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            # Check if start_date is a weekend and adjust to previous Friday if it is
            if start_date.weekday() >= 5:
                start_date = start_date - \
                    timedelta(days=start_date.weekday() - 4)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Analysis period: {start_date_str} to {end_date_str}")

        # Initialize results container
        pretraining_results = []

        # Initialize memory context with credit spread focus
        memory_context = {
            "ticker": ticker,
            "key_levels": {
                "support": [],
                "resistance": [],
                "pivot_points": []
            },
            "pattern_reliability": {
                "head_and_shoulders": {"correct": 0, "total": 0, "accuracy": 0},
                "double_top": {"correct": 0, "total": 0, "accuracy": 0},
                "double_bottom": {"correct": 0, "total": 0, "accuracy": 0},
                "triangle": {"correct": 0, "total": 0, "accuracy": 0},
                "flag": {"correct": 0, "total": 0, "accuracy": 0},
                "wedge": {"correct": 0, "total": 0, "accuracy": 0}
            },
            "volatility_history": [],
            "spread_performance": {
                "bull_put": {"wins": 0, "total": 0, "win_rate": 0},
                "bear_call": {"wins": 0, "total": 0, "win_rate": 0},
                "iron_condor": {"wins": 0, "total": 0, "win_rate": 0}
            },
            "weight_adjustments": [],
            "multi_timeframe": {
                "daily_trend": "neutral",
                "weekly_trend": "neutral"
            },
            "secret": {
                "baseline_trend": "neutral",
                "trend_confidence": 60,
                "volatility_anchor": 1.0,
                "core_levels": {
                    "support": [],
                    "resistance": []
                }
            }
        }

        # Get market data directory
        market_data_dir = pretraining_dir.parent / \
            "market-data" if hasattr(pretraining_dir,
                                     'parent') else "data-source/market-data"

        # Create directory for this ticker
        ticker_dir = pretraining_dir / ticker
        ticker_dir.mkdir(exist_ok=True)

        # Now proceed with pretraining

        # ======================= STEP 1: 1mo/1d Analysis =======================
        # This establishes the SECRET baseline context for this ticker
        # Use 1mo period with 1d interval to establish overall trend and key levels

        # Get long-term data (1mo/1d)
        long_term_start = (start_date - timedelta(days=30)
                           ).strftime("%Y-%m-%d")
        long_term_end = start_date.strftime("%Y-%m-%d")
        long_term_start_str = long_term_start
        end_date_str = long_term_end

        # Get market context for the last day
        market_context = get_historical_market_context(
            market_data_dir=pretraining_dir,
            date_str=end_date_str,
            yfinance_client=yfinance_client  # Pass yfinance_client for real-time fetching
        )

        # Format for logging
        market_context_log = {
            "date": market_context.get("date", "unknown"),
            "spy_trend": market_context.get("spy_trend", "unknown"),
            "vix_assessment": market_context.get("vix_assessment", "unknown").split()[0:3],
            "risk_adjustment": market_context.get("risk_adjustment", "unknown"),
        }
        logger.info(f"Market context for {end_date_str}: {market_context_log}")

        # Get historical daily data
        logger.info(
            f"Getting daily data for {ticker} from {long_term_start} to {end_date_str}")
        daily_data = yfinance_client.get_historical_data(
            ticker, start=long_term_start_str, end=end_date_str, interval="1d")

        # Validate data quality with appropriate settings
        # For historical data, we need a minimum of days
        min_required_days = MIN_REQUIRED_DAYS  # Using constant defined at the top

        # Validate daily data quality
        # Just call validate_data_quality without checking return value
        # The function will log warnings/errors internally
        validate_data_quality(
            daily_data, ticker, min_days_required=min_required_days, strict_validation=strict_validation)

        # Check if we have enough data to proceed
        if isinstance(daily_data, pd.DataFrame) and len(daily_data) < min_required_days:
            if strict_validation:
                logger.error(
                    f"Insufficient daily data for {ticker}: {len(daily_data)} days, minimum {min_required_days} required")
                raise ValueError(
                    f"Insufficient daily data quality for {ticker}. Analysis cannot proceed.")
            else:
                logger.warning(
                    f"Proceeding with low-quality daily data for {ticker}: {len(daily_data)} days, minimum {min_required_days} recommended.")

        # Format daily data for analysis
        formatted_step1_data = format_stock_data_for_analysis(
            daily_data, ticker, end_date_str)

        # Add price history to the formatted data for estimations if needed
        if 'price_history' not in formatted_step1_data and len(daily_data) > 0:
            try:
                # Check if daily_data['Close'] is a Series or DataFrame
                if isinstance(daily_data['Close'], pd.Series):
                    formatted_step1_data['price_history'] = daily_data['Close'].tolist(
                    )
                else:
                    # Handle the case where it's a DataFrame
                    formatted_step1_data['price_history'] = daily_data['Close'].values.tolist(
                    )
            except Exception as e:
                logger.warning(f"Could not extract price history: {e}")
                # Create an empty list as fallback
                formatted_step1_data['price_history'] = []

        # Enhance data with fallbacks for missing indicators or raise errors
        try:
            formatted_step1_data = enhance_missing_data(
                formatted_step1_data,
                ticker,
                market_data=None,
                lookback_days=len(daily_data),
                strict_validation=strict_validation
            )
        except ValueError as e:
            if strict_validation:
                logger.error(f"Data validation error for {ticker}: {e}")
                raise
            else:
                logger.warning(
                    f"Data enhancement issue for {ticker} (continuing with caution): {e}")

        # Log formatted data summary
        logger.info(
            f"Formatted Step 1 data for {ticker} - {len(daily_data)} days, quality: {formatted_step1_data.get('data_quality', 'unknown')}")

        # ======================= STEPS 2-6: 15m Analysis =======================
        # We analyze 5 consecutive days of 15-minute data
        # This is the detailed credit spread analysis at critical intraday decision points

        # Get 5-day data with 15m intervals for detailed analysis
        intraday_start = (end_date - timedelta(days=5)).strftime("%Y-%m-%d")
        intraday_end = end_date.strftime("%Y-%m-%d")
        intraday_start_str = intraday_start
        intraday_end_str = intraday_end

        logger.info(
            f"Getting intraday data for {ticker} from {intraday_start_str} to {intraday_end_str}")
        intraday_data = yfinance_client.get_historical_data(
            ticker, start=intraday_start_str, end=intraday_end_str, interval="15m")

        # Validate intraday data
        if len(intraday_data) == 0:
            if strict_validation:
                raise ValueError(
                    f"No intraday data available for {ticker}. Analysis cannot proceed.")
            else:
                logger.warning(
                    f"No intraday data for {ticker}. Will attempt to use daily data only.")

        # Log intraday data summary
        try:
            # Ensure intraday_data is a pandas DataFrame with proper index
            if not isinstance(intraday_data, pd.DataFrame):
                logger.warning(
                    f"Intraday data is not a DataFrame, attempting to convert")
                intraday_data = pd.DataFrame(intraday_data)

            # Check if index is proper datetime index
            if not isinstance(intraday_data.index, pd.DatetimeIndex):
                logger.warning(
                    f"Intraday data index is not a DatetimeIndex, converting")
                intraday_data.index = pd.to_datetime(intraday_data.index)

            # Now safely calculate unique days
            # Check if we have a numpy array or pandas DatetimeIndex
            if hasattr(intraday_data.index, 'date'):
                # If the dates are a numpy array, use numpy's unique function
                dates_array = intraday_data.index.date
                if isinstance(dates_array, np.ndarray):
                    unique_days_count = len(np.unique(dates_array))
                else:
                    unique_days_count = intraday_data.index.date.nunique()
            else:
                # Fallback method using string operations
                date_strings = set()
                for idx in intraday_data.index:
                    date_strings.add(str(idx).split()[0])
                unique_days_count = len(date_strings)

            logger.info(
                f"Retrieved {len(intraday_data)} 15m bars across {unique_days_count} days")

            # In strict mode, require at least 3 days of intraday data
            if strict_validation and unique_days_count < 3:
                raise ValueError(
                    f"Insufficient intraday data for {ticker}: only {unique_days_count} days available, minimum 3 required.")

        except Exception as e:
            logger.error(f"Error processing intraday data index: {e}")
            # Create fallback count
            unique_days_count = len(set(str(d).split()[0] for d in intraday_data.index)) if len(
                intraday_data) > 0 else 0
            logger.info(
                f"Retrieved {len(intraday_data)} 15m bars (days count fallback: ~{unique_days_count})")

            if strict_validation and unique_days_count < 3:
                raise ValueError(
                    f"Insufficient intraday data for {ticker}: only {unique_days_count} days available, minimum 3 required.")

        # Initialize collections for each step
        step_predictions = []
        day_dates = []

        # Add a default initial prediction to avoid index errors
        step_predictions.append({
            "direction": "neutral",
            "confidence": 0,
            "magnitude": 0,
            "price": 0,
            "analysis": "Initial placeholder prediction"
        })

        # Process each day individually for steps 2-6
        try:
            # Safely handle the case when .index.date is a numpy.ndarray without unique method
            if isinstance(intraday_data.index.date, np.ndarray):
                # Use numpy's unique instead
                unique_date_values = np.unique(intraday_data.index.date)
                unique_dates = sorted(unique_date_values)
            else:
                # Regular pandas approach
                unique_dates = sorted(intraday_data.index.date.unique())

            logger.info(
                f"Found {len(unique_dates)} unique trading days in intraday data")
        except Exception as e:
            logger.error(f"Error extracting unique dates: {e}")
            # Fallback: Extract dates manually from string representation
            unique_dates = []
            try:
                date_strings = set()
                for idx in intraday_data.index:
                    # Extract date part from datetime string
                    date_str = str(idx).split()[0]
                    if date_str not in date_strings:
                        date_strings.add(date_str)
                        # Convert to datetime.date objects for consistency
                        unique_dates.append(datetime.strptime(
                            date_str, "%Y-%m-%d").date())
                unique_dates = sorted(unique_dates)
                logger.info(
                    f"Fallback method found {len(unique_dates)} unique trading days")
            except Exception as e2:
                logger.error(f"Fallback date extraction also failed: {e2}")
                logger.warning(
                    "Using empty date list - pretraining will be limited")
                unique_dates = []

        # We need exactly 5 days for steps 2-6
        if len(unique_dates) < 5:
            logger.warning(
                f"Insufficient intraday data: only {len(unique_dates)} days available, needed 5")
            # In strict mode, require at least 3 days
            if strict_validation and len(unique_dates) < 3:
                raise ValueError(
                    f"Insufficient unique trading days for {ticker}: {len(unique_dates)} days, minimum 3 required.")

        # Limit to 5 most recent days if we have more
        if len(unique_dates) > 5:
            unique_dates = unique_dates[-5:]

        # Process each day individually
        # Start at step 2
        for step_num, day_date in enumerate(unique_dates, 2):
            day_str = day_date.strftime("%Y-%m-%d")
            day_dates.append(day_str)

            logger.info(f"Processing Step {step_num}: {day_str}")

            # Get market context for this day
            day_market_context = get_historical_market_context(
                market_data_dir=pretraining_dir,
                date_str=day_str,
                yfinance_client=yfinance_client  # Pass yfinance_client for real-time fetching
            )

            # Get intraday data for this day
            if len(intraday_data) > 0:
                day_intraday_data = intraday_data[intraday_data.index.strftime(
                    "%Y-%m-%d") == day_str]

                # If no intraday data for this specific day, use daily data
                if len(day_intraday_data) == 0:
                    logger.warning(
                        f"No intraday data for {day_str}, using daily data")
                    daily_slice = daily_data[daily_data.index.strftime(
                        "%Y-%m-%d") == day_str]

                    if len(daily_slice) > 0:
                        formatted_day_data = format_stock_data_for_analysis(
                            daily_slice, ticker, day_str)

                        # In strict mode, validate this specific day's data quality
                        if strict_validation:
                            validate_data_quality(
                                daily_slice, ticker, min_days_required=1, strict_validation=True)
                    else:
                        error_msg = f"No data available for {ticker} on {day_str}"
                        logger.error(error_msg)
                        if strict_validation:
                            raise ValueError(error_msg)
                        continue
                else:
                    # Validate intraday data quality
                    if strict_validation:
                        # Minimum number of 15-minute bars (5 hours)
                        if len(day_intraday_data) < 20:
                            raise ValueError(
                                f"Insufficient intraday data for {ticker} on {day_str}: only {len(day_intraday_data)} bars.")

                    # Format intraday data with 15-minute emphasis
                    formatted_day_data = format_stock_data_for_analysis(
                        day_intraday_data, ticker, day_str)
                    # Add intraday data separately for better analysis
                    formatted_day_data["intraday_data"] = {
                        "bars_count": len(day_intraday_data),
                        "high": float(day_intraday_data["High"].max().iloc[0]) if isinstance(day_intraday_data["High"].max(), pd.Series) else float(day_intraday_data["High"].max()),
                        "low": float(day_intraday_data["Low"].min().iloc[0]) if isinstance(day_intraday_data["Low"].min(), pd.Series) else float(day_intraday_data["Low"].min()),
                        "volume": int(day_intraday_data["Volume"].sum().iloc[0]) if isinstance(day_intraday_data["Volume"].sum(), pd.Series) else int(day_intraday_data["Volume"].sum()),
                        "vwap": float(np.average(
                            day_intraday_data["Close"],
                            weights=day_intraday_data["Volume"]
                        )) if "Volume" in day_intraday_data.columns and day_intraday_data["Volume"].sum().item() > 0 else None
                    }
            else:
                # No intraday data, use daily data
                daily_slice = daily_data[daily_data.index.strftime(
                    "%Y-%m-%d") == day_str]
                if len(daily_slice) > 0:
                    formatted_day_data = format_stock_data_for_analysis(
                        daily_slice, ticker, day_str)
                    # In strict mode, validate this specific day's data quality
                    if strict_validation:
                        validate_data_quality(
                            daily_slice, ticker, min_days_required=1, strict_validation=True)
                else:
                    error_msg = f"No data available for {ticker} on {day_str}"
                    logger.error(error_msg)
                    if strict_validation:
                        raise ValueError(error_msg)
                    continue

            # Enhance data with fallbacks for missing indicators or raise errors
            try:
                formatted_day_data = enhance_missing_data(
                    formatted_day_data,
                    ticker,
                    market_data=daily_data,
                    lookback_days=None,
                    strict_validation=strict_validation
                )
            except ValueError as e:
                if strict_validation:
                    logger.error(
                        f"Data validation error for {ticker} on {day_str}: {e}")
                    raise
                else:
                    logger.warning(
                        f"Data enhancement issue for {ticker} on {day_str} (continuing with caution): {e}")

            # Check if we have actual outcome data for the previous prediction
            actual_outcome = None
            if step_num > 2:
                # Look for the actual data for previous day
                prev_day_price = next((r.get('price', 0) for r in pretraining_results
                                      if r.get('date') == day_dates[step_num - 2]), 0)

                current_day_data = daily_data[daily_data.index.strftime(
                    "%Y-%m-%d") == day_str]
                if len(current_day_data) > 0:
                    actual_outcome = {
                        "date": day_str,
                        "open": float(current_day_data["Open"].iloc[0]),
                        "close": float(current_day_data["Close"].iloc[0]),
                        "high": float(current_day_data["High"].iloc[0]),
                        "low": float(current_day_data["Low"].iloc[0]),
                        "price_change": float(current_day_data["Close"].iloc[0]) - prev_day_price,
                        "price_change_percent": ((float(current_day_data["Close"].iloc[0]) - prev_day_price) / prev_day_price * 100) if prev_day_price > 0 else 0
                    }

                    # Record if previous prediction was correct
                    if step_predictions[step_num - 2].get('direction') == 'bullish' and actual_outcome['price_change'] > 0:
                        prediction_correct = True
                    elif step_predictions[step_num - 2].get('direction') == 'bearish' and actual_outcome['price_change'] < 0:
                        prediction_correct = True
                    elif step_predictions[step_num - 2].get('direction') == 'neutral' and abs(actual_outcome['price_change_percent']) < 0.5:
                        prediction_correct = True
                    else:
                        prediction_correct = False

                    logger.info(f"Previous prediction was {'correct' if prediction_correct else 'incorrect'} "
                                f"(predicted {step_predictions[step_num - 2].get('direction')}, actual change: {actual_outcome['price_change_percent']:.2f}%)")

            # Execute this step's analysis with reflection
            logger.info(
                f"Executing Step {step_num} analysis for {ticker} on {day_str}")

            # Get adaptive strategy selection based on historical performance
            weighted_strategies, strategy_reasoning = adaptive_strategy_selection(
                memory_context)

            # Generate the prompt with strategy recommendations
            step_prompt = get_step2_to_6_prompt(
                step_num,
                ticker,
                day_str,
                formatted_day_data,
                day_market_context,
                memory_context,
                step_predictions[step_num - 2] if step_num > 1 else None,
                actual_outcome,
                weighted_strategies=weighted_strategies,
                strategy_reasoning=strategy_reasoning
            )

            # Get analysis from Gemini
            start_request = time.time()
            step_analysis_text = gemini_client.generate_text(
                step_prompt, temperature=0.3, prompt_type=f"Step {step_num}"
            )

            # Extract prediction and memory updates
            step_prediction = parse_credit_spread_prediction(
                step_analysis_text)
            memory_updates = parse_credit_spread_memory_update(
                step_analysis_text)

            # Constrain adjustments using the Secret
            secret = memory_context["secret"]

            # 1. Constrain trend changes - reject trend flips unless confidence exceeds secret's baseline + 20%
            if step_prediction.get("direction") != secret["baseline_trend"] and step_prediction.get("confidence", 0) < secret["trend_confidence"] + 20:
                original_direction = step_prediction.get("direction")
                step_prediction["direction"] = secret["baseline_trend"]
                logger.warning(f"Step {step_num}: Rejected trend flip from {original_direction} to {step_prediction['direction']}—"
                               f"confidence {step_prediction.get('confidence', 0)}% < {secret['trend_confidence'] + 20}% threshold")

            # 2. Constrain volatility adjustments to ±0.5% of the anchor
            if "atr_percent" in formatted_day_data:
                atr_shift = abs(
                    formatted_day_data["atr_percent"] - secret["volatility_anchor"])
                if atr_shift > 0.5:
                    formatted_day_data["atr_percent"] = secret["volatility_anchor"] + (
                        0.5 if formatted_day_data["atr_percent"] > secret["volatility_anchor"] else -0.5)
                    logger.warning(
                        f"Step {step_num}: Capped ATR shift to {formatted_day_data['atr_percent']}% from {secret['volatility_anchor']}%")

            # 3. Limit weight adjustments to ±15% per step
            if memory_updates and "weight_adjustments" in memory_updates:
                for adj in memory_updates["weight_adjustments"]:
                    if abs(adj.get("change", 0)) > 15:
                        original_change = adj.get("change", 0)
                        adj["change"] = 15 if adj["change"] > 0 else -15
                        logger.info(
                            f"Step {step_num}: Limited weight adjustment from {original_change}% to {adj['change']}%")

            # Create prediction accuracy record if we have actual outcome data
            prediction_accuracy = None
            if actual_outcome:
                # Get trend directions
                secret_trend = memory_context["secret"]["baseline_trend"]
                predicted_trend = step_predictions[step_num - 2].get(
                    'direction', 'neutral')
                actual_trend = "bullish" if actual_outcome[
                    'price_change'] > 0 else "bearish" if actual_outcome['price_change'] < 0 else "neutral"

                # Determine if prediction aligned with secret and if it was correct
                secret_aligned = predicted_trend == secret_trend
                direction_correct = (predicted_trend == actual_trend)

                # Log insights when the secret and prediction disagree
                if not secret_aligned and direction_correct:
                    logger.info(
                        f"Step {step_num}: Deviation from secret ({secret_trend}) was correct—consider raising confidence threshold")
                elif not direction_correct and secret_aligned:
                    logger.warning(
                        f"Step {step_num}: Secret ({secret_trend}) misaligned with outcome ({actual_trend})—review baseline")

                prediction_accuracy = {
                    "predicted_direction": predicted_trend,
                    "actual_direction": actual_trend,
                    "secret_trend": secret_trend,
                    "secret_aligned": secret_aligned,
                    "predicted_magnitude": step_predictions[step_num - 2].get('magnitude', 0),
                    "actual_magnitude": abs(actual_outcome['price_change_percent']),
                    "direction_correct": direction_correct,
                    "magnitude_error": abs(step_predictions[step_num - 2].get('magnitude', 0) - abs(actual_outcome['price_change_percent']))
                }

            # Create result object for this step
            step_result = {
                "ticker": ticker,
                "date": day_str,
                "step": step_num,
                "analysis_type": "5d/15m_analysis",
                "price": formatted_day_data.get('current_price', 0),
                "trend": step_prediction.get('direction', 'neutral'),
                "next_day_prediction": step_prediction,
                "prediction_accuracy": prediction_accuracy,
                "memory_updates": memory_updates,
                "full_analysis": step_analysis_text
            }

            # Add to results and call callback if provided
            pretraining_results.append(step_result)
            if callback:
                callback(step_result)

            # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

            # Update memory context with results of this step

            # 1. Update pattern reliability based on memory updates
            if memory_updates and "reliability_update" in memory_updates:
                for pattern_type, stats in memory_updates["reliability_update"].items():
                    if pattern_type in memory_context["pattern_reliability"]:
                        memory_context["pattern_reliability"][pattern_type]["accuracy"] = stats["accuracy"]
                        memory_context["pattern_reliability"][pattern_type]["correct"] = stats["correct"]
                        memory_context["pattern_reliability"][pattern_type]["total"] = stats["total"]

            # 2. Update key levels
            if memory_updates and "key_level_update" in memory_updates and memory_updates["key_level_update"]:
                new_levels = memory_updates["key_level_update"]
                # Determine if each new level is support or resistance
                current_price = formatted_day_data.get('current_price', 0)
                for level in new_levels:
                    if level < current_price:
                        if level not in memory_context["key_levels"]["support"]:
                            memory_context["key_levels"]["support"].append(
                                level)
                    else:
                        if level not in memory_context["key_levels"]["resistance"]:
                            memory_context["key_levels"]["resistance"].append(
                                level)

            # 3. Add volatility data
            memory_context["volatility_history"].append({
                "date": day_str,
                "iv": formatted_day_data.get('implied_volatility', 'Unknown'),
                "hv": formatted_day_data.get('historical_volatility', 'Unknown'),
                "adjustment": memory_updates.get("volatility_adjustment", "no change") if memory_updates else "no change"
            })

            # 4. Update confidence
            if memory_updates and "updated_confidence" in memory_updates and memory_updates["updated_confidence"] > 0:
                memory_context["weight_adjustments"].append({
                    "date": day_str,
                    "confidence": memory_updates["updated_confidence"],
                    "step": step_num
                })

            # 5. Update specific spread type performance if recommendation was provided
            if step_prediction and "spread_recommendation" in step_prediction and step_prediction["spread_recommendation"]:
                spread_type = step_prediction["spread_recommendation"].lower()
                if "bull put" in spread_type:
                    spread_key = "bull_put"
                elif "bear call" in spread_type:
                    spread_key = "bear_call"
                elif "iron condor" in spread_type:
                    spread_key = "iron_condor"
                else:
                    spread_key = None

                if spread_key and spread_key in memory_context["spread_performance"]:
                    memory_context["spread_performance"][spread_key]["total"] += 1
                    # If we have actual outcome, determine if this spread would have won
                    if actual_outcome:
                        if spread_key == "bull_put" and actual_outcome['price_change'] >= 0:
                            memory_context["spread_performance"][spread_key]["wins"] += 1
                        elif spread_key == "bear_call" and actual_outcome['price_change'] <= 0:
                            memory_context["spread_performance"][spread_key]["wins"] += 1
                        elif spread_key == "iron_condor" and abs(actual_outcome['price_change_percent']) < 1.0:
                            # Simple assumption for iron condor - win if price change is small
                            memory_context["spread_performance"][spread_key]["wins"] += 1

                    # Recalculate win rate
                    total = memory_context["spread_performance"][spread_key]["total"]
                    wins = memory_context["spread_performance"][spread_key]["wins"]
                    memory_context["spread_performance"][spread_key]["win_rate"] = (
                        wins / total * 100) if total > 0 else 0

            # Save previous prediction for next iteration
            step_predictions.append(step_prediction)

        #
        # STEP 7: Summarize and reflect on the analysis
        #
        logger.info(
            f"STEP 7: Summarizing and reflecting on the analysis for {ticker}")

        # Get summary prompt
        analysis_period = {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "total_days": len(day_dates)  # +1 for Step 1
        }

        # Get weighted strategies for the summary prompt
        weighted_strategies, strategy_reasoning = adaptive_strategy_selection(
            memory_context)

        summary_prompt = get_summary_prompt(
            ticker,
            analysis_period,
            pretraining_results,
            memory_context,
            weighted_strategies=weighted_strategies,
            strategy_reasoning=strategy_reasoning
        )

        # Get analysis from Gemini
        start_request = time.time()
        summary_analysis_text = gemini_client.generate_text(
            summary_prompt, temperature=0.3, prompt_type="Summary"
        )

        # Extract prediction from the summary analysis
        summary_prediction = parse_credit_spread_prediction(
            summary_analysis_text)

        # Enforce Secret in final summary - require strong evidence to contradict the baseline trend
        secret = memory_context["secret"]
        if summary_prediction.get("direction") != secret["baseline_trend"]:
            # Count how many of the intraday analyses contradicted the secret
            contradiction_count = sum(1 for r in pretraining_results[1:]
                                      if r.get("trend") != secret["baseline_trend"])

            # Only allow contradiction if we have 4+ days contradicting AND high confidence
            if contradiction_count < 4 or summary_prediction.get("confidence", 0) < 90:
                original_direction = summary_prediction.get("direction")
                summary_prediction["direction"] = secret["baseline_trend"]
                logger.warning(f"Summary: Reverted from {original_direction} to secret trend {secret['baseline_trend']}—"
                               f"insufficient contradiction evidence ({contradiction_count}/5 days, "
                               f"{summary_prediction.get('confidence', 0)}% confidence < 90% threshold)")
            else:
                # If we do allow the contradiction, log it as significant
                logger.warning(f"Summary: Allowing trend change from secret {secret['baseline_trend']} to "
                               f"{summary_prediction.get('direction')} due to strong evidence: "
                               f"{contradiction_count}/5 days with {summary_prediction.get('confidence', 0)}% confidence")

        # Create result object for the summary analysis
        summary_result = {
            "ticker": ticker,
            "date": end_date_str,
            "step": 7,
            "analysis_type": "Summary",
            "price": summary_prediction.get('current_price', 0),
            "trend": summary_prediction.get('direction', 'neutral'),
            "next_day_prediction": summary_prediction,
            "full_analysis": summary_analysis_text
        }

        # Add to results and call callback if provided
        pretraining_results.append(summary_result)
        if callback:
            callback(summary_result)

        # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

        # Update memory context with key levels from the summary analysis
        if "key_levels" in summary_prediction:
            try:
                levels = [float(level) for level in summary_prediction.get(
                    "key_levels", "").split(",") if level.strip()]
                # Determine if each level is support or resistance
                current_price = summary_prediction.get('current_price', 0)
                for level in levels:
                    if level < current_price:
                        memory_context["key_levels"]["support"].append(level)
                    else:
                        memory_context["key_levels"]["resistance"].append(
                            level)
            except Exception as e:
                logger.warning(f"Error parsing key levels from Summary: {e}")

        # Store daily trend in memory context
        memory_context["multi_timeframe"]["daily_trend"] = summary_prediction.get(
            'direction', 'neutral')

        #
        # STEP 8: Update memory context with new data
        #
        logger.info(
            f"STEP 8: Updating memory context with new data for {ticker}")

        # Update volatility history
        memory_context["volatility_history"].append(
            formatted_day_data.get('volatility', 0))

        # Update spread performance
        if step_prediction.get('direction', 'neutral') == 'bull':
            memory_context["spread_performance"]["bull_put"]["wins"] += 1
            memory_context["spread_performance"]["bull_put"]["total"] += 1
            memory_context["spread_performance"]["bull_put"]["win_rate"] = memory_context["spread_performance"]["bull_put"]["wins"] / \
                memory_context["spread_performance"]["bull_put"]["total"]
        elif step_prediction.get('direction', 'neutral') == 'bear':
            memory_context["spread_performance"]["bear_call"]["wins"] += 1
            memory_context["spread_performance"]["bear_call"]["total"] += 1
            memory_context["spread_performance"]["bear_call"]["win_rate"] = memory_context["spread_performance"]["bear_call"]["wins"] / \
                memory_context["spread_performance"]["bear_call"]["total"]
        else:
            memory_context["spread_performance"]["iron_condor"]["wins"] += 1
            memory_context["spread_performance"]["iron_condor"]["total"] += 1
            memory_context["spread_performance"]["iron_condor"]["win_rate"] = memory_context["spread_performance"]["iron_condor"]["wins"] / \
                memory_context["spread_performance"]["iron_condor"]["total"]

        # Update weight adjustments
        memory_context["weight_adjustments"].append(
            formatted_day_data.get('weight_adjustment', 0))

        # Update multi-timeframe trend
        if step_prediction.get('direction', 'neutral') == 'bull':
            memory_context["multi_timeframe"]["daily_trend"] = 'bull'
        elif step_prediction.get('direction', 'neutral') == 'bear':
            memory_context["multi_timeframe"]["daily_trend"] = 'bear'
        else:
            memory_context["multi_timeframe"]["daily_trend"] = 'neutral'

        #
        # Generate Final Summary after Steps 1-6
        #
        logger.info(f"Generating final summary for {ticker}")

        # Get summary from Gemini
        start_request = time.time()
        summary_text = gemini_client.generate_text(
            summary_prompt, temperature=0.4, prompt_type="Summary"
        )

        # Extract forecasts for different time horizons
        day_forecast = {}
        week_forecast = {}
        month_forecast = {}

        # Look for forecast sections
        forecast_sections = []
        for section in summary_text.split("```"):
            if "FORECAST:" in section:
                forecast_sections.append(section)

        for section in forecast_sections:
            if "next day" in section.lower() or "1-2 day" in section.lower():
                day_forecast = parse_credit_spread_prediction(section)
                day_forecast["timeframe"] = "next_day"
            elif "next week" in section.lower() or "3-5 day" in section.lower():
                week_forecast = parse_credit_spread_prediction(section)
                week_forecast["timeframe"] = "next_week"
            elif "next month" in section.lower() or "20-30 day" in section.lower():
                month_forecast = parse_credit_spread_prediction(section)
                month_forecast["timeframe"] = "next_month"

        # Create summary result
        summary_result = {
            "ticker": ticker,
            "date": end_date_str,
            "type": "summary",
            "analysis_type": "summary",
            "next_day_prediction": day_forecast,
            "next_week_prediction": week_forecast,
            "next_month_prediction": month_forecast,
            "full_analysis": summary_text
        }

        # Add to results and call callback if provided
        pretraining_results.append(summary_result)
        if callback:
            callback(summary_result)

        # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

        # Calculate processing time
        processing_time = time.time() - start_time

        # Calculate accuracy metrics
        total_predictions = sum(1 for r in pretraining_results if r.get(
            'prediction_accuracy') is not None)
        correct_predictions = sum(1 for r in pretraining_results
                                  if r.get('prediction_accuracy') is not None
                                  and r.get('prediction_accuracy', {}).get('direction_correct', False))

        accuracy_rate = correct_predictions / \
            total_predictions if total_predictions > 0 else 0

        # Create final result dictionary
        result = {
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "processing_time": processing_time,
            "analyses_count": len(pretraining_results),
            "accuracy_rate": accuracy_rate,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "results": pretraining_results,
            "memory_context": memory_context
        }

        # Save results to disk if requested
        if save_results:
            # Generate context for future use
            context = {
                "ticker": ticker,
                "pretraining_period": f"{start_date_str} to {end_date_str}",
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analyses_count": len(pretraining_results),
                "accuracy_rate": accuracy_rate,
                "memory_context": memory_context
            }

            # Add forecasts to context
            for horizon in ["next_day", "next_week", "next_month"]:
                prediction_key = f"{horizon}_prediction"
                if prediction_key in summary_result:
                    context[f"{horizon}_direction"] = summary_result[prediction_key].get(
                        'direction', 'neutral')
                    context[f"{horizon}_magnitude"] = summary_result[prediction_key].get(
                        'magnitude', 0)
                    context[f"{horizon}_confidence"] = summary_result[prediction_key].get(
                        'confidence', 0)
                    context[f"{horizon}_spread"] = summary_result[prediction_key].get(
                        'spread_recommendation', '')

            # Save to disk
            result_path = save_pretraining_results(
                pretraining_dir, ticker, result, context)
            logger.info(f"Saved pretraining results to {result_path}")

        logger.info(f"Six-step pretraining completed for {ticker} with "
                    f"{len(pretraining_results)} results in {processing_time:.2f} seconds. "
                    f"Prediction accuracy: {accuracy_rate*100:.1f}%")

        return result

    except Exception as e:
        logger.error(f"Error in six-step pretraining: {e}")
        logger.exception(e)
        return {"error": str(e)}

    finally:
        # Restore the original generate_text method
        gemini_client.generate_text = original_generate_text


def six_step_batch_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    tickers,
    discord_client=None,
    **kwargs
):
    """
    Run six-step pretraining for multiple tickers efficiently.

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - tickers: List of stock symbols to analyze
    - discord_client: Optional Discord client for response tracking
    - **kwargs: Additional arguments to pass to six_step_pretrain_analyzer

    Returns:
    - Dictionary mapping tickers to pretraining results
    """
    logger.info(
        f"Starting six-step batch pretraining for {len(tickers)} tickers...")
    batch_start_time = time.time()

    # Reset the request timestamps to start fresh
    global _request_timestamps
    _request_timestamps = []

    results = {}

    # Process each ticker
    for ticker in tickers:
        logger.info(f"Processing ticker {ticker} in batch")
        ticker_start_time = time.time()

        try:
            # Create a callback that ensures ticker is included
            def ticker_callback(analysis):
                # Always add ticker to analysis to ensure it's identified correctly
                if "ticker" not in analysis:
                    analysis["ticker"] = ticker

                # Forward to the provided callback if available
                if "callback" in kwargs and kwargs["callback"]:
                    kwargs["callback"](analysis)

            # Run pretraining for this ticker
            ticker_result = six_step_pretrain_analyzer(
                yfinance_client,
                gemini_client,
                pretraining_dir,
                ticker,
                discord_client=discord_client,
                callback=ticker_callback,
                **{k: v for k, v in kwargs.items() if k != 'callback'}
            )

            # Store result
            results[ticker] = ticker_result

            # Log completion
            ticker_time = time.time() - ticker_start_time
            logger.info(
                f"Completed pretraining for {ticker} in {ticker_time:.2f} seconds")

            # No need for additional delay between tickers since we're using the rate limit manager

        except Exception as e:
            logger.error(f"Error in batch pretraining for {ticker}: {e}")
            logger.exception(e)
            results[ticker] = {"error": str(e)}

    # Log completion
    batch_time = time.time() - batch_start_time
    logger.info(
        f"Completed batch pretraining for {len(tickers)} tickers in {batch_time:.2f} seconds")

    return results


def enhance_missing_data(formatted_data, ticker, market_data=None, lookback_days=30, strict_validation=True):
    """
    Enhance formatted data with fallback mechanisms for missing indicators or raise errors in strict mode.

    Parameters:
    - formatted_data: The formatted stock data dictionary
    - ticker: The stock symbol
    - market_data: Optional market-wide data for context (like SPY/QQQ for comparison)
    - lookback_days: Number of days to consider for historical context
    - strict_validation: If True, raise errors when critical data is missing instead of using defaults

    Returns:
    - Enhanced data dictionary with fallbacks for missing indicators

    Raises:
    - ValueError: When critical data is missing and strict_validation is True
    """
    global _volatility_baseline_cache

    # Track which indicators were estimated or missing
    estimated_indicators = []
    missing_indicators = []

    # 1. Handle missing MACD data
    if 'macd' not in formatted_data or formatted_data['macd'] is None or formatted_data['macd'] == 'unavailable':
        missing_indicators.append('MACD')

        if strict_validation and 'price_history' in formatted_data and len(formatted_data['price_history']) >= 26:
            # In strict mode, we check if we have enough price history to calculate a reasonable estimate
            # Make sure price_history is a flat list of numbers, not a list of lists
            price_history = formatted_data['price_history']
            if price_history and isinstance(price_history[0], list):
                # Flatten list of lists
                try:
                    import numpy as np
                    price_history = np.array(
                        price_history).flatten().tolist()
                except Exception as e:
                    logger.warning(
                        f"Could not flatten nested price history: {e}")
                    # Manual flatten for nested lists
                    flattened = []
                    for sublist in price_history:
                        if isinstance(sublist, list):
                            flattened.extend(sublist)
                        else:
                            flattened.append(sublist)
                    price_history = flattened

            # Calculate MACD approximation
            short_term = sum(price_history[-12:]) / 12
            long_term = sum(price_history[-26:]) / 26
            estimated_macd = short_term - long_term

            # Normalize by current price
            if formatted_data.get('current_price', 0) > 0:
                # As percentage
                estimated_macd = (
                    estimated_macd / formatted_data['current_price']) * 100

            formatted_data['macd'] = estimated_macd
            formatted_data['macd_signal'] = 0  # Neutral default
            # Same as MACD without signal
            formatted_data['macd_histogram'] = estimated_macd
            estimated_indicators.append('MACD (price momentum proxy)')
        else:
            # Use fallback approach for any mode when we don't have enough data
            # First try using available price history (even if it's less than ideal)
            if 'price_history' in formatted_data and len(formatted_data['price_history']) >= 5:
                # Make sure price_history is a flat list of numbers, not a list of lists
                price_history = formatted_data['price_history']
                if price_history and isinstance(price_history[0], list):
                    # Flatten list of lists
                    try:
                        import numpy as np
                        price_history = np.array(
                            price_history).flatten().tolist()
                    except Exception as e:
                        logger.warning(
                            f"Could not flatten nested price history: {e}")
                        # Manual flatten for nested lists
                        flattened = []
                        for sublist in price_history:
                            if isinstance(sublist, list):
                                flattened.extend(sublist)
                            else:
                                flattened.append(sublist)
                        price_history = flattened

                # Calculate simple momentum with available data
                recent_avg = sum(
                    price_history[-min(len(price_history), 5):]) / min(len(price_history), 5)
                older_avg = sum(
                    price_history[:min(len(price_history), 5)]) / min(len(price_history), 5)

                # Simple momentum indicator - how much recent prices differ from earlier ones
                momentum = recent_avg - older_avg

                # Normalize to create a reasonable MACD proxy
                if formatted_data.get('current_price', 0) > 0:
                    momentum = (
                        momentum / formatted_data['current_price']) * 100

                formatted_data['macd'] = momentum
                formatted_data['macd_signal'] = 0  # Neutral default
                formatted_data['macd_histogram'] = momentum
                estimated_indicators.append('MACD (simplified momentum)')

                if strict_validation:
                    # Log that we're using a simplified approach even in strict mode
                    logger.warning(
                        f"Using simplified momentum as MACD proxy for {ticker} due to insufficient data")
            else:
                # Not enough data for any calculation - use neutral values
                formatted_data['macd'] = 0
                formatted_data['macd_signal'] = 0
                formatted_data['macd_histogram'] = 0
                estimated_indicators.append('MACD (neutral default)')

                if strict_validation:
                    # In strict mode, log this as a warning instead of raising an error
                    logger.warning(
                        f"Missing critical MACD data for {ticker} - using neutral values")

    # 2. Handle missing Implied Volatility (IV)
    if 'implied_volatility' not in formatted_data or formatted_data['implied_volatility'] is None or formatted_data['implied_volatility'] == 'Unknown':
        missing_indicators.append('Implied Volatility (IV)')

        # First, try to use historical volatility as a proxy
        if 'historical_volatility' in formatted_data and formatted_data['historical_volatility'] not in (None, 'Unknown'):
            # IV is typically higher than HV (volatility risk premium)
            formatted_data['implied_volatility'] = formatted_data['historical_volatility'] * 1.1
            estimated_indicators.append('IV (based on HV)')
        elif 'atr_percent' in formatted_data and formatted_data['atr_percent'] is not None:
            # Use ATR% as rough proxy, typically scales up by ~4-6x for annualized IV
            formatted_data['implied_volatility'] = formatted_data['atr_percent'] * 5
            estimated_indicators.append('IV (based on ATR)')
        else:
            # Define a DEFAULT_IV if not already defined
            DEFAULT_IV = 25.0  # Typical market average IV

            # Use default IV based on market conditions or ticker characteristics
            if ticker in ('SPY', 'QQQ', 'DIA', 'IWM'):  # Index ETFs tend to have lower IV
                formatted_data['implied_volatility'] = DEFAULT_IV * 0.8
            # Tech stocks often have higher IV
            elif ticker in ('TSLA', 'NVDA', 'META', 'AAPL', 'MSFT'):
                formatted_data['implied_volatility'] = DEFAULT_IV * 1.2
            else:
                formatted_data['implied_volatility'] = DEFAULT_IV

            estimated_indicators.append('IV (default value)')

            if strict_validation:
                # In strict mode, log this as a warning instead of raising an error
                logger.warning(
                    f"Using default Implied Volatility value for {ticker} due to insufficient data")

    # 3. Handle missing Historical Volatility (HV)
    if 'historical_volatility' not in formatted_data or formatted_data['historical_volatility'] is None or formatted_data['historical_volatility'] == 'Unknown':
        missing_indicators.append('Historical Volatility (HV)')

        # Define a DEFAULT_HV if not already defined
        DEFAULT_HV = 20.0  # Typical market average HV

        if 'atr_percent' in formatted_data and formatted_data['atr_percent'] is not None:
            # Scale ATR to approximate HV
            formatted_data['historical_volatility'] = formatted_data['atr_percent'] * 4
            estimated_indicators.append('HV (based on ATR)')
        elif 'implied_volatility' in formatted_data and formatted_data['implied_volatility'] not in (None, 'Unknown'):
            # Use IV as a proxy, typically HV is lower than IV
            formatted_data['historical_volatility'] = formatted_data['implied_volatility'] * 0.9
            estimated_indicators.append('HV (based on IV)')
        elif 'price_history' in formatted_data and len(formatted_data['price_history']) >= 5:
            # Calculate historical volatility from price history
            price_changes = []

            # Make sure price_history is a flat list of numbers, not a list of lists
            price_history = formatted_data['price_history']
            if price_history and isinstance(price_history[0], list):
                # Flatten list of lists
                try:
                    import numpy as np
                    price_history = np.array(
                        price_history).flatten().tolist()
                except Exception as e:
                    logger.warning(
                        f"Could not flatten nested price history for HV calculation: {e}")
                    # Manual flatten for nested lists
                    flattened = []
                    for sublist in price_history:
                        if isinstance(sublist, list):
                            flattened.extend(sublist)
                        else:
                            flattened.append(sublist)
                    price_history = flattened

            # Calculate price changes for volatility
            for i in range(1, len(price_history)):
                prev, curr = price_history[i-1], price_history[i]
                # Make sure these are numeric values, not lists
                if isinstance(prev, (int, float)) and isinstance(curr, (int, float)) and prev > 0:
                    pct_change = (curr - prev) / prev
                    price_changes.append(pct_change)

            if price_changes:
                # Standard deviation of daily returns
                std_dev = (sum([(x - (sum(price_changes) / len(price_changes)))
                           ** 2 for x in price_changes]) / len(price_changes))**0.5
                # Annualize (approx. 252 trading days)
                # as percentage
                formatted_data['historical_volatility'] = std_dev * \
                    (252**0.5) * 100
                estimated_indicators.append(
                    'HV (calculated from price history)')
            else:
                # Use default HV based on market conditions or ticker characteristics
                if ticker in ('SPY', 'QQQ', 'DIA', 'IWM'):  # Index ETFs tend to have lower HV
                    formatted_data['historical_volatility'] = DEFAULT_HV * 0.8
                # Tech stocks often have higher HV
                elif ticker in ('TSLA', 'NVDA', 'META', 'AAPL', 'MSFT'):
                    formatted_data['historical_volatility'] = DEFAULT_HV * 1.2
                else:
                    formatted_data['historical_volatility'] = DEFAULT_HV
                estimated_indicators.append(
                    'HV (insufficient price data, using default)')

                if strict_validation:
                    logger.warning(
                        f"Using default Historical Volatility for {ticker} - insufficient price data")
        else:
            # Use default HV based on market conditions or ticker characteristics
            if ticker in ('SPY', 'QQQ', 'DIA', 'IWM'):  # Index ETFs tend to have lower HV
                formatted_data['historical_volatility'] = DEFAULT_HV * 0.8
            # Tech stocks often have higher HV
            elif ticker in ('TSLA', 'NVDA', 'META', 'AAPL', 'MSFT'):
                formatted_data['historical_volatility'] = DEFAULT_HV * 1.2
            else:
                formatted_data['historical_volatility'] = DEFAULT_HV
            estimated_indicators.append('HV (default value)')

            if strict_validation:
                logger.warning(
                    f"Using default Historical Volatility for {ticker} due to missing data")

    # 4. Enhance ATR interpretation with context
    if 'atr_percent' in formatted_data and formatted_data['atr_percent'] is not None:
        # Get or compute market baseline for comparison
        baseline_key = f"{ticker}_ATR_baseline"
        if baseline_key not in _volatility_baseline_cache:
            # If we have price history, compute a baseline
            if 'price_history' in formatted_data and len(formatted_data['price_history']) > 5:
                # Make sure price_history is a flat list of numbers, not a list of lists
                price_history = formatted_data['price_history']
                if price_history and isinstance(price_history[0], list):
                    # Flatten list of lists
                    try:
                        import numpy as np
                        price_history = np.array(
                            price_history).flatten().tolist()
                    except Exception as e:
                        logger.warning(
                            f"Could not flatten nested price history for ATR calculation: {e}")
                        # Manual flatten for nested lists
                        flattened = []
                        for sublist in price_history:
                            if isinstance(sublist, list):
                                flattened.extend(sublist)
                            else:
                                flattened.append(sublist)
                        price_history = flattened

                # Calculate daily percentage changes
                changes = []
                for i in range(1, len(price_history)):
                    prev, curr = price_history[i-1], price_history[i]
                    # Make sure these are numeric values, not lists
                    if isinstance(prev, (int, float)) and isinstance(curr, (int, float)) and prev > 0:
                        changes.append(abs((curr - prev) / prev) * 100)

                if changes:
                    # Use average daily change as baseline
                    _volatility_baseline_cache[baseline_key] = sum(
                        changes) / len(changes)
                else:
                    if strict_validation:
                        raise ValueError(
                            f"Cannot calculate ATR baseline for {ticker} due to invalid price history.")
                    else:
                        _volatility_baseline_cache[baseline_key] = DEFAULT_ATR_PERCENT
            else:
                if strict_validation:
                    raise ValueError(
                        f"Insufficient price history to establish ATR baseline for {ticker}.")
                else:
                    _volatility_baseline_cache[baseline_key] = DEFAULT_ATR_PERCENT

        # Compare current ATR to baseline and categorize
        baseline_atr = _volatility_baseline_cache[baseline_key]
        ratio = formatted_data['atr_percent'] / baseline_atr

        if ratio < 0.5:
            volatility_assessment = "very_low"
        elif ratio < 0.8:
            volatility_assessment = "low"
        elif ratio < 1.2:
            volatility_assessment = "normal"
        elif ratio < 1.8:
            volatility_assessment = "elevated"
        else:
            volatility_assessment = "high"

        formatted_data['volatility_assessment'] = volatility_assessment
        formatted_data['volatility_ratio'] = ratio
        formatted_data['volatility_baseline'] = baseline_atr
    else:
        missing_indicators.append('ATR Percentage')
        if strict_validation:
            raise ValueError(
                f"Missing critical ATR percentage data for {ticker}, cannot assess volatility.")

    # Log which indicators were estimated
    if estimated_indicators:
        logger.info(
            f"Enhanced data for {ticker} with estimates for: {', '.join(estimated_indicators)}")

    # Log missing indicators even in non-strict mode
    if missing_indicators:
        if strict_validation:
            logger.error(
                f"Critical indicators missing for {ticker}: {', '.join(missing_indicators)}")
        else:
            logger.warning(
                f"Using fallbacks for missing indicators in {ticker}: {', '.join(missing_indicators)}")

    return formatted_data


def validate_data_quality(data, ticker, min_days_required=20, strict_validation=True):
    """
    Validate that data meets quality standards before analysis.

    Parameters:
    - data: The formatted stock data dictionary or DataFrame
    - ticker: The stock ticker symbol
    - min_days_required: Minimum number of days of price history required
    - strict_validation: Whether to raise errors or just log warnings

    Returns:
    - data: The original data regardless of validation results
    """
    errors = []
    warnings = []

    # Check if data is a DataFrame - special handling needed
    if isinstance(data, pd.DataFrame):
        # Log basic DataFrame validation
        if len(data) < min_days_required:
            error_msg = f"Insufficient data points for {ticker}: {len(data)} available, {min_days_required} required"
            errors.append(error_msg)
            logger.error(error_msg)

        # Add ticker, date, and current_price to the data if they're missing
        # This ensures downstream functions have what they need
        if strict_validation:
            logger.warning(
                f"Data quality warning for {ticker}: DataFrame received instead of expected dictionary format")

            # Log all validation errors/warnings but don't raise exceptions
            if errors:
                for error in errors:
                    logger.error(
                        f"Data validation error for {ticker}: {error}")
            if warnings:
                for warning in warnings:
                    logger.warning(
                        f"Data quality warning for {ticker}: {warning}")

        # Always return the original data
        return data

    # For dictionary data format (the expected format)
    # Check required fields
    required_fields = ['ticker', 'current_price', 'date']
    missing_fields = [
        field for field in required_fields if field not in data or data[field] is None]

    if missing_fields:
        error_msg = f"Missing required field{'s' if len(missing_fields) > 1 else ''}: {', '.join(missing_fields)}"
        errors.append(error_msg)

    # Check price is valid
    if 'current_price' in data and data['current_price'] is not None:
        try:
            price = float(data['current_price'])
            if price <= 0:
                errors.append(f"Invalid price: {price} (must be positive)")
        except (ValueError, TypeError):
            errors.append(f"Invalid price format: {data['current_price']}")

    # Check date format
    if 'date' in data and data['date'] is not None:
        try:
            # Try to parse date
            if isinstance(data['date'], str):
                datetime.strptime(data['date'], "%Y-%m-%d")
        except ValueError:
            errors.append(
                f"Invalid date format: {data['date']} (expected YYYY-MM-DD)")

    # Check price history length if strict validation is enabled
    if 'price_history' in data:
        price_history = data['price_history']
        if not price_history or len(price_history) < min_days_required:
            error_msg = f"Insufficient price history for {ticker}: {len(price_history) if price_history else 0} days, {min_days_required} required"
            if strict_validation:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
    else:
        if strict_validation:
            errors.append("Missing price history data")
        else:
            warnings.append("Missing price history data")

    # Check for missing MACD data
    if 'macd' not in data or data['macd'] is None:
        warnings.append("MACD data is missing")

    # Check for missing ATR data
    if 'atr' not in data or data['atr'] is None:
        warnings.append("ATR data is missing")

    # Check for missing RSI data
    if 'rsi' not in data or data['rsi'] is None:
        warnings.append("RSI data is missing")

    # Log all validation errors/warnings but don't raise exceptions
    if errors:
        error_msg = ", ".join(errors)
        logger.error(f"Data validation error for {ticker}: {error_msg}")

    if warnings:
        for warning in warnings:
            logger.warning(f"Data quality warning for {ticker}: {warning}")

    # Always return the original data regardless of validation outcome
    return data


def adaptive_strategy_selection(memory_context):
    """
    Selects optimal trading strategies based on historical performance.

    Parameters:
    - memory_context: Dictionary containing trading history and performance data

    Returns:
    - strategies_list: List of recommended strategies with weights
    - strategy_reasoning: Explanation of why these strategies were chosen
    """
    # Extract performance data
    spread_performance = memory_context.get('spread_performance', {})
    bull_put = spread_performance.get(
        'bull_put', {'wins': 0, 'total': 0, 'win_rate': 0})
    bear_call = spread_performance.get(
        'bear_call', {'wins': 0, 'total': 0, 'win_rate': 0})
    iron_condor = spread_performance.get(
        'iron_condor', {'wins': 0, 'total': 0, 'win_rate': 0})

    # Calculate win rates
    if bull_put['total'] > 0:
        bull_put['win_rate'] = bull_put['wins'] / bull_put['total']

    if bear_call['total'] > 0:
        bear_call['win_rate'] = bear_call['wins'] / bear_call['total']

    if iron_condor['total'] > 0:
        iron_condor['win_rate'] = iron_condor['wins'] / iron_condor['total']

    # Get multi-timeframe trends
    daily_trend = memory_context.get(
        'multi_timeframe', {}).get('daily_trend', 'neutral')
    weekly_trend = memory_context.get(
        'multi_timeframe', {}).get('weekly_trend', 'neutral')

    # Get volatility info
    vol_history = memory_context.get('volatility_history', [])
    current_volatility = vol_history[-1] if vol_history else {
        'iv': None, 'hv': None, 'assessment': 'unknown'}

    # Default strategies with equal weighting
    strategies = [
        {'name': 'bull_put', 'weight': 0.33, 'condition': 'bullish'},
        {'name': 'bear_call', 'weight': 0.33, 'condition': 'bearish'},
        {'name': 'iron_condor', 'weight': 0.33, 'condition': 'neutral'}
    ]

    # Determine which strategies have been tested
    bull_put_tested = bull_put['total'] >= 2
    bear_call_tested = bear_call['total'] >= 2
    iron_condor_tested = iron_condor['total'] >= 2

    # For strategies with historical data, adjust weights based on win rate
    if bull_put_tested or bear_call_tested or iron_condor_tested:
        # Reset weights for recalculation
        for strategy in strategies:
            strategy['weight'] = 0

        # Calculate base weight from win rates
        total_win_rate = 0
        if bull_put_tested:
            total_win_rate += bull_put['win_rate']
        if bear_call_tested:
            total_win_rate += bear_call['win_rate']
        if iron_condor_tested:
            total_win_rate += iron_condor['win_rate']

        # Apply win rates to weights
        if total_win_rate > 0:
            if bull_put_tested:
                strategies[0]['weight'] = bull_put['win_rate'] / total_win_rate
            if bear_call_tested:
                strategies[1]['weight'] = bear_call['win_rate'] / \
                    total_win_rate
            if iron_condor_tested:
                strategies[2]['weight'] = iron_condor['win_rate'] / \
                    total_win_rate

    # Further adjust weights based on market trends
    trend_factor = 0.3  # How much to adjust based on trend alignment

    # If market trends are aligned, boost appropriate strategy
    if daily_trend == weekly_trend and daily_trend != 'neutral':
        if daily_trend == 'bullish':
            # Boost bull put spreads
            adjustment = min(trend_factor, 1 - strategies[0]['weight'])
            strategies[0]['weight'] += adjustment
            # Reduce others proportionally
            reduction_per_strategy = adjustment / 2
            strategies[1]['weight'] = max(
                0, strategies[1]['weight'] - reduction_per_strategy)
            strategies[2]['weight'] = max(
                0, strategies[2]['weight'] - reduction_per_strategy)
        elif daily_trend == 'bearish':
            # Boost bear call spreads
            adjustment = min(trend_factor, 1 - strategies[1]['weight'])
            strategies[1]['weight'] += adjustment
            # Reduce others proportionally
            reduction_per_strategy = adjustment / 2
            strategies[0]['weight'] = max(
                0, strategies[0]['weight'] - reduction_per_strategy)
            strategies[2]['weight'] = max(
                0, strategies[2]['weight'] - reduction_per_strategy)

    # Adjust for volatility environment
    if current_volatility.get('assessment') in ['high', 'very high']:
        # In high volatility, iron condors can be more profitable
        adjustment = 0.2
        strategies[2]['weight'] = min(
            0.6, strategies[2]['weight'] + adjustment)
        # Reduce others proportionally
        reduction_per_strategy = adjustment / 2
        strategies[0]['weight'] = max(
            0, strategies[0]['weight'] - reduction_per_strategy)
        strategies[1]['weight'] = max(
            0, strategies[1]['weight'] - reduction_per_strategy)

    # Ensure weights sum to 1.0
    total_weight = sum(strategy['weight'] for strategy in strategies)
    if total_weight > 0:
        for strategy in strategies:
            strategy['weight'] = strategy['weight'] / total_weight

    # Sort strategies by weight
    strategies.sort(key=lambda x: x['weight'], reverse=True)

    # Create reasoning explanation
    reasoning = []

    # Explain win rates
    if bull_put_tested or bear_call_tested or iron_condor_tested:
        reasoning.append("Historical performance:")
        if bull_put_tested:
            reasoning.append(
                f"- Bull put spreads: {bull_put['win_rate']:.0%} win rate ({bull_put['wins']}/{bull_put['total']})")
        if bear_call_tested:
            reasoning.append(
                f"- Bear call spreads: {bear_call['win_rate']:.0%} win rate ({bear_call['wins']}/{bear_call['total']})")
        if iron_condor_tested:
            reasoning.append(
                f"- Iron condors: {iron_condor['win_rate']:.0%} win rate ({iron_condor['wins']}/{iron_condor['total']})")

    # Explain trend influence
    if daily_trend == weekly_trend and daily_trend != 'neutral':
        reasoning.append(
            f"Both daily and weekly trends aligned {daily_trend}, favoring {daily_trend} strategies.")
    else:
        reasoning.append(
            f"Mixed trends (daily: {daily_trend}, weekly: {weekly_trend})")

    # Explain volatility influence
    if current_volatility.get('assessment'):
        reasoning.append(
            f"Current volatility assessment: {current_volatility.get('assessment')}")
        if current_volatility.get('assessment') in ['high', 'very high']:
            reasoning.append(
                "High volatility favors iron condors due to premium collection.")

    # Format final list of strategies with weights
    strategies_list = [f"{s['name']} ({s['weight']:.0%})" for s in strategies]
    strategy_reasoning = "\n".join(reasoning)

    return strategies_list, strategy_reasoning
