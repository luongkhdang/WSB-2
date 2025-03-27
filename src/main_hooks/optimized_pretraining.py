"""
Optimized pretraining implementation for WSB trading system.

This module provides an enhanced pretraining process that:
1. Uses batch processing to maximize each API call
2. Manages API rate limits efficiently
3. Combines analysis and reflection in single calls
4. Implements a sliding window approach
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import json
import math
import pandas as pd

# Import utilities
from src.gemini.hooks.enhanced_pretraining_prompt import (
    get_enhanced_pretraining_prompt,
    get_enhanced_summary_prompt,
    get_enhanced_batch_prompt,
    parse_enhanced_prediction,
    parse_pattern_storage,
    parse_reliability_update
)
from src.main_utilities.data_processor import format_stock_data_for_analysis
from src.main_utilities.file_operations import save_pretraining_results, get_historical_market_context

logger = logging.getLogger(__name__)

# Constants for API rate limiting
MAX_REQUESTS_PER_MINUTE = 30
# Time between requests in seconds - set to a shorter interval to maximize usage
REQUEST_INTERVAL = 2  # 2 second interval allows for 30 requests per minute
MAX_CONSECUTIVE_REQUESTS = 25  # Leave buffer for other system activities

# Track request timestamps to manage rate limiting
_request_timestamps = []


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
    - prompt_type: Type of prompt (first_day, batch, summary)

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

        # Send the full response as a file since it might be too long for a message
        # Unfortunately we can't do this directly with the Discord webhook API
        # We'll send chunks of the full response in separate messages

        # Determine if we need to split the response
        MAX_CHUNK_SIZE = 1800  # Discord has a 2000 char limit, leaving some room for formatting

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


def optimized_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    ticker: str,
    start_date=None,
    end_date=None,
    save_results=True,
    callback=None,
    discord_client=None  # Add discord_client parameter
):
    """
    An optimized implementation of the pretraining analyzer that uses batching and
    rate limit management to maximize the value of each API call.

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - ticker: Stock symbol to analyze
    - start_date: Starting date for pretraining (defaults to 5 days before end_date)
    - end_date: End date for pretraining (defaults to yesterday)
    - save_results: Whether to save pretraining results to disk
    - callback: Optional callback function to receive each analysis result
    - discord_client: Optional Discord client for sending responses to webhook

    Returns:
    - Dictionary containing pretraining results and context for future analysis
    """
    logger.info(f"Starting optimized pretraining for {ticker}...")
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

        if not start_date:
            start_date = end_date - timedelta(days=5)  # 5 days before end_date
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Analysis period: {start_date_str} to {end_date_str}")

        # Initialize results container
        pretraining_results = []

        # Initialize memory context with enhanced structure
        memory_context = {
            "ticker": ticker,
            "successful_patterns": [],
            "lessons_learned": [],
            "weight_adjustments": [],
            "prediction_accuracy": {},
            # New pattern recognition structures
            "pattern_library": {
                "head_and_shoulders": [],
                "double_top": [],
                "double_bottom": [],
                "triangle": [],
                "flag": [],
                "wedge": [],
                "channel": [],
                "rectangle": [],
                "consolidation": []
            },
            "key_levels": {
                "support": [],
                "resistance": [],
                "pivot_points": []
            },
            "pattern_reliability": {
                # Track reliability of different pattern types
                "head_and_shoulders": {"correct": 0, "total": 0, "accuracy": 0},
                "double_top": {"correct": 0, "total": 0, "accuracy": 0},
                "double_bottom": {"correct": 0, "total": 0, "accuracy": 0},
                "triangle": {"correct": 0, "total": 0, "accuracy": 0},
                "flag": {"correct": 0, "total": 0, "accuracy": 0},
                "wedge": {"correct": 0, "total": 0, "accuracy": 0}
            },
            "multi_timeframe": {
                "daily_patterns": [],
                "weekly_patterns": [],
                "timeframe_confluence": [],
                "weekly_data_points": 0,
                "weekly_trend": "neutral"
            }
        }

        # STEP 1: Get historical data for context with enhanced lookback
        logger.info(f"STEP 1: Getting historical data for {ticker}")

        # Get market data dir
        market_data_dir = pretraining_dir.parent / \
            "market-data" if hasattr(pretraining_dir,
                                     'parent') else "data-source/market-data"

        # Enhanced progressive lookback strategy with multiple fallbacks
        historical_data = None
        quality_warning = None

        # Try increasingly aggressive lookback periods
        lookback_periods = [
            100,   # Initial lookback (100 days)
            200,   # First fallback (200 days)
            365,   # Second fallback (1 year)
            730,   # Third fallback (2 years)
            1095   # Final fallback (3 years)
        ]

        for lookback_days in lookback_periods:
            historical_start_date = start_date - timedelta(days=lookback_days)
            historical_start_str = historical_start_date.strftime("%Y-%m-%d")

            logger.info(
                f"Attempting to get historical data with {lookback_days} day lookback")

            try:
                # Get historical data with current lookback period
                temp_historical_data = yfinance_client.get_historical_data(
                    ticker,
                    start=historical_start_str,
                    end=end_date_str,
                    interval="1d"
                )

                # Check if we got enough data points (at least 60)
                if len(temp_historical_data) >= 60:
                    historical_data = temp_historical_data
                    if lookback_days > 100:
                        quality_warning = f"Using extended historical data ({len(historical_data)} days with {lookback_days} day lookback) for better pattern recognition."
                    else:
                        quality_warning = None
                    logger.info(
                        f"Successfully retrieved {len(historical_data)} days of data with {lookback_days} day lookback")
                    break
                else:
                    logger.warning(
                        f"Insufficient historical data ({len(temp_historical_data)} points) with {lookback_days} day lookback. Trying longer period.")
                    # Store the data anyway in case all attempts fail
                    if historical_data is None or len(temp_historical_data) > len(historical_data):
                        historical_data = temp_historical_data
            except Exception as e:
                logger.error(
                    f"Error getting historical data with {lookback_days} day lookback: {e}")

        # Check if we have any data at all
        if historical_data is None or len(historical_data) == 0:
            logger.error(
                f"Failed to retrieve any historical data for {ticker}")
            return {"error": f"Failed to get historical data for {ticker}: No data available"}

        # If we have data but less than ideal amount, warn but proceed
        if len(historical_data) < 60:
            logger.warning(
                f"Proceeding with limited historical data ({len(historical_data)} points), which may affect pattern recognition quality.")
            quality_warning = f"Limited historical data available ({len(historical_data)} days). Pattern recognition may be impaired."

            # Special handling for extremely limited data
            if len(historical_data) < 30:
                logger.warning(
                    "Extremely limited data. Disabling certain pattern recognition features.")
                quality_warning = f"SEVERELY limited historical data ({len(historical_data)} days). Most pattern recognition features disabled."

        # Get weekly data for multi-timeframe analysis with the same extended lookback
        try:
            weekly_data = yfinance_client.get_historical_data(
                ticker,
                start=historical_start_str,
                end=end_date_str,
                interval="1wk"
            )

            if len(weekly_data) > 0:
                logger.info(
                    f"Got {len(weekly_data)} weeks of historical data for multi-timeframe analysis")
                memory_context["multi_timeframe"]["weekly_data_points"] = len(
                    weekly_data)
            else:
                logger.warning(
                    "No weekly data available for multi-timeframe analysis")
        except Exception as e:
            logger.warning(f"Error getting weekly data: {e}")
            weekly_data = pd.DataFrame()  # Empty dataframe as fallback

        # STEP 2: Get market context data for all dates in the period
        logger.info(
            f"STEP 2: Getting market context data for {start_date_str} to {end_date_str}")

        try:
            # Get market context for all dates in the analysis period
            # Since get_historical_market_context only accepts one date, we'll loop through dates
            market_context = {}
            current_date = start_date
            while current_date <= end_date:
                current_date_str = current_date.strftime("%Y-%m-%d")
                try:
                    # Get context for this specific date
                    day_context = get_historical_market_context(
                        market_data_dir, current_date_str)
                    if day_context:
                        market_context[current_date_str] = day_context
                except Exception as day_error:
                    logger.warning(
                        f"Error getting market context for {current_date_str}: {day_error}")

                # Move to next day
                current_date += timedelta(days=1)

            logger.info(f"Got market context for {len(market_context)} dates")
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            market_context = {}  # Use empty dict if not available

        # STEP 3: Initialize directory structure for this ticker
        ticker_dir = pretraining_dir / ticker
        ticker_dir.mkdir(exist_ok=True)

        # STEP 4: Generate batched day-by-day analysis with rate limiting
        logger.info(f"STEP 4: Generating batched analysis for {ticker}")

        # Get date range for pretraining (inclusive of start/end dates)
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)

        # Get data for the entire period (for actual outcomes)
        all_period_data = yfinance_client.get_historical_data(
            ticker,
            start=start_date_str,
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d"
        )

        # Data quality check for analysis period
        if len(all_period_data) == 0:
            logger.error(
                f"No data available for {ticker} during analysis period")
            return {"error": f"No data available for {ticker} during analysis period"}

        # Filter out weekends and holidays when no trading occurred
        date_range = [date for date in date_range if date.strftime(
            "%Y-%m-%d") in all_period_data.index.strftime("%Y-%m-%d").tolist()]

        # Calculate number of batches
        num_days = len(date_range)
        num_batches = math.ceil(num_days / BATCH_SIZE)
        logger.info(
            f"Processing {num_days} trading days in {num_batches} batches of up to {BATCH_SIZE} days each")

        # Process data in batches to maximize API calls
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * BATCH_SIZE
            batch_end_idx = min(batch_start_idx + BATCH_SIZE, num_days)
            batch_dates = date_range[batch_start_idx:batch_end_idx]

            logger.info(
                f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_dates)} days")

            # Prepare batch data
            batch_data = []
            batch_market_contexts = {}

            for analysis_date in batch_dates:
                current_date_str = analysis_date.strftime("%Y-%m-%d")

                # Filter data up to this date
                cutoff_date = (analysis_date + timedelta(days=1)
                               ).strftime("%Y-%m-%d")
                current_data = all_period_data[all_period_data.index < cutoff_date]

                # Skip days with no data
                if len(current_data) == 0 or current_date_str not in current_data.index.strftime("%Y-%m-%d").tolist():
                    logger.warning(
                        f"No data available for {ticker} on {current_date_str}, skipping")
                    continue

                # Get the current day's data
                try:
                    current_day_data = current_data[current_data.index.strftime(
                        "%Y-%m-%d") == current_date_str]
                    if len(current_day_data) == 0:
                        logger.warning(
                            f"No data specifically for {current_date_str}, using most recent available")
                        current_day_data = current_data.iloc[-1:]
                except Exception as e:
                    logger.error(f"Error extracting current day data: {e}")
                    logger.warning(f"Using most recent available data instead")
                    current_day_data = current_data.iloc[-1:]

                # Format data for analysis with multi-timeframe support
                formatted_data = format_stock_data_for_analysis(
                    current_data, ticker, current_date_str)

                # Add weekly data if available
                if len(weekly_data) > 0:
                    # Get the current week's data (using the date to find the matching week)
                    week_end = analysis_date + \
                        timedelta(days=6-analysis_date.weekday()
                                  )  # Find Friday
                    # Find previous Saturday
                    week_start = week_end - timedelta(days=6)

                    weekly_slice = weekly_data[
                        (weekly_data.index >= week_start.strftime("%Y-%m-%d")) &
                        (weekly_data.index <= week_end.strftime("%Y-%m-%d"))
                    ]

                    if len(weekly_slice) > 0:
                        # Format weekly data
                        weekly_formatted = {
                            'open': float(weekly_slice['Open'].iloc[0]),
                            'high': float(weekly_slice['High'].iloc[0]),
                            'low': float(weekly_slice['Low'].iloc[0]),
                            'close': float(weekly_slice['Close'].iloc[0]),
                            'volume': int(weekly_slice['Volume'].iloc[0]),
                        }

                        # Add technical indicators if available
                        if 'EMA_9' in weekly_slice.columns:
                            weekly_formatted['ema_9'] = float(
                                weekly_slice['EMA_9'].iloc[0])
                        if 'EMA_21' in weekly_slice.columns:
                            weekly_formatted['ema_21'] = float(
                                weekly_slice['EMA_21'].iloc[0])
                        if 'SMA_50' in weekly_slice.columns:
                            weekly_formatted['sma_50'] = float(
                                weekly_slice['SMA_50'].iloc[0])
                        if 'SMA_200' in weekly_slice.columns:
                            weekly_formatted['sma_200'] = float(
                                weekly_slice['SMA_200'].iloc[0])

                        # Add weekly data to formatted data
                        formatted_data['weekly_data'] = weekly_formatted

                # Add quality warning if applicable
                if quality_warning:
                    formatted_data["quality_warning"] = quality_warning

                # Get market context for current day
                current_market_context = market_context.get(
                    current_date_str, {"spy_trend": "neutral"})

                # Add to batch
                batch_data.append(formatted_data)
                batch_market_contexts[current_date_str] = current_market_context

            # Skip empty batches
            if not batch_data:
                logger.warning(
                    f"Batch {batch_idx+1} has no valid data, skipping")
                continue

            # For first batch, use single-day analysis to establish baseline
            if batch_idx == 0 and len(batch_data) > 0:
                # Process first day separately to establish baseline
                first_day = batch_data[0]
                first_date = first_day.get('date', 'unknown')
                first_market_context = batch_market_contexts.get(
                    first_date, {"spy_trend": "neutral"})

                logger.info(
                    f"Processing first day ({first_date}) separately to establish baseline")

                # Create prompt for first day
                first_prompt = get_enhanced_pretraining_prompt(
                    first_day,
                    first_market_context,
                    memory_context
                )

                # Call the AI Analyzer to generate the first day analysis
                start_request = time.time()
                try:
                    first_day_analysis = gemini_client.generate_text(
                        first_prompt, temperature=0.1, prompt_type="first_day")
                except Exception as e:
                    logger.error(f"Error generating first day analysis: {e}")
                    return {
                        "error": f"Failed to generate first day analysis: {str(e)}",
                        "processing_time": time.time() - start_time
                    }

                # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

                # Create a result for the first day analysis
                first_day_result = {
                    "ticker": ticker,
                    "date": first_date,
                    "type": "first_day",
                    "analysis_type": "daily",
                    "full_analysis": first_day_analysis
                }

                # Call the callback if provided
                if callback:
                    callback(first_day_result)

                # Add to the list of analyses
                pretraining_results.append(first_day_result)

                # Remove first day from batch if it contains more days
                if len(batch_data) > 1:
                    batch_data = batch_data[1:]

            # Process remaining days in batch
            if batch_data:
                logger.info(f"Processing batch of {len(batch_data)} days")

                # Create prompt for batch processing
                batch_prompt = get_enhanced_batch_prompt(
                    ticker,
                    batch_data,
                    batch_market_contexts,
                    memory_context
                )

                # Get analysis from Gemini
                start_request = time.time()
                batch_analysis_text = gemini_client.generate_text(
                    batch_prompt, temperature=0.3, prompt_type="batch_analysis"
                )

                # Process batch results - split by date sections
                date_analyses = []
                current_date = None
                current_analysis = ""

                for line in batch_analysis_text.split('\n'):
                    # Check for date header
                    if line.strip().startswith("DATE:"):
                        # Save previous analysis if it exists
                        if current_date and current_analysis:
                            date_analyses.append({
                                "date": current_date,
                                "analysis": current_analysis
                            })

                        # Extract new date
                        date_part = line.split("DATE:", 1)[1].strip()
                        current_date = date_part
                        current_analysis = line + "\n"
                    elif current_date is not None:
                        # Add line to current analysis
                        current_analysis += line + "\n"

                # Add the last analysis
                if current_date and current_analysis:
                    date_analyses.append({
                        "date": current_date,
                        "analysis": current_analysis
                    })

                # Extract sequence analysis and memory update
                sequence_section = ""
                memory_update_section = ""

                if "SEQUENCE ANALYSIS:" in batch_analysis_text:
                    sequence_parts = batch_analysis_text.split(
                        "SEQUENCE ANALYSIS:", 1)
                    if len(sequence_parts) > 1:
                        if "MEMORY UPDATE:" in sequence_parts[1]:
                            sequence_section, memory_update_section = sequence_parts[1].split(
                                "MEMORY UPDATE:", 1)
                        else:
                            sequence_section = sequence_parts[1]

                # Process individual date analyses
                for date_analysis in date_analyses:
                    analysis_date = date_analysis["date"]
                    analysis_text = date_analysis["analysis"]

                    # Extract prediction
                    prediction = parse_enhanced_prediction(analysis_text)

                    # Find matching stock data
                    matching_data = next(
                        (d for d in batch_data if d.get('date') == analysis_date), None)

                    if matching_data:
                        # Create result object
                        result = {
                            "ticker": ticker,
                            "date": analysis_date,
                            "price": matching_data.get('current_price', 0),
                            "analysis_type": "batch_analysis",
                            "trend": "bullish" if prediction.get('direction') == 'bullish' else
                                     "bearish" if prediction.get(
                                         'direction') == 'bearish' else "neutral",
                            "next_day_prediction": prediction,
                            "full_analysis": analysis_text
                        }

                        # Add to results
                        pretraining_results.append(result)

                        # Call the callback if provided
                        if callback:
                            callback(result)

                # Extract new patterns from memory update
                patterns = parse_pattern_storage(memory_update_section)
                if patterns:
                    for pattern in patterns:
                        pattern_type = pattern.get('pattern_type', '').lower()
                        if pattern_type in memory_context['pattern_library']:
                            memory_context['pattern_library'][pattern_type].append(
                                pattern)

                # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

                # Generate a summary analysis if we have enough data
                if len(pretraining_results) > 1:
                    logger.info(f"Generating summary analysis for {ticker}")

                    try:
                        # Create prompt for summary
                        summary_prompt = get_enhanced_summary_prompt(
                            ticker,
                            {"start_date": start_date_str,
                                "end_date": end_date_str},
                            pretraining_results,
                            memory_context
                        )

                        # Get summary from Gemini
                        start_request = time.time()
                        summary_text = gemini_client.generate_text(
                            summary_prompt, temperature=0.4, prompt_type="summary"
                        )

                        # Create a result object for the summary
                        summary = {
                            "ticker": ticker,
                            "type": "summary",
                            "analysis_type": "multi_timeframe",
                            "start_date": start_date_str,
                            "end_date": end_date_str,
                            "full_analysis": summary_text
                        }

                        # Add to results list
                        pretraining_results.append(summary)

                        # Call the callback if provided
                        if callback:
                            callback(summary)

                        # The rate limiting is now handled by manage_rate_limit() in tracked_generate_text

                    except Exception as e:
                        logger.error(f"Error in final summary: {e}")
                        logger.exception(e)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create result dictionary
        result = {
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "processing_time": processing_time,
            "analyses_count": len(pretraining_results),
            "results": pretraining_results
        }

        # Save results to disk if requested
        if save_results:
            # Generate context for future use
            context = {
                "ticker": ticker,
                "pretraining_period": f"{start_date_str} to {end_date_str}",
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analyses_count": len(pretraining_results),
                "has_reflections": any(r.get("analysis_type", "").startswith("reflection") for r in pretraining_results),
                "has_summary": any(r.get("type", "") == "summary" for r in pretraining_results),
                "memory_context": memory_context  # Include memory context for future pretraining
            }

            # If there's a summary, extract predictions
            for r in pretraining_results:
                if r.get("type") == "summary":
                    for horizon in ["next_day", "next_week", "next_month"]:
                        if f"{horizon}_prediction" in r:
                            context[f"{horizon}_direction"] = r[f"{horizon}_prediction"].get(
                                "direction", "neutral")
                            context[f"{horizon}_magnitude"] = r[f"{horizon}_prediction"].get(
                                "magnitude", 0)
                            context[f"{horizon}_confidence"] = r[f"{horizon}_prediction"].get(
                                "confidence", 0)

            # Save to disk
            result_path = save_pretraining_results(
                pretraining_dir, ticker, result, context)
            logger.info(f"Saved pretraining results to {result_path}")

        logger.info(
            f"Pretraining completed for {ticker} with {len(pretraining_results)} results in {processing_time:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Error in pretraining: {e}")
        logger.exception(e)
        return {"error": str(e)}
    finally:
        # Restore the original generate_text method
        gemini_client.generate_text = original_generate_text


def optimized_batch_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    tickers,
    discord_client=None,  # Add discord_client parameter
    **kwargs
):
    """
    Run optimized pretraining for multiple tickers efficiently.

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - tickers: List of stock symbols to analyze
    - discord_client: Optional Discord client for response tracking
    - **kwargs: Additional arguments to pass to optimized_pretrain_analyzer

    Returns:
    - Dictionary mapping tickers to pretraining results
    """
    logger.info(
        f"Starting optimized batch pretraining for {len(tickers)} tickers...")
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
            ticker_result = optimized_pretrain_analyzer(
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
