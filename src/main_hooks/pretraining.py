#!/usr/bin/env python3
"""
Pretraining Module (src/main_hooks/pretraining.py)
--------------------------------------------------
Core functionality for pretraining the AI analyzer on historical stock data.

Functions:
  - pretrain_analyzer - Trains model on historical data for a specific ticker
  - batch_pretrain_analyzer - Processes multiple tickers in batch
  - evaluate_pretraining_predictions - Evaluates accuracy of past predictions

Dependencies:
  - src.main_utilities.analysis_parser - For parsing analysis results
  - src.main_utilities.data_processor - For formatting stock data
  - src.main_utilities.file_operations - For file I/O operations
  - YFinance client - For historical stock data
  - Gemini client - For AI analysis

Used by:
  - main.py for pretraining the AI model on historical data
"""

import logging
import time
from datetime import datetime, timedelta
import json

# Import utilities
from src.main_utilities.analysis_parser import parse_stock_analysis, parse_reflection
from src.main_utilities.data_processor import format_stock_data_for_analysis, extract_next_day_prediction
from src.main_utilities.file_operations import save_pretraining_results, get_historical_market_context

logger = logging.getLogger(__name__)


def pretrain_analyzer(yfinance_client, gemini_client, pretraining_dir, ticker, pretraining_prompt_hook, start_date=None, end_date=None, save_results=True, callback=None):
    """
    Pretrain the AI Analyzer on historical stock data in a staged, reflective sequence.

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - ticker: Stock symbol to analyze
    - pretraining_prompt_hook: Function to generate pretraining prompt
    - start_date: Starting date for pretraining (defaults to 5 days before end_date)
    - end_date: End date for pretraining (defaults to yesterday)
    - save_results: Whether to save pretraining results to disk
    - callback: Optional callback function to receive each analysis result

    Returns:
    - Dictionary containing pretraining results and context for use in future analysis
    """
    logger.info(f"Starting pretraining for {ticker}...")
    start_time = time.time()

    try:
        # Set up dates
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)  # yesterday
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if start_date is None:
            start_date = end_date - timedelta(days=5)  # 5 days before end date
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Validate dates
        if start_date >= end_date:
            logger.error(
                f"Start date {start_date} must be before end date {end_date}")
            return {"error": "Invalid date range"}

        # Format dates for logging
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        logger.info(f"Pretraining period: {start_date_str} to {end_date_str}")

        # Initialize pretraining results
        pretraining_results = []

        # Memory context to maintain across iterations
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

        # Get initial market context
        market_data_dir = pretraining_dir.parent / \
            "market-data" if hasattr(pretraining_dir,
                                     'parent') else "data-source/market-data"

        # STEP 1: Get historical data for context with multi-timeframe support
        logger.info(f"STEP 1: Getting historical data for {ticker}")

        # Get daily data with extended lookback for pattern recognition
        daily_data = yfinance_client.get_historical_data(
            ticker,
            start=(start_date - timedelta(days=100)).strftime("%Y-%m-%d"),
            end=start_date_str,
            interval="1d"
        )

        if len(daily_data) == 0:
            logger.error(f"Failed to get historical data for {ticker}")
            return {"error": f"No data available for {ticker}"}

        # Data quality check - ensure we have at least 60 days of historical data for pattern recognition
        if len(daily_data) < 60:
            logger.warning(
                f"Limited historical data for {ticker} - only {len(daily_data)} days available (minimum 60 recommended for pattern recognition)")
            # Try to get more historical data with a longer lookback
            daily_data = yfinance_client.get_historical_data(
                ticker,
                start=(start_date - timedelta(days=200)).strftime("%Y-%m-%d"),
                end=start_date_str,
                interval="1d"
            )
            if len(daily_data) < 60:
                logger.warning(
                    f"Still insufficient historical data for optimal pattern recognition: {len(daily_data)} < 60 days")
                if len(daily_data) < 10:
                    logger.error(
                        f"Insufficient historical data for {ticker} - minimum 10 days required")
                    return {"error": f"Insufficient historical data for {ticker} - minimum 10 days required"}

        # STEP 1b: Get weekly data for multi-timeframe analysis
        logger.info(
            f"Getting weekly data for multi-timeframe analysis of {ticker}")
        try:
            weekly_data = yfinance_client.get_historical_data(
                ticker,
                start=(start_date - timedelta(days=365)
                       ).strftime("%Y-%m-%d"),  # 1 year for weekly
                end=start_date_str,
                interval="1wk"
            )

            has_weekly_data = len(weekly_data) >= 10  # Need at least 10 weeks

            if has_weekly_data:
                logger.info(
                    f"Successfully retrieved {len(weekly_data)} weeks of data for {ticker}")

                # Format weekly data for analysis
                weekly_formatted = format_stock_data_for_analysis(
                    weekly_data, ticker, f"{start_date_str}_weekly")

                if "error" not in weekly_formatted:
                    # Store information about weekly timeframe
                    memory_context["multi_timeframe"]["weekly_data_points"] = len(
                        weekly_data)
                    memory_context["multi_timeframe"]["weekly_trend"] = "neutral"

                    # Determine weekly trend
                    if weekly_formatted.get("ema9", 0) > 0 and weekly_formatted.get("ema21", 0) > 0:
                        if weekly_formatted["current_price"] > weekly_formatted["ema9"] > weekly_formatted["ema21"]:
                            memory_context["multi_timeframe"]["weekly_trend"] = "bullish"
                        elif weekly_formatted["current_price"] < weekly_formatted["ema9"] < weekly_formatted["ema21"]:
                            memory_context["multi_timeframe"]["weekly_trend"] = "bearish"
            else:
                logger.warning(
                    f"Insufficient weekly data for {ticker}: {len(weekly_data)} weeks")
                memory_context["multi_timeframe"]["weekly_data_points"] = 0

        except Exception as e:
            logger.error(f"Error retrieving weekly data for {ticker}: {e}")
            has_weekly_data = False
            memory_context["multi_timeframe"]["weekly_data_points"] = 0

        # Update historical_data to be our daily data (for compatibility with rest of function)
        historical_data = daily_data

        # Check data quality - ensure all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [
            col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            missing_cols_str = ', '.join(missing_columns)
            logger.error(
                f"Missing required data columns for {ticker}: {missing_cols_str}")
            return {"error": f"Missing required data columns: {missing_cols_str}"}

        # Check for null values in critical columns
        if historical_data['Close'].isnull().any().item():
            logger.error(
                f"Data quality issue: NULL values found in Close prices for {ticker}")
            # Fill null values with preceding values to prevent analysis failures
            historical_data = historical_data.fillna(method='ffill')
            # Still alert about the issue in the result
            quality_warning = f"Warning: Some price data was missing and filled with preceding values"
        else:
            quality_warning = None

        # STEP 2: Get market context from history
        logger.info(f"STEP 2: Getting market context for analysis period")
        market_context = get_historical_market_context(
            market_data_dir, start_date_str)

        # Ensure market context exists
        if not market_context or (isinstance(market_context, dict) and "error" in market_context):
            logger.warning(
                "Could not retrieve historical market context, using default")
            market_context = {"spy_trend": "neutral"}

        # STEP 3: Initialize directory structure for this ticker
        ticker_dir = pretraining_dir / ticker
        ticker_dir.mkdir(exist_ok=True)

        # STEP 4: Generate day-by-day analysis + reflection sequence
        logger.info(f"STEP 4: Generating day-by-day analysis for {ticker}")

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
            # Add extra day to ensure end date is included
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

        # Iterate through each day in the date range
        previous_analysis = None
        previous_day_data = None

        for day_index, analysis_date in enumerate(date_range):
            current_date_str = analysis_date.strftime("%Y-%m-%d")
            logger.info(
                f"Analyzing {ticker} for {current_date_str} (Day {day_index+1} of {len(date_range)})")

            try:
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

                # Format data for analysis
                formatted_data = format_stock_data_for_analysis(
                    current_data, ticker, current_date_str)

                # Data quality check - ensure we have prices and technical indicators
                if not formatted_data or "error" in formatted_data:
                    logger.warning(
                        f"Data quality issue for {ticker} on {current_date_str}: {formatted_data.get('error', 'Unknown error')}")
                    # Try with alternative formatting if possible
                    if len(current_data) > 0:
                        # Basic manual formatting as fallback
                        last_row = current_data.iloc[-1]
                        formatted_data = {
                            "ticker": ticker,
                            "date": current_date_str,
                            "current_price": float(last_row['Close']),
                            "volume": float(last_row['Volume']) if 'Volume' in last_row else 0,
                            "data_points": len(current_data)
                        }
                        logger.info(
                            f"Using fallback data formatting for {ticker} on {current_date_str}")
                    else:
                        logger.error(
                            f"Cannot proceed with analysis for {ticker} on {current_date_str} - insufficient data")
                        continue

                # STEP 4b: Detect patterns in price data if we have enough data points
                if formatted_data.get("data_points", 0) >= 60 and formatted_data.get("has_pattern_data", False):
                    try:
                        # Import pattern detection module
                        from src.pattern_recognition.detection import detect_patterns

                        logger.info(
                            f"Running pattern detection for {ticker} on {current_date_str}")
                        detected_patterns = detect_patterns(current_data)

                        # Add detected patterns to formatted data
                        if detected_patterns:
                            pattern_types = []
                            pattern_details = []

                            # Extract pattern info
                            for pattern_name, pattern_info in detected_patterns.items():
                                if pattern_info.get("found", False):
                                    pattern_types.append(pattern_name)
                                    pattern_details.append(
                                        pattern_info.get("details", ""))

                            # Add to formatted data
                            if pattern_types:
                                formatted_data["pattern_type"] = ", ".join(
                                    pattern_types)
                                formatted_data["pattern_recognition"] = "\n".join(
                                    pattern_details)
                                formatted_data["has_detected_patterns"] = True
                                logger.info(
                                    f"Detected patterns for {ticker}: {formatted_data['pattern_type']}")
                        else:
                            formatted_data["has_detected_patterns"] = False
                            logger.debug(
                                f"No patterns detected for {ticker} on {current_date_str}")

                        # Create chart visualization if patterns were found
                        if detected_patterns and any(p.get("found", False) for p in detected_patterns.values()):
                            try:
                                from src.pattern_recognition.visualization import visualize_pattern

                                # Get the first detected pattern for visualization
                                for pattern_name, pattern_info in detected_patterns.items():
                                    if pattern_info.get("found", False):
                                        # Generate visualization
                                        chart_path = ticker_dir / \
                                            f"{ticker}_{current_date_str}_{pattern_name}.png"
                                        visualize_pattern(
                                            current_data, pattern_info, str(chart_path))
                                        logger.info(
                                            f"Generated pattern visualization: {chart_path}")
                                        break
                            except Exception as viz_error:
                                logger.error(
                                    f"Error visualizing pattern: {viz_error}")

                    except ImportError as e:
                        logger.warning(
                            f"Pattern detection module not available: {e}")
                    except Exception as e:
                        logger.error(f"Error during pattern detection: {e}")
                else:
                    logger.debug(
                        f"Skipping pattern detection for {ticker}: insufficient data points ({formatted_data.get('data_points', 0)})")
                    formatted_data["has_detected_patterns"] = False

                # Generate market context with weight adjustments from previous reflections
                if day_index > 0 and previous_analysis and "reflection" in previous_analysis.get("analysis_type", ""):
                    # Extract weight adjustment from previous reflection
                    weight_adjustment = previous_analysis.get(
                        "weight_adjustment", "")
                    if weight_adjustment:
                        memory_context["weight_adjustments"].append(
                            weight_adjustment)

                # Get market context for current day
                current_market_context = market_context.get(
                    current_date_str, {"spy_trend": "neutral"})

                # STEP 4A: Intraday Analysis
                logger.info(
                    f"STEP 4A: Performing intraday analysis for {ticker} on {current_date_str}")

                # Format current day data
                formatted_intraday = format_stock_data_for_analysis(
                    current_day_data, ticker, current_date_str)

                # Skip if formatting failed
                if not formatted_intraday or "error" in formatted_intraday:
                    logger.warning(
                        f"Skipping intraday analysis due to data formatting error: {formatted_intraday.get('error', 'Unknown error')}")
                    continue

                # Add quality warning if applicable
                if quality_warning:
                    formatted_intraday["quality_warning"] = quality_warning

                # Add previous analysis for context if available
                prompt_kwargs = {}
                if previous_analysis and previous_day_data:
                    prompt_kwargs["previous_analysis"] = previous_analysis
                    prompt_kwargs["current_date"] = current_date_str

                # ENHANCED: Generate prompt with continuous learning context
                prompt_kwargs = {
                    'current_date': current_date_str
                }

                if previous_analysis:
                    # Add the continuous learning context if available
                    prompt_kwargs['previous_analysis'] = previous_analysis

                prompt = pretraining_prompt_hook(
                    formatted_intraday,
                    current_market_context,
                    **prompt_kwargs
                )

                # Get analysis from Gemini
                intraday_analysis_text = gemini_client.generate_text(
                    prompt, temperature=0.3)

                # Parse the analysis
                intraday_analysis = parse_stock_analysis(
                    intraday_analysis_text)
                intraday_analysis["ticker"] = ticker
                intraday_analysis["date"] = current_date_str
                intraday_analysis["price"] = formatted_intraday.get(
                    "current_price", 0)
                intraday_analysis["atr_percent"] = formatted_intraday.get(
                    "atr_percent", 0)
                intraday_analysis["analysis_type"] = f"intraday_day{day_index+1}"
                intraday_analysis["full_analysis"] = intraday_analysis_text

                # Extract next day prediction
                next_day_prediction = extract_next_day_prediction(
                    intraday_analysis_text)
                intraday_analysis["next_day_prediction"] = next_day_prediction

                # Add to results
                pretraining_results.append(intraday_analysis)

                # Call the callback if provided
                if callback:
                    callback(intraday_analysis)

                # ENHANCED: Get next day data for reflection if not the last day
                if current_date < end_date:
                    next_date = current_date + timedelta(days=1)
                    next_date_str = next_date.strftime("%Y-%m-%d")

                    # Get data for next day
                    next_day_data = yfinance_client.get_historical_data(
                        ticker,
                        start=next_date_str,
                        end=(next_date + timedelta(days=1)
                             ).strftime("%Y-%m-%d"),
                        interval="1d"
                    )

                    if len(next_day_data) > 0:
                        # Format next day data and create actual outcome context
                        formatted_next_day = format_stock_data_for_analysis(
                            next_day_data, ticker, next_date_str)

                        # Calculate actual outcome metrics
                        actual_price = formatted_next_day.get(
                            "current_price", 0)
                        previous_price = intraday_analysis.get("price", 0)

                        # Prevent division by zero
                        if previous_price > 0 and actual_price > 0:
                            actual_change_pct = (
                                (actual_price - previous_price) / previous_price) * 100

                            # Determine accuracy of prediction
                            predicted_direction = next_day_prediction.get(
                                "direction", "neutral")
                            predicted_magnitude = next_day_prediction.get(
                                "magnitude", 0)

                            actual_direction = "neutral"
                            if actual_change_pct > 0.5:
                                actual_direction = "bullish"
                            elif actual_change_pct < -0.5:
                                actual_direction = "bearish"

                            correct_direction = predicted_direction == actual_direction
                            magnitude_error = abs(
                                predicted_magnitude - abs(actual_change_pct))

                            # Create outcome context
                            outcome_context = {
                                "actual_price": actual_price,
                                "actual_change_pct": actual_change_pct,
                                "actual_direction": actual_direction,
                                "predicted_direction": predicted_direction,
                                "predicted_magnitude": predicted_magnitude,
                                "correct_direction": correct_direction,
                                "magnitude_error": magnitude_error
                            }

                            # ENHANCED: Direct feedback loop - pass both prediction and outcome to Gemini
                            reflection_prompt = pretraining_prompt_hook(
                                formatted_next_day,
                                current_market_context,
                                previous_analysis=intraday_analysis,
                                current_date=next_date_str,
                                is_reflection=True,
                                outcome_context=outcome_context  # Add actual outcome context
                            )

                            # Get reflection from Gemini
                            reflection_text = gemini_client.generate_text(
                                reflection_prompt, temperature=0.4)

                            # Parse reflection
                            reflection = parse_reflection(reflection_text)
                            reflection["ticker"] = ticker
                            reflection["date"] = next_date_str
                            reflection["analysis_type"] = f"reflection_day{day_index+1}"
                            reflection["full_reflection"] = reflection_text
                            reflection["prediction_accuracy"] = 1 if correct_direction else 0
                            reflection["correct_direction"] = correct_direction
                            reflection["magnitude_error"] = magnitude_error

                            # Add to results
                            pretraining_results.append(reflection)

                            # ENHANCED: Update memory context with successful patterns and lessons
                            if correct_direction and magnitude_error < 1.0:
                                # Extract successful pattern with enhanced details
                                successful_pattern = {
                                    "date": current_date_str,
                                    "technical_indicators": {
                                        k: intraday_analysis.get(k) for k in [
                                            "trend", "technical_score", "sentiment_score",
                                            "market_alignment", "rsi", "macd", "bb_width", "adx"
                                        ]
                                    },
                                    "pattern_recognition": intraday_analysis.get("pattern_recognition", ""),
                                    "pattern_type": intraday_analysis.get("pattern_type", ""),
                                    "key_levels": {
                                        "support": intraday_analysis.get("support_levels", []),
                                        "resistance": intraday_analysis.get("resistance_levels", [])
                                    },
                                    "price_data": {
                                        "price": formatted_data.get("current_price", 0),
                                        "volume": formatted_data.get("volume", 0),
                                        "ema9": formatted_data.get("ema9", 0),
                                        "ema21": formatted_data.get("ema21", 0),
                                        "sma50": formatted_data.get("sma50", 0),
                                        "sma200": formatted_data.get("sma200", 0)
                                    },
                                    "advanced_indicators": {
                                        "rsi": formatted_data.get("rsi", 50),
                                        "macd": formatted_data.get("macd", 0),
                                        "macd_signal": formatted_data.get("macd_signal", 0),
                                        "bb_width": formatted_data.get("bb_width", 0),
                                        "adx": formatted_data.get("adx", 0)
                                    },
                                    "prediction": next_day_prediction,
                                    "outcome": outcome_context,
                                    "confidence": next_day_prediction.get("confidence", 0),
                                    "duration_days": 1,  # Default duration
                                    "notes": reflection.get("pattern_insights", "")
                                }

                                # Add to successful patterns list
                                memory_context["successful_patterns"].append(
                                    successful_pattern)

                                # Add to pattern library by type if a pattern was identified
                                pattern_type = intraday_analysis.get(
                                    "pattern_type", "").lower()
                                if pattern_type:
                                    # Handle multiple pattern types (comma-separated)
                                    pattern_types = [
                                        p.strip() for p in pattern_type.split(",")]

                                    for pattern in pattern_types:
                                        # Add to appropriate category if it exists
                                        if pattern in memory_context["pattern_library"]:
                                            memory_context["pattern_library"][pattern].append(
                                                successful_pattern)

                                            # Update pattern reliability stats
                                            if pattern in memory_context["pattern_reliability"]:
                                                memory_context["pattern_reliability"][pattern]["total"] += 1
                                                if correct_direction:
                                                    memory_context["pattern_reliability"][pattern]["correct"] += 1
                                                # Update accuracy percentage
                                                total = memory_context["pattern_reliability"][pattern]["total"]
                                                correct = memory_context["pattern_reliability"][pattern]["correct"]
                                                if total > 0:
                                                    memory_context["pattern_reliability"][pattern]["accuracy"] = (
                                                        correct / total) * 100

                                # Store key levels if present
                                if intraday_analysis.get("support_levels"):
                                    for level in intraday_analysis.get("support_levels"):
                                        if level not in memory_context["key_levels"]["support"]:
                                            memory_context["key_levels"]["support"].append(
                                                level)

                                if intraday_analysis.get("resistance_levels"):
                                    for level in intraday_analysis.get("resistance_levels"):
                                        if level not in memory_context["key_levels"]["resistance"]:
                                            memory_context["key_levels"]["resistance"].append(
                                                level)

                            # Extract lessons learned
                            if "lessons_learned" in reflection and reflection["lessons_learned"]:
                                memory_context["lessons_learned"].extend(
                                    reflection["lessons_learned"])

                            # Extract weight adjustments
                            if "weight_adjustment" in reflection and reflection["weight_adjustment"]:
                                memory_context["weight_adjustments"].append(
                                    reflection["weight_adjustment"])

                            # Update prediction accuracy stats
                            timeframe = "next_day"  # Default to next day
                            if timeframe not in memory_context["prediction_accuracy"]:
                                memory_context["prediction_accuracy"][timeframe] = {
                                    "correct": 0, "total": 0, "errors": []}

                            memory_context["prediction_accuracy"][timeframe]["total"] += 1
                            if correct_direction:
                                memory_context["prediction_accuracy"][timeframe]["correct"] += 1
                            memory_context["prediction_accuracy"][timeframe]["errors"].append(
                                magnitude_error)

                            # Call the callback if provided
                            if callback:
                                callback(reflection)

                            # ENHANCED: Update continuous context for next iteration
                            continuous_context = {
                                "previous_analysis": intraday_analysis,
                                "reflection": reflection,
                                "actual_outcome": outcome_context,
                                "memory": memory_context
                            }

                # Move to next day
                current_date += timedelta(days=1)
                day_index += 1

            except Exception as e:
                logger.error(
                    f"Error processing day {day_index+1} ({current_date_str}): {e}")
                logger.exception(e)
                # Continue to next day despite error
                current_date += timedelta(days=1)
                day_index += 1

        # STEP 7: Generate a comprehensive summary with memory context
        try:
            # Use all analyses to generate a more informed summary
            analyses_count = len(pretraining_results)
            recent_analyses = pretraining_results[-3:
                                                  ] if analyses_count >= 3 else pretraining_results

            # Format summary context with memory
            summary_context = {
                "ticker": ticker,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "analyses_count": analyses_count,
                "recent_analyses": recent_analyses,
                "prediction_horizons": ["next_day", "next_week", "next_month"],
                "memory": memory_context  # Include memory context
            }

            # Get market context for the end date
            market_context = get_historical_market_context(
                market_data_dir, end_date_str)

            # Generate a final summary prompt
            summary_prompt = pretraining_prompt_hook(
                summary_context, market_context, is_summary=True)

            # Get summary from Gemini
            summary_text = gemini_client.generate_text(
                summary_prompt, temperature=0.4)

            # Create summary result
            summary = {
                "type": "summary",
                "ticker": ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "pretraining_period": f"{start_date_str} to {end_date_str}",
                "analyses_count": analyses_count,
                "full_summary": summary_text,
                "memory_context": memory_context  # Include memory for future reference
            }

            # Extract predictions for different time horizons
            for horizon in ["next_day", "next_week", "next_month"]:
                horizon_section = ""
                for section in summary_text.split("\n\n"):
                    if horizon.replace("_", " ") in section.lower():
                        horizon_section = section
                        break

                if horizon_section:
                    prediction = extract_next_day_prediction(horizon_section)
                    summary[f"{horizon}_prediction"] = prediction

            # Add to results
            pretraining_results.append(summary)

            # Call the callback if provided for the summary
            if callback:
                callback(summary)

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


def evaluate_pretraining_predictions(yfinance_client, pretraining_dir, ticker, lookback_days=30):
    """
    Evaluate the accuracy of pretraining predictions

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - pretraining_dir: Directory with pretraining data
    - ticker: Stock symbol to evaluate
    - lookback_days: Number of days to look back for predictions

    Returns:
    - Dictionary with evaluation results
    """
    logger.info(
        f"Evaluating pretraining predictions for {ticker} over the last {lookback_days} days")

    try:
        # Load all pretraining data for ticker from the specified path
        from pathlib import Path
        import json
        import glob

        ticker_dir = Path(pretraining_dir) / ticker
        if not ticker_dir.exists():
            logger.error(f"No pretraining directory found for {ticker}")
            return {"error": f"No pretraining directory found for {ticker}"}

        # Find all pretraining result files for this ticker
        result_files = list(ticker_dir.glob(f"{ticker}_pretraining_*.json"))
        if not result_files:
            logger.error(f"No pretraining result files found for {ticker}")
            return {"error": f"No pretraining result files found for {ticker}"}

        # Sort files by modification time (most recent first)
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Get the lookback period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Get actual price data for the evaluation period
        price_data = yfinance_client.get_historical_data(ticker,
                                                         start=start_date.strftime(
                                                             "%Y-%m-%d"),
                                                         end=end_date.strftime(
                                                             "%Y-%m-%d"),
                                                         interval="1d")

        if len(price_data) == 0:
            logger.error(
                f"No price data available for {ticker} during evaluation period")
            return {"error": f"No price data available for {ticker} during evaluation period"}

        # Create a mapping of dates to actual price changes
        actual_changes = {}
        for i in range(1, len(price_data)):
            date = price_data.index[i-1].strftime("%Y-%m-%d")
            next_day_date = price_data.index[i].strftime("%Y-%m-%d")
            close_price = price_data['Close'].iloc[i-1]
            next_day_price = price_data['Close'].iloc[i]
            percent_change = (
                (next_day_price - close_price) / close_price) * 100
            actual_changes[date] = {
                "date": date,
                "next_day_date": next_day_date,
                "close_price": close_price,
                "next_day_price": next_day_price,
                "percent_change": percent_change
            }

        # Initialize metrics for different prediction horizons
        metrics = {
            "next_day": {
                "directional_accuracy": "0%",
                "avg_magnitude_error": "0%",
                "sample_size": 0,
                "correct_directions": 0,
                "total_predictions": 0,
                "magnitude_errors": []
            },
            "next_week": {
                "directional_accuracy": "0%",
                "avg_magnitude_error": "0%",
                "sample_size": 0,
                "correct_directions": 0,
                "total_predictions": 0,
                "magnitude_errors": []
            },
            "next_month": {
                "directional_accuracy": "0%",
                "avg_magnitude_error": "0%",
                "sample_size": 0,
                "correct_directions": 0,
                "total_predictions": 0,
                "magnitude_errors": []
            }
        }

        # Collect all predictions from pretraining results
        all_predictions = []

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)

                # Process all results in the file
                for analysis in result_data.get('results', []):
                    analysis_date = analysis.get('date')
                    if not analysis_date:
                        continue

                    # Check if this result is within our lookback period
                    try:
                        analysis_datetime = datetime.strptime(
                            analysis_date, "%Y-%m-%d")
                        if analysis_datetime < start_date or analysis_datetime > end_date:
                            continue
                    except ValueError:
                        continue

                    # Look for next_day_prediction in analysis
                    next_day_prediction = analysis.get('next_day_prediction')
                    if next_day_prediction:
                        prediction = {
                            "date": analysis_date,
                            "horizon": "next_day",
                            "direction": next_day_prediction.get('direction', 'neutral'),
                            "magnitude": next_day_prediction.get('magnitude', 0),
                            "confidence": next_day_prediction.get('confidence', 0)
                        }
                        all_predictions.append(prediction)

                    # Look for summary predictions
                    if analysis.get('type') == 'summary':
                        for horizon in ["next_day", "next_week", "next_month"]:
                            horizon_prediction = analysis.get(
                                f'{horizon}_prediction')
                            if horizon_prediction:
                                prediction = {
                                    "date": analysis_date,
                                    "horizon": horizon,
                                    "direction": horizon_prediction.get('direction', 'neutral'),
                                    "magnitude": horizon_prediction.get('magnitude', 0),
                                    "confidence": horizon_prediction.get('confidence', 0)
                                }
                                all_predictions.append(prediction)

            except Exception as e:
                logger.error(
                    f"Error processing pretraining file {result_file}: {e}")
                continue

        # Evaluate predictions against actual changes
        for prediction in all_predictions:
            date = prediction['date']
            horizon = prediction['horizon']

            # Skip if actual data is not available
            if date not in actual_changes:
                continue

            actual = actual_changes[date]

            # Determine actual direction
            actual_direction = "neutral"
            if actual['percent_change'] > 0.5:
                actual_direction = "bullish"
            elif actual['percent_change'] < -0.5:
                actual_direction = "bearish"

            # Check if direction was correct
            correct_direction = prediction['direction'] == actual_direction

            # Calculate magnitude error
            magnitude_error = abs(
                prediction['magnitude'] - abs(actual['percent_change']))

            # Update metrics
            metrics[horizon]['total_predictions'] += 1
            metrics[horizon]['sample_size'] += 1
            if correct_direction:
                metrics[horizon]['correct_directions'] += 1
            metrics[horizon]['magnitude_errors'].append(magnitude_error)

        # Calculate final metrics
        for horizon in metrics:
            if metrics[horizon]['total_predictions'] > 0:
                directional_accuracy = (
                    metrics[horizon]['correct_directions'] / metrics[horizon]['total_predictions']) * 100
                metrics[horizon]['directional_accuracy'] = f"{directional_accuracy:.1f}%"

                if metrics[horizon]['magnitude_errors']:
                    avg_magnitude_error = sum(
                        metrics[horizon]['magnitude_errors']) / len(metrics[horizon]['magnitude_errors'])
                    metrics[horizon]['avg_magnitude_error'] = f"{avg_magnitude_error:.2f}%"

        # Create evaluation result
        evaluation = {
            "ticker": ticker,
            "evaluation_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "prediction_count": len(all_predictions),
            "metrics": metrics
        }

        logger.info(
            f"Completed evaluation for {ticker} with {len(all_predictions)} predictions")
        return evaluation

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.exception(e)
        return {"error": str(e)}


def batch_pretrain_analyzer(yfinance_client, gemini_client, pretraining_dir, tickers, pretraining_prompt_hook, **kwargs):
    """
    Perform pretraining for multiple tickers in batch

    Parameters:
    - yfinance_client: YFinance client for getting stock data
    - gemini_client: Gemini client for AI analysis
    - pretraining_dir: Directory to save pretraining results
    - tickers: List of stock symbols to analyze
    - pretraining_prompt_hook: Function to generate pretraining prompt
    - **kwargs: Additional arguments to pass to pretrain_analyzer
      - callback: Optional callback function to receive each analysis result

    Returns:
    - Dictionary with results for each ticker
    """
    logger.info(f"Starting batch pretraining for {len(tickers)} tickers")

    # Extract the callback if provided
    callback = kwargs.pop('callback', None)
    if callback:
        logger.info("Callback function provided for batch pretraining")

    results = {}
    for ticker in tickers:
        try:
            logger.info(f"Starting pretraining for {ticker}")

            # Create a copy of kwargs for this ticker
            ticker_kwargs = kwargs.copy()

            # If we have a callback, add it to the kwargs
            if callback:
                # Ensure each analysis has the ticker
                def ticker_callback(analysis):
                    # Always add ticker to analysis to ensure it's identified correctly
                    analysis["ticker"] = ticker

                    # Make a deep copy to prevent shared reference issues
                    import copy
                    analysis_copy = copy.deepcopy(analysis)

                    # Call the original callback with the modified analysis
                    try:
                        callback(analysis_copy)
                        logger.info(
                            f"Called callback for {ticker} analysis step")
                    except Exception as e:
                        logger.error(f"Error in callback for {ticker}: {e}")

                ticker_kwargs['callback'] = ticker_callback

            # Run pretraining with the modified kwargs
            ticker_result = pretrain_analyzer(
                yfinance_client,
                gemini_client,
                pretraining_dir,
                ticker,
                pretraining_prompt_hook,
                **ticker_kwargs
            )

            results[ticker] = ticker_result
            logger.info(f"Completed pretraining for {ticker}")

            # If no callback was used but results contains full data, send a summary to Discord
            if 'results' in ticker_result and len(ticker_result['results']) > 0:
                logger.info(
                    f"Pretraining results available for {ticker} with {len(ticker_result['results'])} items")

                # Ensure the ticker has associated data in the results
                if not any(item.get("ticker") == ticker for item in ticker_result['results']):
                    logger.warning(
                        f"No ticker information found in results for {ticker}, adding it now")
                    for item in ticker_result['results']:
                        item["ticker"] = ticker

                # Check if there's a summary in the results
                summary = None
                for item in ticker_result['results']:
                    if item.get('type') == 'summary':
                        summary = item
                        break

                # If there's a summary and the original callback exists, call it directly
                if summary and callback and not ticker_kwargs.get('callback'):
                    # Ensure ticker is in the summary
                    if "ticker" not in summary:
                        summary["ticker"] = ticker

                    try:
                        callback(summary)
                        logger.info(
                            f"Sent summary for {ticker} via fallback callback")
                    except Exception as e:
                        logger.error(
                            f"Error sending summary via fallback callback: {e}")

        except Exception as e:
            logger.error(f"Error pretraining {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    logger.info(f"Completed batch pretraining for {len(tickers)} tickers")
    return results
