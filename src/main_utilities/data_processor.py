import logging
import re
import math
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import talib
from talib import EMA, SMA, RSI, MACD, ATR, BBANDS, ADX

logger = logging.getLogger(__name__)


def format_stock_data_for_analysis(historical_data, ticker, date_str, min_data_points=60):
    """
    Format historical stock data for analysis with enhanced pattern recognition support

    Parameters:
    - historical_data: DataFrame with stock price data
    - ticker: Stock symbol
    - date_str: Date string for the analysis
    - min_data_points: Minimum number of data points required for pattern recognition

    Returns:
    - Dictionary with formatted data for analysis

    Raises:
    - ValueError: When critical data is missing or invalid
    """
    logger.info(f"Formatting stock data for {ticker} as of {date_str}")

    try:
        # Check if data is None or empty before processing
        if historical_data is None or (hasattr(historical_data, 'empty') and historical_data.empty):
            error_msg = f"No historical data available for {ticker} on {date_str}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Clone dataframe to avoid modifying original
        data = historical_data.copy()

        # Skip if data is empty
        if len(data) == 0:
            error_msg = f"No historical data available for {ticker} on {date_str}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log data details for debugging
        logger.debug(
            f"Data shape: {data.shape}, Index range: {data.index[0]} to {data.index[-1]}")

        # Check if we have enough data points for pattern recognition
        if len(data) < min_data_points:
            error_msg = f"Insufficient data points for pattern recognition: {len(data)} < {min_data_points}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        data_quality = "good"

        # Handle multi-index columns (from yfinance)
        # If columns are multi-index with (Price, Ticker) structure,
        # simplify to just the Price level
        if isinstance(data.columns, pd.MultiIndex):
            # For standard yfinance format where columns have (Price, Ticker) structure
            if len(data.columns.levels) == 2:
                # Get the first level which is the price type (Open, High, Low, Close, Volume)
                # We'll use xs to extract just the columns for our ticker
                try:
                    # Try to get just the data for our ticker
                    data = data.xs(ticker, level=1, axis=1)
                    logger.debug(
                        f"Successfully extracted {ticker} data from multi-index columns")
                except (KeyError, ValueError):
                    # If that fails, just take the first ticker in the data
                    first_ticker = data.columns.get_level_values(1)[0]
                    data = data.xs(first_ticker, level=1, axis=1)
                    logger.warning(
                        f"Could not find {ticker} in data, using {first_ticker} instead")
            else:
                # If the structure is different than expected, just flatten the column index
                data.columns = [col[0] if isinstance(
                    col, tuple) else col for col in data.columns]
                logger.info(f"Flattened complex column structure for {ticker}")

        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [
            col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(
                f"Missing required columns for {ticker}: {missing_cols}")
            return {
                "ticker": ticker,
                "date": date_str,
                "error": f"Missing required data columns: {missing_cols}"
            }

        # Get current price (most recent close)
        if 'Close' in data.columns and len(data) > 0:
            # Check for NaN values in Close
            if data['Close'].iloc[-1] is None or np.isnan(data['Close'].iloc[-1]):
                error_msg = f"Last Close value is NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                close_series = data['Close'].iloc[-1]

            # Extract as float
            current_price = float(close_series.iloc[0]) if hasattr(
                close_series, 'iloc') else float(close_series)
            logger.debug(f"Current price for {ticker}: {current_price}")
        else:
            error_msg = f"Cannot determine current price for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate percent change
        if len(data) >= 2:
            prev_close = data['Close'].iloc[-2]
            if prev_close is None or np.isnan(prev_close):
                error_msg = f"Previous Close value is NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            prev_close_value = float(prev_close.iloc[0]) if hasattr(
                prev_close, 'iloc') else float(prev_close)

            if prev_close_value > 0:
                percent_change = (
                    (current_price - prev_close_value) / prev_close_value) * 100
            else:
                error_msg = f"Previous close is zero or negative for {ticker}: {prev_close_value}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Insufficient data to calculate percent change for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Determine date range of the data
        try:
            start_date = data.index[0].strftime('%Y-%m-%d')
            end_date = data.index[-1].strftime('%Y-%m-%d')
        except Exception as e:
            error_msg = f"Error determining date range for {ticker}: {e}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate volume metrics
        if 'Volume' in data.columns and not data['Volume'].isnull().all():
            volume = float(data['Volume'].iloc[-1])
            avg_volume = float(data['Volume'].rolling(
                window=20).mean().iloc[-1])

            # Calculate volume change
            if avg_volume > 0:
                volume_change = ((volume - avg_volume) / avg_volume) * 100
            else:
                error_msg = f"Average volume is zero for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Volume data is missing or invalid for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate technical indicators
        # EMA 9 and 21
        if len(data) >= 21:
            ema9 = EMA(data['Close'], timeperiod=9)
            ema21 = EMA(data['Close'], timeperiod=21)

            if np.isnan(ema9.iloc[-1]) or np.isnan(ema21.iloc[-1]):
                error_msg = f"EMA calculation resulted in NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            ema9_value = float(ema9.iloc[-1])
            ema21_value = float(ema21.iloc[-1])

            # Calculate distance from price
            ema9_dist = ((current_price - ema9_value) / ema9_value) * 100
            ema21_dist = ((current_price - ema21_value) / ema21_value) * 100
        else:
            error_msg = f"Insufficient data for EMA calculation for {ticker}: {len(data)} < 21. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # SMA 50 and 200
        if len(data) >= 50:
            sma50 = SMA(data['Close'], timeperiod=50)
            sma50_value = float(sma50.iloc[-1])
            sma50_dist = ((current_price - sma50_value) / sma50_value) * 100
        else:
            error_msg = f"Insufficient data for SMA50 calculation for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Adaptive SMA long-term calculation based on available data
        if len(data) >= 200:
            # Use standard SMA200 if we have enough data
            sma200 = SMA(data['Close'], timeperiod=200)
            sma200_value = float(sma200.iloc[-1])
        elif len(data) >= 60:
            # Use adaptive SMA with the available data (e.g., SMA83 if we have 83 days)
            available_days = len(data)
            logger.warning(
                f"Using adaptive SMA{available_days} instead of SMA200 for {ticker} due to limited historical data")
            sma_adaptive = SMA(data['Close'], timeperiod=available_days)
            sma200_value = float(sma_adaptive.iloc[-1])
        else:
            error_msg = f"Insufficient data for long-term SMA calculation for {ticker}. Need at least 60 days. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate the distance from current price to SMA200 (or adaptive SMA)
        sma200_dist = ((current_price - sma200_value) / sma200_value) * 100

        # VWAP if we have high/low/close/volume
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if all(col in data.columns for col in required_columns):
            try:
                # Reset index to use date as a column
                df_reset = data.reset_index()
                df_reset['Date'] = pd.to_datetime(df_reset['Date'])

                # Group by date for daily VWAP
                date_only = df_reset['Date'].dt.date
                df_reset['date_only'] = date_only

                # Calculate VWAP
                df_reset['vwap'] = (
                    df_reset['Close'] * df_reset['Volume']).cumsum() / df_reset['Volume'].cumsum()

                # Get most recent VWAP
                vwap = float(df_reset['vwap'].iloc[-1])
            except Exception as e:
                error_msg = f"Error calculating VWAP for {ticker}: {e}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Missing required columns for VWAP calculation for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate RSI
        if len(data) >= 14:
            rsi = RSI(data['Close'], timeperiod=14)
            if np.isnan(rsi.iloc[-1]):
                error_msg = f"RSI calculation returned NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)
            rsi = float(rsi.iloc[-1])
        else:
            error_msg = f"Insufficient data for RSI calculation for {ticker}: {len(data)} < 14. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate MACD
        if len(data) >= 26:
            macd, macd_signal, macd_hist = MACD(
                data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

            if np.isnan(macd.iloc[-1]):
                error_msg = f"MACD calculation returned NaN for {ticker} despite sufficient data length {len(data)}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            macd_value = float(macd.iloc[-1])
            macd_signal_value = float(macd_signal.iloc[-1])
            macd_hist_value = float(macd_hist.iloc[-1])
        else:
            error_msg = f"Insufficient data for MACD calculation for {ticker}: {len(data)} < 26. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate Bollinger Bands
        if len(data) >= 20:
            upper, middle, lower = BBANDS(
                data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

            if np.isnan(upper.iloc[-1]) or np.isnan(middle.iloc[-1]) or np.isnan(lower.iloc[-1]):
                error_msg = f"Bollinger Bands calculation returned NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            bb_width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            bb_width = float(bb_width)
        else:
            error_msg = f"Insufficient data for Bollinger Bands calculation for {ticker}: {len(data)} < 20. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate ATR (Average True Range)
        if len(data) >= 14 and all(col in data.columns for col in ['High', 'Low', 'Close']):
            atr = ATR(data['High'], data['Low'],
                      data['Close'], timeperiod=14)

            if np.isnan(atr.iloc[-1]):
                error_msg = f"ATR calculation returned NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            atr = float(atr.iloc[-1])
            atr_percent = (atr / current_price) * 100
        else:
            error_msg = f"Insufficient data for ATR calculation for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate ADX (Average Directional Index)
        if len(data) >= 14 and all(col in data.columns for col in ['High', 'Low', 'Close']):
            adx = ADX(data['High'], data['Low'],
                      data['Close'], timeperiod=14)

            if np.isnan(adx.iloc[-1]):
                error_msg = f"ADX calculation returned NaN for {ticker}. Application stopped for debugging."
                logger.error(error_msg)
                raise ValueError(error_msg)

            adx = float(adx.iloc[-1])
        else:
            error_msg = f"Insufficient data for ADX calculation for {ticker}. Application stopped for debugging."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Return formatted data with enhanced indicators
        result = {
            "ticker": ticker,
            "date": date_str,
            "current_price": current_price,
            "percent_change": percent_change,
            "atr": atr,
            "atr_percent": atr_percent,
            "vwap": vwap,
            "volume": volume,
            "avg_volume": avg_volume,
            "volume_change": volume_change,
            "ema9": ema9_value,
            "ema21": ema21_value,
            "sma50": sma50_value,
            "sma200": sma200_value,
            "ema9_dist": ema9_dist,
            "ema21_dist": ema21_dist,
            "sma50_dist": sma50_dist,
            "sma200_dist": sma200_dist,
            "rsi": rsi,
            "macd": macd_value,
            "macd_signal": macd_signal_value,
            "macd_hist": macd_hist_value,
            "bb_width": bb_width,
            "adx": adx,
            "data_points": len(data),
            "data_quality": data_quality,
            "start_date": start_date,
            "end_date": end_date,
            "has_pattern_data": len(data) >= min_data_points
        }

        # Log summary of calculation results
        logger.info(f"Formatted data for {ticker} - Price: {current_price:.2f}, Change: {percent_change:.2f}%, " +
                    f"ATR: {atr:.4f} ({atr_percent:.2f}%), RSI: {rsi:.1f}, MACD: {macd_value:.4f}")

        return result

    except Exception as e:
        error_msg = f"Error formatting stock data for {ticker}: {e}. Application stopped for debugging."
        logger.error(error_msg)
        logger.exception(e)
        raise ValueError(error_msg)


def process_options_data(options_data):
    """
    Process options data from YFinance client for analysis

    Args:
        options_data: Options chain data from YFinance client

    Returns:
        dict: Processed options data with additional metrics
    """
    if not options_data:
        return {}

    try:
        result = {
            "expiration_date": options_data.get("expiration_date", ""),
            "call_count": 0,
            "put_count": 0,
            "call_volume": 0,
            "put_volume": 0,
            "call_open_interest": 0,
            "put_open_interest": 0,
            "call_put_volume_ratio": 0,
            "call_put_oi_ratio": 0,
            "top_calls": [],
            "top_puts": []
        }

        # Process calls
        if "calls" in options_data and options_data["calls"]:
            calls = options_data["calls"]

            # If calls is a dict (from to_dict conversion), we need to extract dataframes
            if isinstance(calls, dict) and "strike" in calls:
                # Get number of entries
                if "strike" in calls and isinstance(calls["strike"], list):
                    result["call_count"] = len(calls["strike"])
                else:
                    # Try to get length from other fields
                    for field in ["volume", "openInterest", "lastPrice"]:
                        if field in calls and isinstance(calls[field], list):
                            result["call_count"] = len(calls[field])
                            break

                # Sum volumes
                if "volume" in calls and isinstance(calls["volume"], list):
                    result["call_volume"] = sum(
                        v for v in calls["volume"] if v is not None)

                # Sum open interest
                if "openInterest" in calls and isinstance(calls["openInterest"], list):
                    result["call_open_interest"] = sum(
                        oi for oi in calls["openInterest"] if oi is not None)

                # Get top calls by volume
                if result["call_count"] > 0 and "volume" in calls and "strike" in calls:
                    try:
                        # Create sorted list of (index, volume) tuples
                        volume_idx = [(i, vol) for i, vol in enumerate(
                            calls["volume"]) if vol is not None]
                        volume_idx.sort(key=lambda x: x[1], reverse=True)

                        # Get top 3 by volume
                        for i, (idx, _) in enumerate(volume_idx[:3]):
                            if idx < len(calls["strike"]):
                                call_info = {
                                    "strike": calls["strike"][idx],
                                    "volume": calls["volume"][idx] if "volume" in calls else 0,
                                    "openInterest": calls["openInterest"][idx] if "openInterest" in calls else 0,
                                    "lastPrice": calls["lastPrice"][idx] if "lastPrice" in calls else 0
                                }
                                result["top_calls"].append(call_info)
                    except Exception as e:
                        print(f"Error processing top calls: {e}")

        # Process puts
        if "puts" in options_data and options_data["puts"]:
            puts = options_data["puts"]

            # If puts is a dict (from to_dict conversion), we need to extract dataframes
            if isinstance(puts, dict) and "strike" in puts:
                # Get number of entries
                if "strike" in puts and isinstance(puts["strike"], list):
                    result["put_count"] = len(puts["strike"])
                else:
                    # Try to get length from other fields
                    for field in ["volume", "openInterest", "lastPrice"]:
                        if field in puts and isinstance(puts[field], list):
                            result["put_count"] = len(puts[field])
                            break

                # Sum volumes
                if "volume" in puts and isinstance(puts["volume"], list):
                    result["put_volume"] = sum(
                        v for v in puts["volume"] if v is not None)

                # Sum open interest
                if "openInterest" in puts and isinstance(puts["openInterest"], list):
                    result["put_open_interest"] = sum(
                        oi for oi in puts["openInterest"] if oi is not None)

                # Get top puts by volume
                if result["put_count"] > 0 and "volume" in puts and "strike" in puts:
                    try:
                        # Create sorted list of (index, volume) tuples
                        volume_idx = [(i, vol) for i, vol in enumerate(
                            puts["volume"]) if vol is not None]
                        volume_idx.sort(key=lambda x: x[1], reverse=True)

                        # Get top 3 by volume
                        for i, (idx, _) in enumerate(volume_idx[:3]):
                            if idx < len(puts["strike"]):
                                put_info = {
                                    "strike": puts["strike"][idx],
                                    "volume": puts["volume"][idx] if "volume" in puts else 0,
                                    "openInterest": puts["openInterest"][idx] if "openInterest" in puts else 0,
                                    "lastPrice": puts["lastPrice"][idx] if "lastPrice" in puts else 0
                                }
                                result["top_puts"].append(put_info)
                    except Exception as e:
                        print(f"Error processing top puts: {e}")

        # Calculate ratios
        if result["put_volume"] > 0:
            result["call_put_volume_ratio"] = result["call_volume"] / \
                result["put_volume"]

        if result["put_open_interest"] > 0:
            result["call_put_oi_ratio"] = result["call_open_interest"] / \
                result["put_open_interest"]

        return result

    except Exception as e:
        print(f"Error processing options data: {e}")
        return {}


def safe_extract(value, default=0):
    """
    Safely extract a value, convert to a float if possible, or return a default

    Parameters:
    - value: The value to extract
    - default: Default value to return if extraction fails

    Returns:
    - Extracted value as float if possible, otherwise the default
    """
    if value is None:
        return default

    try:
        # Try converting to float
        return float(value)
    except (ValueError, TypeError):
        # If it's a string with a percentage
        if isinstance(value, str) and "%" in value:
            try:
                return float(value.replace("%", "").strip())
            except (ValueError, TypeError):
                return default
        return default


def extract_next_day_prediction(analysis_text):
    """
    Extract next day price prediction from analysis text

    Parameters:
    - analysis_text: Full analysis text from AI

    Returns:
    - Dictionary with prediction details
    """
    # Initialize prediction
    prediction = {
        "direction": "neutral",
        "magnitude": 0,
        "magnitude_low": 0,  # For range predictions
        "magnitude_high": 0,  # For range predictions
        "confidence": 0,
        "reasoning": "",
        "technical_score": 0,
        "fundamental_score": 0,
        "predictability_score": 0,  # New field
        "sentiment_score": 0,      # Keep for backward compatibility
        "total_score": 0
    }

    try:
        # Look for next day prediction section
        prediction_section = ""

        # First look for PREDICTION: section
        prediction_match = re.search(
            r'PREDICTION:\s*(.*?)(?:\n\n|\n[A-Z]|$)', analysis_text, re.IGNORECASE | re.DOTALL)
        if prediction_match:
            prediction_section = prediction_match.group(1)
        else:
            # Fall back to looking for next day references
            sections = analysis_text.split("\n\n")
            for section in sections:
                if "next day" in section.lower() or "tomorrow" in section.lower() or "1-day" in section.lower():
                    prediction_section = section
                    break

        if not prediction_section:
            return prediction

        # Extract direction - more thorough pattern matching
        if "bullish" in prediction_section.lower() or "upward" in prediction_section.lower() or "higher" in prediction_section.lower() or "upside" in prediction_section.lower():
            prediction["direction"] = "bullish"
        elif "bearish" in prediction_section.lower() or "downward" in prediction_section.lower() or "lower" in prediction_section.lower() or "downside" in prediction_section.lower():
            prediction["direction"] = "bearish"

        # Look for the specific formatted prediction
        formatted_prediction = re.search(
            r'(?:Next day prediction:|1-day:|Tomorrow:)\s*(\w+)\s+move\s+of\s+(\d+\.?\d*)-(\d+\.?\d*)%\s+with\s+(\d+)%\s+confidence', prediction_section, re.IGNORECASE)
        if formatted_prediction:
            # This is our new precise format
            direction_word = formatted_prediction.group(1).lower()
            if "bull" in direction_word or "up" in direction_word:
                prediction["direction"] = "bullish"
            elif "bear" in direction_word or "down" in direction_word:
                prediction["direction"] = "bearish"

            # Extract the range values
            prediction["magnitude_low"] = float(formatted_prediction.group(2))
            prediction["magnitude_high"] = float(formatted_prediction.group(3))
            # Use midpoint for single magnitude value
            prediction["magnitude"] = (
                prediction["magnitude_low"] + prediction["magnitude_high"]) / 2

            # Extract confidence
            prediction["confidence"] = int(formatted_prediction.group(4))
        else:
            # Fall back to older pattern matching
            # Extract magnitude - try to find range pattern first (X.X-Y.Y%)
            magnitude_range_match = re.search(
                r'(\d+\.?\d*)-(\d+\.?\d*)%', prediction_section)
            if magnitude_range_match:
                prediction["magnitude_low"] = float(
                    magnitude_range_match.group(1))
                prediction["magnitude_high"] = float(
                    magnitude_range_match.group(2))
                # Use midpoint for single magnitude value
                prediction["magnitude"] = (
                    prediction["magnitude_low"] + prediction["magnitude_high"]) / 2
            else:
                # Look for single percentage value
                magnitude_match = re.search(
                    r'(\d+\.?\d*)%', prediction_section)
                if magnitude_match:
                    prediction["magnitude"] = float(magnitude_match.group(1))
                    # Set range values to the same
                    prediction["magnitude_low"] = prediction["magnitude"]
                    prediction["magnitude_high"] = prediction["magnitude"]

            # Extract confidence
            confidence_match = re.search(
                r'(?:confidence|probability)(?:\s+level)?:?\s*(\d+)(?:\s*%)?', prediction_section, re.IGNORECASE)
            if confidence_match:
                prediction["confidence"] = int(confidence_match.group(1))

        # Extract scores
        # Look for Technical Score in the full analysis
        technical_match = re.search(
            r'Technical Score:?\s*(\d+)', analysis_text, re.IGNORECASE)
        if technical_match:
            prediction["technical_score"] = int(technical_match.group(1))

        # Look for Predictability Score first, fall back to Sentiment Score
        predictability_match = re.search(
            r'Predictability Score:?\s*(\d+)', analysis_text, re.IGNORECASE)
        if predictability_match:
            prediction["predictability_score"] = int(
                predictability_match.group(1))
            # For backward compatibility
            prediction["sentiment_score"] = prediction["predictability_score"]
        else:
            # Look for Sentiment Score as fallback
            sentiment_match = re.search(
                r'Sentiment Score:?\s*(\d+)', analysis_text, re.IGNORECASE)
            if sentiment_match:
                prediction["sentiment_score"] = int(sentiment_match.group(1))
                # For forward compatibility
                prediction["predictability_score"] = prediction["sentiment_score"]

        # Look for Fundamental Score in the full analysis
        fundamental_match = re.search(
            r'Fundamental Score:?\s*(\d+)', analysis_text, re.IGNORECASE)
        if fundamental_match:
            prediction["fundamental_score"] = int(fundamental_match.group(1))

        # Calculate total score using max of predictability/sentiment
        prediction["total_score"] = (
            prediction["technical_score"] +
            prediction["fundamental_score"] +
            max(prediction["predictability_score"],
                prediction["sentiment_score"])
        )

        # Extract reasoning (use the whole section for now)
        prediction["reasoning"] = prediction_section

        return prediction

    except Exception as e:
        logger.error(f"Error extracting next day prediction: {e}")
        return prediction


def process_prediction(prediction, forecast_data):
    """Process a prediction and update forecast data"""
    if not prediction or not forecast_data:
        return forecast_data

    try:
        # Get basic direction and confidence
        direction = prediction.get("direction", "neutral")
        confidence = prediction.get("confidence", 50)
        magnitude = prediction.get("magnitude", 0)

        # Skip if no clear direction or very low confidence
        if direction == "neutral" or confidence < 30:
            return forecast_data

        # Update directional counts
        forecast_data["total_predictions"] += 1

        if direction == "bullish":
            forecast_data["bullish_count"] += 1
            forecast_data["bullish_confidence"].append(confidence)
            forecast_data["bullish_magnitude"].append(magnitude)
        elif direction == "bearish":
            forecast_data["bearish_count"] += 1
            forecast_data["bearish_confidence"].append(confidence)
            forecast_data["bearish_magnitude"].append(magnitude)

        # Calculate new overall confidence
        if forecast_data["total_predictions"] > 0:
            total_confidence = 0
            total_count = 0

            if forecast_data["bullish_count"] > 0:
                total_confidence += sum(forecast_data["bullish_confidence"])
                total_count += forecast_data["bullish_count"]

            if forecast_data["bearish_count"] > 0:
                total_confidence += sum(forecast_data["bearish_confidence"])
                total_count += forecast_data["bearish_count"]

            if total_count > 0:
                forecast_data["avg_confidence"] = total_confidence / total_count

        # Update strength values
        if forecast_data["bullish_count"] > 0 and forecast_data["bullish_confidence"]:
            forecast_data["bullish_strength"] = sum(
                forecast_data["bullish_confidence"]) / len(forecast_data["bullish_confidence"])

        if forecast_data["bearish_count"] > 0 and forecast_data["bearish_confidence"]:
            forecast_data["bearish_strength"] = sum(
                forecast_data["bearish_confidence"]) / len(forecast_data["bearish_confidence"])

        # Calculate overall direction and magnitude
        if forecast_data["bullish_count"] > forecast_data["bearish_count"]:
            forecast_data["overall_direction"] = "bullish"
            if forecast_data["bullish_magnitude"]:
                forecast_data["overall_magnitude"] = sum(
                    forecast_data["bullish_magnitude"]) / len(forecast_data["bullish_magnitude"])
        elif forecast_data["bearish_count"] > forecast_data["bullish_count"]:
            forecast_data["overall_direction"] = "bearish"
            if forecast_data["bearish_magnitude"]:
                forecast_data["overall_magnitude"] = sum(
                    forecast_data["bearish_magnitude"]) / len(forecast_data["bearish_magnitude"])
        else:
            # Equal counts - determine by strength
            if forecast_data["bullish_strength"] > forecast_data["bearish_strength"]:
                forecast_data["overall_direction"] = "bullish"
                if forecast_data["bullish_magnitude"]:
                    forecast_data["overall_magnitude"] = sum(
                        forecast_data["bullish_magnitude"]) / len(forecast_data["bullish_magnitude"])
            elif forecast_data["bearish_strength"] > forecast_data["bullish_strength"]:
                forecast_data["overall_direction"] = "bearish"
                if forecast_data["bearish_magnitude"]:
                    forecast_data["overall_magnitude"] = sum(
                        forecast_data["bearish_magnitude"]) / len(forecast_data["bearish_magnitude"])

        # Calculate conviction score
        total_strength = forecast_data["bullish_strength"] + \
            forecast_data["bearish_strength"]
        if total_strength > 0:
            if forecast_data["overall_direction"] == "bullish":
                forecast_data["conviction"] = (
                    forecast_data["bullish_strength"] / total_strength) * 100
            elif forecast_data["overall_direction"] == "bearish":
                forecast_data["conviction"] = (
                    forecast_data["bearish_strength"] / total_strength) * 100

        return forecast_data

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        return forecast_data


def process_error_data(error_data, error_metrics):
    """Process error data and update error metrics"""
    if not error_data or not error_metrics:
        return error_metrics

    try:
        # Get basic error values
        actual_change = error_data.get("actual_change", 0)
        predicted_change = error_data.get("predicted_change", 0)
        predicted_direction = error_data.get("predicted_direction", "neutral")

        # Skip if no clear direction predicted
        if predicted_direction == "neutral":
            return error_metrics

        # Determine actual direction
        actual_direction = "neutral"
        if actual_change > 0.5:  # Half percent threshold to avoid noise
            actual_direction = "bullish"
        elif actual_change < -0.5:
            actual_direction = "bearish"

        # Increment counters
        error_metrics["total_predictions"] += 1

        # Check directional accuracy
        if predicted_direction == actual_direction:
            error_metrics["correct_directions"] += 1

        # Calculate magnitude error
        magnitude_error = abs(actual_change - predicted_change)
        error_metrics["magnitude_errors"].append(magnitude_error)

        # Update averages
        if error_metrics["total_predictions"] > 0:
            error_metrics["directional_accuracy"] = (
                error_metrics["correct_directions"] / error_metrics["total_predictions"]) * 100

        if error_metrics["magnitude_errors"]:
            error_metrics["avg_magnitude_error"] = sum(
                error_metrics["magnitude_errors"]) / len(error_metrics["magnitude_errors"])

        return error_metrics

    except Exception as e:
        logger.error(f"Error processing error data: {e}")
        return error_metrics


def format_credit_spread_data(data, ticker, date=None, include_intraday=False):
    """
    Format stock data with emphasis on parameters relevant for credit spreads.

    Parameters:
    - data: DataFrame with stock historical data
    - ticker: Stock symbol
    - date: Optional specific date to format
    - include_intraday: Whether to include intraday data

    Returns:
    - Dictionary with formatted data optimized for credit spread analysis
    """
    if data is None or len(data) == 0:
        return {
            "ticker": ticker,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "error": "No data available"
        }

    # Filter by date if provided
    if date:
        date_data = data[data.index.strftime("%Y-%m-%d") == date]
        if len(date_data) == 0:
            # Try to get the last available data point before this date
            date_data = data[data.index <= date].tail(1)
            if len(date_data) == 0:
                return {
                    "ticker": ticker,
                    "date": date,
                    "error": f"No data available for {date}"
                }
    else:
        # Use the most recent data point
        date_data = data.tail(1)
        date = date_data.index[0].strftime("%Y-%m-%d")

    # Get last row for the current values
    last_row = date_data.iloc[-1]

    # Calculate key metrics for credit spreads
    current_price = float(last_row["Close"])

    # Calculate ATR if possible (typically 14-day)
    atr = None
    atr_percent = None
    try:
        if len(data) >= 14:
            high_low = data["High"] - data["Low"]
            high_close = abs(data["High"] - data["Close"].shift())
            low_close = abs(data["Low"] - data["Close"].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_percent = (atr / current_price) * 100
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")

    # Get implied volatility if available
    implied_volatility = last_row.get("IV", None)

    # Calculate historical volatility (20-day standard deviation annualized)
    historical_volatility = None
    try:
        if len(data) >= 20:
            # Calculate daily returns
            returns = data["Close"].pct_change().dropna()
            # Calculate 20-day standard deviation
            std_dev = returns.rolling(20).std().iloc[-1]
            # Annualize (multiply by sqrt(252))
            historical_volatility = std_dev * \
                np.sqrt(252) * 100  # as percentage
    except Exception as e:
        logger.warning(f"Error calculating historical volatility: {e}")

    # Find support and resistance levels
    support_levels = []
    resistance_levels = []
    try:
        # Use the last 30 days of data for levels
        level_data = data.tail(30)

        # Find local minima (support)
        for i in range(1, len(level_data) - 1):
            if level_data["Low"].iloc[i] < level_data["Low"].iloc[i-1] and \
               level_data["Low"].iloc[i] < level_data["Low"].iloc[i+1]:
                support_levels.append(float(level_data["Low"].iloc[i]))

        # Find local maxima (resistance)
        for i in range(1, len(level_data) - 1):
            if level_data["High"].iloc[i] > level_data["High"].iloc[i-1] and \
               level_data["High"].iloc[i] > level_data["High"].iloc[i+1]:
                resistance_levels.append(float(level_data["High"].iloc[i]))

        # Keep only the 3 most significant levels (closest to current price)
        support_levels = sorted(
            support_levels, key=lambda x: abs(current_price - x))[:3]
        resistance_levels = sorted(
            resistance_levels, key=lambda x: abs(current_price - x))[:3]
    except Exception as e:
        logger.warning(f"Error identifying support/resistance levels: {e}")

    # Create result dictionary with credit spread emphasis
    result = {
        "ticker": ticker,
        "date": date,
        "current_price": current_price,
        "open": float(last_row["Open"]),
        "high": float(last_row["High"]),
        "low": float(last_row["Low"]),
        "volume": int(last_row["Volume"]),
        "atr": atr,
        "atr_percent": atr_percent,
        "implied_volatility": implied_volatility,
        "historical_volatility": historical_volatility,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels
    }

    # Add technical indicators if available
    if "RSI" in last_row:
        result["rsi"] = float(last_row["RSI"])
    if "MACD" in last_row:
        result["macd"] = float(last_row["MACD"])
    if "EMA_9" in last_row:
        result["ema_9"] = float(last_row["EMA_9"])
    if "EMA_21" in last_row:
        result["ema_21"] = float(last_row["EMA_21"])
    if "SMA_50" in last_row:
        result["sma_50"] = float(last_row["SMA_50"])
    if "SMA_200" in last_row:
        result["sma_200"] = float(last_row["SMA_200"])
    if "BB_Upper" in last_row and "BB_Lower" in last_row:
        result["bollinger_bands"] = {
            "upper": float(last_row["BB_Upper"]),
            "lower": float(last_row["BB_Lower"]),
            "width": float(last_row["BB_Upper"] - last_row["BB_Lower"]) / current_price * 100
        }

    # Add intraday data if requested and available
    # More than one row for the day
    if include_intraday and len(date_data) > 1:
        result["intraday_data"] = {
            "bars_count": len(date_data),
            "open": float(date_data["Open"].iloc[0]),
            "high": float(date_data["High"].max()),
            "low": float(date_data["Low"].min()),
            "close": float(date_data["Close"].iloc[-1]),
            "volume": int(date_data["Volume"].sum()),
        }

        # Add VWAP if volume is available
        if "Volume" in date_data and date_data["Volume"].sum() > 0:
            result["intraday_data"]["vwap"] = float(
                np.average(date_data["Close"], weights=date_data["Volume"])
            )

    return result


def check_for_fallbacks(data, ticker, strict_mode=True):
    """
    Check if any critical data points are using fallback values

    Parameters:
    - data: Dictionary with the formatted data
    - ticker: Stock symbol
    - strict_mode: If True, raise errors when fallbacks are detected

    Returns:
    - Dictionary with fallback information

    Raises:
    - ValueError: If strict_mode is True and fallbacks are detected
    """
    fallbacks = {
        'detected': False,
        'missing_indicators': [],
        'fallback_values': {}
    }

    # Check for NaN or None values in critical indicators
    critical_indicators = [
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'adx', 'atr',
        'sma50', 'sma200',
        'ema9', 'ema21'
    ]

    for indicator in critical_indicators:
        if indicator not in data or data[indicator] is None or (hasattr(data[indicator], 'dtype') and np.isnan(data[indicator])):
            fallbacks['detected'] = True
            fallbacks['missing_indicators'].append(indicator)
            fallbacks['fallback_values'][indicator] = None

            # Log and raise error in strict mode
            error_msg = f"Missing critical indicator {indicator} for {ticker}. Application stopped for debugging."
            logger.error(error_msg)

            if strict_mode:
                raise ValueError(error_msg)

    # Check for placeholder values that might indicate fallbacks
    if 'macd' in data and data['macd'] == 0 and 'macd_signal' in data and data['macd_signal'] == 0:
        # This pattern of both MACD and signal being exactly 0 is highly unlikely in real data
        fallbacks['detected'] = True
        fallbacks['missing_indicators'].append('macd (suspect fallback)')
        fallbacks['fallback_values']['macd'] = 0

        error_msg = f"Suspected fallback value detected for MACD ({data['macd']}) for {ticker}. Application stopped for debugging."
        logger.error(error_msg)

        if strict_mode:
            raise ValueError(error_msg)

    # Check for unreasonable volatility values
    if 'atr_percent' in data and data['atr_percent'] == 0:
        # ATR% is never exactly 0 in real data
        fallbacks['detected'] = True
        fallbacks['missing_indicators'].append(
            'atr_percent (suspect fallback)')
        fallbacks['fallback_values']['atr_percent'] = 0

        error_msg = f"Suspected fallback value detected for ATR% ({data['atr_percent']}) for {ticker}. Application stopped for debugging."
        logger.error(error_msg)

        if strict_mode:
            raise ValueError(error_msg)

    # If fallbacks were detected but we're not in strict mode, log a warning
    if fallbacks['detected'] and not strict_mode:
        logger.warning(
            f"Fallbacks detected for {ticker}: {', '.join(fallbacks['missing_indicators'])}")

    return fallbacks


def enforce_strict_validation(func):
    """
    Decorator to enforce strict validation on data processing functions.
    Will check the returned data for fallbacks and raise ValueError if found.

    Parameters:
    - func: The function to decorate

    Returns:
    - Decorated function that validates no fallbacks are present

    Example:
    @enforce_strict_validation
    def process_data(data, ticker):
        # Process data
        return processed_data
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the ticker parameter - could be a positional or keyword argument
        ticker = None
        # Check if ticker is a keyword arg
        if 'ticker' in kwargs:
            ticker = kwargs['ticker']
        # If not found, try to get it from the positional args
        # Most data processing functions take ticker as the second argument after data
        elif len(args) >= 2 and isinstance(args[1], str):
            ticker = args[1]

        # Call the original function
        result = func(*args, **kwargs)

        # If the result is a dictionary and we have a ticker, check for fallbacks
        if isinstance(result, dict) and ticker:
            check_for_fallbacks(result, ticker, strict_mode=True)

        return result

    return wrapper
