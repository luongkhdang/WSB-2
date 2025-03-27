#!/usr/bin/env python3
"""
Market Analysis Module (src/main_hooks/market_analysis.py)
---------------------------------------------------------
Analyzes overall market conditions using multiple indices and enhanced with pretraining data.

Functions:
  - analyze_market - Analyzes market conditions using multiple indices
  - analyze_market_with_pretraining - Enhances market analysis with pretraining insights
  - extract_movement_forecasts_from_pretraining - Extracts forecasts from pretraining results
  - generate_credit_spread_recommendations - Generates recommendations based on forecasts
  - process_options_data - Processes options data for analysis

Dependencies:
  - src.main_utilities.analysis_parser - For parsing market analysis
  - src.main_utilities.data_processor - For processing prediction data
  - src.main_utilities.file_operations - For loading pretraining results
  - YFinance client - For market data
  - Gemini client - For AI analysis

Used by:
  - main.py for analyzing market conditions
"""

import logging
import re
from datetime import datetime, timedelta

# Import utilities
from src.main_utilities.analysis_parser import parse_market_analysis
from src.main_utilities.data_processor import process_prediction

logger = logging.getLogger(__name__)


def analyze_market(yfinance_client, gemini_client, market_trend_prompt_hook):
    """
    Analyze market conditions using multiple indices

    Parameters:
    - yfinance_client: YFinance client for getting market data
    - gemini_client: Gemini client for AI analysis
    - market_trend_prompt_hook: Function to generate market trend prompt

    Returns:
    - Dictionary with market analysis
    """
    logger.info("Analyzing comprehensive market conditions...")

    try:
        # Step 1: Gather all market data from YFinance client
        market_indices_data = yfinance_client.get_market_data()
        vix_data = yfinance_client.get_volatility_filter()
        options_sentiment = yfinance_client.get_options_sentiment()

        if not market_indices_data or not vix_data:
            logger.error("Failed to retrieve market indices or VIX data")
            return None

        # For backwards compatibility with code that expects spy_market_data
        spy_market_data = market_indices_data.get("SPY", {})

        # Step 2: Format the data into a structured format
        market_data_for_analysis = {
            # SPY data
            "spy_price": spy_market_data.get("info", {}).get("regularMarketPrice"),
            "spy_open": spy_market_data.get("info", {}).get("open"),
            "spy_previous_close": spy_market_data.get("info", {}).get("previousClose"),
            "spy_volume": spy_market_data.get("info", {}).get("regularMarketVolume"),
            "spy_avg_volume": spy_market_data.get("info", {}).get("averageVolume"),
            "spy_50d_avg": spy_market_data.get("info", {}).get("fiftyDayAverage"),
            "spy_200d_avg": spy_market_data.get("info", {}).get("twoHundredDayAverage"),
            "spy_ema9": spy_market_data.get("info", {}).get("ema9"),
            "spy_ema21": spy_market_data.get("info", {}).get("ema21"),
            "spy_daily_change": spy_market_data.get("info", {}).get("dailyPctChange"),

            # QQQ data (tech)
            "qqq_price": market_indices_data.get("QQQ", {}).get("info", {}).get("regularMarketPrice"),
            "qqq_daily_change": market_indices_data.get("QQQ", {}).get("info", {}).get("dailyPctChange"),
            "qqq_ema9": market_indices_data.get("QQQ", {}).get("info", {}).get("ema9"),
            "qqq_ema21": market_indices_data.get("QQQ", {}).get("info", {}).get("ema21"),

            # IWM data (small caps)
            "iwm_price": market_indices_data.get("IWM", {}).get("info", {}).get("regularMarketPrice"),
            "iwm_daily_change": market_indices_data.get("IWM", {}).get("info", {}).get("dailyPctChange"),
            "iwm_ema9": market_indices_data.get("IWM", {}).get("info", {}).get("ema9"),
            "iwm_ema21": market_indices_data.get("IWM", {}).get("info", {}).get("ema21"),

            # VTV data (value)
            "vtv_price": market_indices_data.get("VTV", {}).get("info", {}).get("regularMarketPrice"),
            "vtv_daily_change": market_indices_data.get("VTV", {}).get("info", {}).get("dailyPctChange"),
            "vtv_ema9": market_indices_data.get("VTV", {}).get("info", {}).get("ema9"),
            "vtv_ema21": market_indices_data.get("VTV", {}).get("info", {}).get("ema21"),

            # VGLT data (bonds/treasuries)
            "vglt_price": market_indices_data.get("VGLT", {}).get("info", {}).get("regularMarketPrice"),
            "vglt_daily_change": market_indices_data.get("VGLT", {}).get("info", {}).get("dailyPctChange"),
            "vglt_ema9": market_indices_data.get("VGLT", {}).get("info", {}).get("ema9"),
            "vglt_ema21": market_indices_data.get("VGLT", {}).get("info", {}).get("ema21"),

            # DIA data (Dow Jones)
            "dia_price": market_indices_data.get("DIA", {}).get("info", {}).get("regularMarketPrice"),
            "dia_daily_change": market_indices_data.get("DIA", {}).get("info", {}).get("dailyPctChange"),
            "dia_ema9": market_indices_data.get("DIA", {}).get("info", {}).get("ema9"),
            "dia_ema21": market_indices_data.get("DIA", {}).get("info", {}).get("ema21"),

            # BND data (Total Bond Market)
            "bnd_price": market_indices_data.get("BND", {}).get("info", {}).get("regularMarketPrice"),
            "bnd_daily_change": market_indices_data.get("BND", {}).get("info", {}).get("dailyPctChange"),
            "bnd_ema9": market_indices_data.get("BND", {}).get("info", {}).get("ema9"),
            "bnd_ema21": market_indices_data.get("BND", {}).get("info", {}).get("ema21"),

            # BTC-USD data (Bitcoin)
            "btc_price": market_indices_data.get("BTC-USD", {}).get("info", {}).get("regularMarketPrice"),
            "btc_daily_change": market_indices_data.get("BTC-USD", {}).get("info", {}).get("dailyPctChange"),
            "btc_ema9": market_indices_data.get("BTC-USD", {}).get("info", {}).get("ema9"),
            "btc_ema21": market_indices_data.get("BTC-USD", {}).get("info", {}).get("ema21"),

            # VIX data
            "vix_price": vix_data.get("price"),
            "vix_stability": vix_data.get("stability"),
            "risk_adjustment": vix_data.get("risk_adjustment"),

            # Options sentiment
            "call_put_volume_ratio": options_sentiment.get("call_put_volume_ratio") if options_sentiment else None,
            "call_put_oi_ratio": options_sentiment.get("call_put_oi_ratio") if options_sentiment else None,
            "call_iv_avg": options_sentiment.get("call_iv_avg") if options_sentiment else None,
            "put_iv_avg": options_sentiment.get("put_iv_avg") if options_sentiment else None,
            "iv_skew": options_sentiment.get("iv_skew") if options_sentiment else None
        }

        # Detect sector rotation based on relative performance
        daily_changes = {
            "SPY": market_data_for_analysis["spy_daily_change"],
            "QQQ": market_data_for_analysis["qqq_daily_change"],
            "IWM": market_data_for_analysis["iwm_daily_change"],
            "VTV": market_data_for_analysis["vtv_daily_change"],
            "VGLT": market_data_for_analysis["vglt_daily_change"],
            "DIA": market_data_for_analysis["dia_daily_change"],
            "BND": market_data_for_analysis["bnd_daily_change"],
            "BTC-USD": market_data_for_analysis["btc_daily_change"]
        }

        # Remove None values
        daily_changes = {k: v for k,
                         v in daily_changes.items() if v is not None}

        if daily_changes:
            # Find leading and lagging indices
            leading_index = max(daily_changes.items(), key=lambda x: x[1])
            lagging_index = min(daily_changes.items(), key=lambda x: x[1])

            # Determine rotation pattern
            sector_rotation = ""
            if leading_index[0] == "QQQ" and daily_changes["QQQ"] > 0:
                sector_rotation = "TECH LEADERSHIP: Growth/Tech leading the market"
            elif leading_index[0] == "IWM" and daily_changes["IWM"] > 0:
                sector_rotation = "SMALL CAP STRENGTH: Risk-on environment with small caps leading"
            elif leading_index[0] == "VTV" and daily_changes["VTV"] > 0:
                sector_rotation = "VALUE ROTATION: Defensive positioning with value stocks leading"
            elif leading_index[0] == "VGLT" and daily_changes["VGLT"] > 0:
                sector_rotation = "FLIGHT TO SAFETY: Bonds leading, potential risk-off environment"

            # Add notable divergences
            spy_change = daily_changes.get("SPY", 0)
            qqq_change = daily_changes.get("QQQ", 0)
            iwm_change = daily_changes.get("IWM", 0)

            if spy_change > 0 and qqq_change < 0:
                sector_rotation += " | TECH WEAKNESS: SPY up while QQQ down, potential rotation from tech"
            elif spy_change < 0 and qqq_change > 0:
                sector_rotation += " | TECH RESILIENCE: Tech holding up despite broader market weakness"

            if spy_change > 0 and iwm_change < 0:
                sector_rotation += " | SMALL CAP WEAKNESS: Large caps outperforming small caps"
            elif spy_change < 0 and iwm_change > 0:
                sector_rotation += " | SMALL CAP RESILIENCE: Risk appetite remains for small caps"

            market_data_for_analysis["sector_rotation"] = sector_rotation
            market_data_for_analysis["leading_index"] = leading_index[0]
            market_data_for_analysis["lagging_index"] = lagging_index[0]

        # Step 4: Get the market trend prompt from hooks
        prompt = market_trend_prompt_hook(market_data_for_analysis)

        # Step 5: Send to Gemini for analysis
        market_analysis_text = gemini_client.generate_text(
            prompt, temperature=0.2)

        # Step 6: Parse the response into structured data
        market_trend = parse_market_analysis(market_analysis_text)

        # Add raw data for reference
        market_trend["raw_data"] = market_data_for_analysis

        return market_trend

    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
        return None


def analyze_market_with_pretraining(yfinance_client, gemini_client, pretraining_dir, ticker, market_trend_prompt_hook, spy_options_prompt_hook):
    """
    Analyze market with pretraining data

    Parameters:
    - yfinance_client: YFinance client
    - gemini_client: Gemini client
    - pretraining_dir: Directory with pretraining data
    - ticker: Ticker with pretraining data to use
    - market_trend_prompt_hook: Function to generate market trend prompt
    - spy_options_prompt_hook: Function to generate SPY options prompt

    Returns:
    - Dictionary with market analysis enhanced with pretraining insights
    """
    logger.info(f"Analyzing market with pretraining context for {ticker}")

    try:
        # First get a standard market analysis
        market_analysis = analyze_market(
            yfinance_client, gemini_client, market_trend_prompt_hook)

        if not market_analysis:
            logger.error("Failed to get standard market analysis")
            return None

        # Now check if pretraining data is available
        from src.main_utilities.file_operations import load_pretraining_results
        pretraining_data = load_pretraining_results(pretraining_dir, ticker)

        if not pretraining_data or "results" not in pretraining_data:
            logger.warning(
                f"No pretraining data found for {ticker}, using standard analysis")
            return market_analysis

        # Extract forecasts from pretraining data
        forecasts = extract_movement_forecasts_from_pretraining(
            pretraining_data["results"])

        if not forecasts or not forecasts.get("aggregated"):
            logger.warning(
                "No forecasts extracted from pretraining, using standard analysis")
            return market_analysis

        # Get SPY options data for broader market context
        spy_options = None
        try:
            spy_options = yfinance_client.get_options_chain("SPY")
            if spy_options is None:
                logger.warning("Failed to retrieve SPY options data")
        except Exception as e:
            logger.error(f"Error getting SPY options data: {e}")

        # Generate credit spread recommendations based on forecasts
        credit_spread_recommendations = generate_credit_spread_recommendations(
            forecasts["aggregated"])

        # Combine all data for enhanced market context
        enhanced_context = {
            "standard_analysis": market_analysis,
            "pretraining_forecasts": forecasts,
            "ticker_forecasts": {
                ticker: forecasts["aggregated"].get(
                    "overall_direction", "neutral")
            },
            "credit_spread_recommendations": credit_spread_recommendations,
            "spy_options": process_options_data(spy_options) if spy_options else {}
        }

        # Generate a special prompt for Gemini that includes pretraining insights
        prompt = spy_options_prompt_hook(enhanced_context)

        # Get enhanced analysis from Gemini
        enhanced_analysis_text = gemini_client.generate_text(
            prompt, temperature=0.2)

        # Parse enhanced analysis
        enhanced_analysis = parse_market_analysis(enhanced_analysis_text)

        # Add standard analysis data for reference
        enhanced_analysis["standard_analysis"] = market_analysis
        enhanced_analysis["pretraining_insights"] = {
            "ticker": ticker,
            "forecasts": forecasts,
            "credit_spread_recommendations": credit_spread_recommendations
        }
        enhanced_analysis["raw_data"] = market_analysis.get("raw_data", {})

        return enhanced_analysis

    except Exception as e:
        logger.error(f"Error analyzing market with pretraining: {e}")
        return None


def extract_movement_forecasts_from_pretraining(pretraining_results):
    """
    Extract movement forecasts from pretraining results

    Parameters:
    - pretraining_results: List of pretraining result entries

    Returns:
    - Dictionary of forecasts by timeframe
    """
    logger.info("Extracting movement forecasts from pretraining results")

    try:
        # Initialize forecast data
        forecasts = {
            "next_day": {
                "total_predictions": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "bullish_confidence": [],
                "bearish_confidence": [],
                "bullish_magnitude": [],
                "bearish_magnitude": [],
                "bullish_strength": 0,
                "bearish_strength": 0,
                "overall_direction": "neutral",
                "overall_magnitude": 0,
                "conviction": 0
            },
            "next_week": {
                "total_predictions": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "bullish_confidence": [],
                "bearish_confidence": [],
                "bullish_magnitude": [],
                "bearish_magnitude": [],
                "bullish_strength": 0,
                "bearish_strength": 0,
                "overall_direction": "neutral",
                "overall_magnitude": 0,
                "conviction": 0
            },
            "next_month": {
                "total_predictions": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "bullish_confidence": [],
                "bearish_confidence": [],
                "bullish_magnitude": [],
                "bearish_magnitude": [],
                "bullish_strength": 0,
                "bearish_strength": 0,
                "overall_direction": "neutral",
                "overall_magnitude": 0,
                "conviction": 0
            },
            "aggregated": {
                "total_predictions": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "bullish_confidence": [],
                "bearish_confidence": [],
                "bullish_magnitude": [],
                "bearish_magnitude": [],
                "bullish_strength": 0,
                "bearish_strength": 0,
                "overall_direction": "neutral",
                "overall_magnitude": 0,
                "conviction": 0
            }
        }

        # Go through each pretraining result
        for entry in pretraining_results:
            if "next_day_prediction" not in entry:
                continue

            prediction = entry.get("next_day_prediction", {})

            # Process the next day prediction for the "next_day" timeframe
            forecasts["next_day"] = process_prediction(
                prediction, forecasts["next_day"])

            # Also add to the aggregated forecast
            forecasts["aggregated"] = process_prediction(
                prediction, forecasts["aggregated"])

            # For weekly predictions, we'll use the daily predictions but give them less weight
            # (this is a simplification - in a real system we might have separate weekly forecasts)
            if "direction" in prediction and prediction["direction"] != "neutral":
                weekly_prediction = prediction.copy()
                weekly_prediction["confidence"] = max(
                    30, int(prediction.get("confidence", 50) * 0.8))
                forecasts["next_week"] = process_prediction(
                    weekly_prediction, forecasts["next_week"])

            # For monthly predictions, we'll use even less weight
            if "direction" in prediction and prediction["direction"] != "neutral" and prediction.get("confidence", 0) > 70:
                monthly_prediction = prediction.copy()
                monthly_prediction["confidence"] = max(
                    20, int(prediction.get("confidence", 50) * 0.6))
                forecasts["next_month"] = process_prediction(
                    monthly_prediction, forecasts["next_month"])

        return forecasts

    except Exception as e:
        logger.error(f"Error extracting movement forecasts: {e}")
        return None


def generate_credit_spread_recommendations(forecast_data):
    """
    Generate credit spread recommendations based on aggregated forecasts

    Parameters:
    - forecast_data: Aggregated forecast data

    Returns:
    - Dictionary with credit spread recommendations
    """
    logger.info("Generating credit spread recommendations from forecast data")

    recommendations = {
        "direction": "neutral",
        "conviction": 0,
        "strategy": "wait",
        "position_size": "none",
        "expiration": "none",
        "ideal_setup": {}
    }

    try:
        # Skip if no forecast data or not enough predictions
        if not forecast_data or forecast_data.get("total_predictions", 0) < 3:
            return recommendations

        # Get the overall direction and conviction
        direction = forecast_data.get("overall_direction", "neutral")
        conviction = forecast_data.get("conviction", 0)

        # Update recommendation
        recommendations["direction"] = direction
        recommendations["conviction"] = conviction

        # Determine the strategy based on direction and conviction
        if direction == "neutral" or conviction < 60:
            recommendations["strategy"] = "wait"
            recommendations["position_size"] = "none"
            recommendations["expiration"] = "none"
        elif direction == "bullish" and conviction >= 80:
            # Sell put spreads in bullish market
            recommendations["strategy"] = "put credit spread"
            recommendations["position_size"] = "full"
            recommendations["expiration"] = "2-3 weeks"
            recommendations["ideal_setup"] = {
                "spread_type": "bull put spread",
                "short_delta": "0.30",
                "long_delta": "0.20",
                "width": "2-5 points",
                "target_premium": "20-30% of width"
            }
        elif direction == "bullish" and conviction >= 60:
            recommendations["strategy"] = "put credit spread"
            recommendations["position_size"] = "half"
            recommendations["expiration"] = "1-2 weeks"
            recommendations["ideal_setup"] = {
                "spread_type": "bull put spread",
                "short_delta": "0.25",
                "long_delta": "0.15",
                "width": "2-3 points",
                "target_premium": "15-25% of width"
            }
        elif direction == "bearish" and conviction >= 80:
            # Sell call spreads in bearish market
            recommendations["strategy"] = "call credit spread"
            recommendations["position_size"] = "full"
            recommendations["expiration"] = "2-3 weeks"
            recommendations["ideal_setup"] = {
                "spread_type": "bear call spread",
                "short_delta": "0.30",
                "long_delta": "0.20",
                "width": "2-5 points",
                "target_premium": "20-30% of width"
            }
        elif direction == "bearish" and conviction >= 60:
            recommendations["strategy"] = "call credit spread"
            recommendations["position_size"] = "half"
            recommendations["expiration"] = "1-2 weeks"
            recommendations["ideal_setup"] = {
                "spread_type": "bear call spread",
                "short_delta": "0.25",
                "long_delta": "0.15",
                "width": "2-3 points",
                "target_premium": "15-25% of width"
            }

        return recommendations

    except Exception as e:
        logger.error(f"Error generating credit spread recommendations: {e}")
        return recommendations


def process_options_data(options_data):
    """Process options data for analysis"""
    from src.main_utilities.data_processor import process_options_data as process_opt_data
    return process_opt_data(options_data)
