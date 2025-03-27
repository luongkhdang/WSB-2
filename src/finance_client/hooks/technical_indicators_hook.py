import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

# Import utilities
from src.finance_client.utilities.technical_indicators import (
    calculate_moving_averages,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
    calculate_volatility,
    calculate_support_resistance,
    calculate_fibonacci_levels
)

logger = logging.getLogger(__name__)


def enrich_with_technical_indicators(data: pd.DataFrame,
                                     ticker: str,
                                     include_volatility: bool = True,
                                     include_support_resistance: bool = True,
                                     custom_periods: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Enriches price data with technical indicators

    Args:
        data: DataFrame with OHLC price data
        ticker: Stock ticker symbol
        include_volatility: Whether to include volatility calculations
        include_support_resistance: Whether to include support/resistance levels
        custom_periods: Custom periods for moving averages (default: [9, 21, 50, 200])

    Returns:
        DataFrame enriched with technical indicators
    """
    logger.info(f"Enriching {ticker} data with technical indicators")

    if data is None or len(data) == 0:
        logger.warning(f"No data provided for {ticker}")
        return data

    try:
        # Make a copy to avoid modifying the original dataframe
        result = data.copy()

        # Default periods if none provided
        if custom_periods is None:
            custom_periods = [9, 21, 50, 200]

        # Add moving averages (SMA, EMA)
        result = calculate_moving_averages(result, periods=custom_periods)

        # Add Bollinger Bands (default 20-period, 2 standard deviations)
        result = calculate_bollinger_bands(result, window=20, num_std=2)

        # Add RSI (default 14-period)
        result = calculate_rsi(result, window=14)

        # Add MACD (default 12, 26, 9)
        result = calculate_macd(result, fast=12, slow=26, signal=9)

        # Add volatility measures if requested
        if include_volatility:
            result = calculate_volatility(result, windows=[5, 10, 20])

        # Add support/resistance levels if requested and enough data
        if include_support_resistance and len(result) > 30:
            support_resistance = calculate_support_resistance(result)

            # Add as attributes since they're not time-series data
            result.attrs['support_levels'] = support_resistance['support']
            result.attrs['resistance_levels'] = support_resistance['resistance']

            # Add Fibonacci retracement levels if we have support/resistance
            fib_levels = calculate_fibonacci_levels(result)
            result.attrs['fibonacci_levels'] = fib_levels

        # Add indicator evaluation and summary
        result = _evaluate_indicators(result, ticker)

        logger.info(f"Successfully added technical indicators for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Error adding technical indicators for {ticker}: {e}")
        # Return original data if there was an error
        return data


def _evaluate_indicators(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Evaluates technical indicators and adds summary metrics

    Args:
        data: DataFrame with technical indicators
        ticker: Stock ticker symbol

    Returns:
        DataFrame with indicator evaluation
    """
    if len(data) < 2:
        logger.warning(f"Not enough data to evaluate indicators for {ticker}")
        return data

    try:
        result = data.copy()
        current_price = data['Close'].iloc[-1]

        # Create indicators dictionary
        indicators = {}

        # Trend indicators based on moving averages
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            sma_50 = data['SMA_50'].iloc[-1]
            sma_200 = data['SMA_200'].iloc[-1]

            # Check for golden/death cross (50-day crossing 200-day)
            if len(data) > 2:
                prev_sma_50 = data['SMA_50'].iloc[-2]
                prev_sma_200 = data['SMA_200'].iloc[-2]

                golden_cross = prev_sma_50 < prev_sma_200 and sma_50 > sma_200
                death_cross = prev_sma_50 > prev_sma_200 and sma_50 < sma_200

                if golden_cross:
                    indicators['golden_cross'] = True
                    indicators['trend_signal'] = 'bullish'
                elif death_cross:
                    indicators['death_cross'] = True
                    indicators['trend_signal'] = 'bearish'

            # General trend
            if sma_50 > sma_200:
                indicators['ma_trend'] = 'bullish'
            else:
                indicators['ma_trend'] = 'bearish'

        # RSI evaluation
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            indicators['rsi'] = float(rsi)

            if rsi > 70:
                indicators['rsi_signal'] = 'overbought'
            elif rsi < 30:
                indicators['rsi_signal'] = 'oversold'
            else:
                indicators['rsi_signal'] = 'neutral'

        # MACD evaluation
        if all(x in data.columns for x in ['MACD', 'MACD_Signal']):
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            indicators['macd'] = float(macd)
            indicators['macd_signal'] = float(macd_signal)

            # Check for crossover
            if len(data) > 2:
                prev_macd = data['MACD'].iloc[-2]
                prev_macd_signal = data['MACD_Signal'].iloc[-2]

                bullish_crossover = prev_macd < prev_macd_signal and macd > macd_signal
                bearish_crossover = prev_macd > prev_macd_signal and macd < macd_signal

                if bullish_crossover:
                    indicators['macd_crossover'] = 'bullish'
                elif bearish_crossover:
                    indicators['macd_crossover'] = 'bearish'

            # Above or below zero
            if macd > 0:
                indicators['macd_position'] = 'positive'
            else:
                indicators['macd_position'] = 'negative'

        # Bollinger Band evaluation
        if all(x in data.columns for x in ['Bollinger_Upper', 'Bollinger_Lower']):
            upper_band = data['Bollinger_Upper'].iloc[-1]
            lower_band = data['Bollinger_Lower'].iloc[-1]

            # Check if price is near or outside bands
            band_range = upper_band - lower_band
            upper_threshold = upper_band - (band_range * 0.1)
            lower_threshold = lower_band + (band_range * 0.1)

            if current_price > upper_band:
                indicators['bollinger_position'] = 'above_upper'
                indicators['bollinger_signal'] = 'overbought'
            elif current_price < lower_band:
                indicators['bollinger_position'] = 'below_lower'
                indicators['bollinger_signal'] = 'oversold'
            elif current_price > upper_threshold:
                indicators['bollinger_position'] = 'near_upper'
            elif current_price < lower_threshold:
                indicators['bollinger_position'] = 'near_lower'
            else:
                indicators['bollinger_position'] = 'middle'

        # Volatility evaluation
        if 'Volatility_20' in data.columns:
            volatility = data['Volatility_20'].iloc[-1]
            indicators['volatility_20d'] = float(volatility)

            # Compare to historical volatility
            if len(data) >= 60:
                hist_vol = data['Volatility_20'].iloc[-60:-20].mean()
                vol_ratio = volatility / hist_vol if hist_vol > 0 else 1.0

                indicators['volatility_ratio'] = float(vol_ratio)

                if vol_ratio > 1.5:
                    indicators['volatility_signal'] = 'high'
                elif vol_ratio < 0.5:
                    indicators['volatility_signal'] = 'low'
                else:
                    indicators['volatility_signal'] = 'normal'

        # Create an overall technical score (-100 to +100)
        score = 0

        # Moving average component (max ±30)
        if 'ma_trend' in indicators:
            if indicators['ma_trend'] == 'bullish':
                score += 30
            else:
                score -= 30

        # RSI component (max ±25)
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            # Convert RSI to score: 0->-25, 50->0, 100->+25
            rsi_score = ((rsi - 50) / 50) * 25
            score += rsi_score

        # MACD component (max ±25)
        if 'macd_crossover' in indicators:
            if indicators['macd_crossover'] == 'bullish':
                score += 25
            elif indicators['macd_crossover'] == 'bearish':
                score -= 25
        elif 'macd_position' in indicators:
            if indicators['macd_position'] == 'positive':
                score += 10
            else:
                score -= 10

        # Bollinger component (max ±20)
        if 'bollinger_signal' in indicators:
            if indicators['bollinger_signal'] == 'oversold':
                score += 20
            elif indicators['bollinger_signal'] == 'overbought':
                score -= 20

        # Cap score between -100 and 100
        score = max(-100, min(100, score))
        indicators['technical_score'] = float(score)

        # Convert score to signal
        if score >= 70:
            indicators['overall_signal'] = 'strong_buy'
        elif score >= 30:
            indicators['overall_signal'] = 'buy'
        elif score <= -70:
            indicators['overall_signal'] = 'strong_sell'
        elif score <= -30:
            indicators['overall_signal'] = 'sell'
        else:
            indicators['overall_signal'] = 'neutral'

        # Store indicators in DataFrame attributes
        result.attrs['indicators'] = indicators
        result.attrs['technical_score'] = float(score)

        return result

    except Exception as e:
        logger.error(f"Error evaluating indicators for {ticker}: {e}")
        return data


def get_indicator_summary(data: pd.DataFrame, ticker: str) -> Dict[str, Union[str, float, Dict]]:
    """
    Gets a summary of technical indicators for a stock

    Args:
        data: DataFrame with technical indicators
        ticker: Stock ticker symbol

    Returns:
        Dictionary with indicator summary
    """
    if data is None or len(data) == 0:
        logger.warning(f"No data to summarize for {ticker}")
        return {"error": "No data available"}

    # If data hasn't been enriched yet, enrich it
    if 'indicators' not in data.attrs or data.attrs['indicators'] is None:
        data = enrich_with_technical_indicators(data, ticker)

    # Get current price and basic info
    current_price = data['Close'].iloc[-1] if len(data) > 0 else None
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else None

    # Calculate daily change
    if current_price is not None and prev_close is not None:
        daily_change = current_price - prev_close
        daily_pct_change = (daily_change / prev_close) * 100
    else:
        daily_change = None
        daily_pct_change = None

    summary = {
        "ticker": ticker,
        "current_price": float(current_price) if current_price is not None else None,
        "daily_change": float(daily_change) if daily_change is not None else None,
        "daily_pct_change": float(daily_pct_change) if daily_pct_change is not None else None,
        "indicators": data.attrs.get('indicators', {}),
        "technical_score": data.attrs.get('technical_score'),
        "support_levels": data.attrs.get('support_levels', []),
        "resistance_levels": data.attrs.get('resistance_levels', []),
        "fibonacci_levels": data.attrs.get('fibonacci_levels', {})
    }

    return summary


def get_trade_recommendations(data: pd.DataFrame, ticker: str) -> Dict[str, Union[str, float, Dict]]:
    """
    Gets trading recommendations based on technical analysis

    Args:
        data: DataFrame with technical indicators
        ticker: Stock ticker symbol

    Returns:
        Dictionary with trading recommendations
    """
    summary = get_indicator_summary(data, ticker)

    # If there was an error getting the summary
    if "error" in summary:
        return {"error": "Cannot generate recommendations: " + summary["error"]}

    indicators = summary.get("indicators", {})
    technical_score = summary.get("technical_score")
    current_price = summary.get("current_price")

    if current_price is None or technical_score is None:
        return {"error": "Insufficient data for recommendations"}

    # Get support/resistance
    support_levels = sorted([level for level in summary.get(
        "support_levels", []) if level < current_price])
    resistance_levels = sorted([level for level in summary.get(
        "resistance_levels", []) if level > current_price])

    # Find nearest support and resistance
    nearest_support = support_levels[-1] if support_levels else None
    nearest_resistance = resistance_levels[0] if resistance_levels else None

    # Determine risk:reward ratio
    risk = current_price - nearest_support if nearest_support else None
    reward = nearest_resistance - current_price if nearest_resistance else None

    risk_reward_ratio = None
    if risk is not None and reward is not None and risk > 0:
        risk_reward_ratio = reward / risk

    # Generate recommendation based on technical score and risk:reward
    recommendation = {}
    recommendation["technical_score"] = float(technical_score)
    recommendation["signal"] = indicators.get("overall_signal", "neutral")

    if risk_reward_ratio is not None:
        recommendation["risk_reward_ratio"] = float(risk_reward_ratio)

    # Set entry, stop loss and target prices
    if technical_score >= 30:  # Buy signal
        entry_price = current_price
        stop_loss = nearest_support * 0.99 if nearest_support else current_price * 0.95
        target_price = nearest_resistance if nearest_resistance else current_price * 1.1

        recommendation["position"] = "long"
        recommendation["entry_price"] = float(entry_price)
        recommendation["stop_loss"] = float(stop_loss)
        recommendation["target_price"] = float(target_price)

    elif technical_score <= -30:  # Sell signal
        entry_price = current_price
        stop_loss = nearest_resistance * 1.01 if nearest_resistance else current_price * 1.05
        target_price = nearest_support if nearest_support else current_price * 0.9

        recommendation["position"] = "short"
        recommendation["entry_price"] = float(entry_price)
        recommendation["stop_loss"] = float(stop_loss)
        recommendation["target_price"] = float(target_price)

    else:  # Neutral
        recommendation["position"] = "neutral"
        recommendation["note"] = "No clear signal. Wait for better setup."

    # Add additional context
    recommendation["key_indicators"] = {
        "trend": indicators.get("ma_trend"),
        "rsi": indicators.get("rsi_signal"),
        "macd": indicators.get("macd_crossover", indicators.get("macd_position")),
        "volatility": indicators.get("volatility_signal")
    }

    # Filter out None values
    recommendation["key_indicators"] = {
        k: v for k, v in recommendation["key_indicators"].items() if v is not None}

    return recommendation
