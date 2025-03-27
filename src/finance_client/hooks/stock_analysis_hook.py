import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Union

# Import utilities
from src.finance_client.utilities.yfinance_api import download_ticker_data
from src.finance_client.utilities.data_cache import DataCache
from src.finance_client.utilities.technical_indicators import calculate_all_indicators
from src.finance_client.hooks.technical_indicators_hook import enrich_with_technical_indicators

logger = logging.getLogger(__name__)


class StockAnalysisHook:
    """Hook for analyzing stock data with technical indicators"""

    def __init__(self, use_cache: bool = True, cache_dir: str = "./data_cache"):
        """
        Initialize the stock analysis hook

        Args:
            use_cache: Whether to use caching for API calls
            cache_dir: Directory to store cached data
        """
        logger.info("Initializing StockAnalysisHook")
        self.use_cache = use_cache

        if use_cache:
            self.cache = DataCache(cache_dir)

    def get_stock_analysis(self, ticker_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get enhanced stock analysis with advanced technical indicators

        Args:
            ticker_symbol: Stock ticker symbol

        Returns:
            Dictionary containing stock analysis data
        """
        try:
            # Check cache first if enabled
            if self.use_cache:
                cached_data = self.cache.get(
                    "get_stock_analysis", ticker_symbol, max_age_hours=4)
                if cached_data is not None:
                    return cached_data

            # Get stock data - multiple timeframes for better analysis
            stock_history_hourly = download_ticker_data(
                ticker_symbol, period="5d", interval="1h")
            stock_history_daily = download_ticker_data(
                ticker_symbol, period="6mo", interval="1d")
            stock_history_weekly = download_ticker_data(
                ticker_symbol, period="1y", interval="1wk")

            if stock_history_daily.empty:
                logger.error(
                    f"Failed to retrieve daily data for {ticker_symbol}")
                return None

            # Basic stock data
            current_price = stock_history_daily['Close'].iloc[-1]
            open_price = stock_history_daily['Open'].iloc[-1]
            day_high = stock_history_daily['High'].iloc[-1]
            day_low = stock_history_daily['Low'].iloc[-1]
            recent_volume = stock_history_daily['Volume'].iloc[-5:].mean() if len(
                stock_history_daily) >= 5 else None

            # Build comprehensive technical indicators
            # Start with daily timeframe (most important)
            daily_indicators = self._calculate_technical_indicators(
                stock_history_daily, ticker_symbol)

            # Add hourly timeframe if available
            hourly_indicators = {}
            if not stock_history_hourly.empty:
                hourly_indicators = self._calculate_technical_indicators(
                    stock_history_hourly, ticker_symbol)
                daily_indicators.update(
                    {f"hourly_{k}": v for k, v in hourly_indicators.items()})

            # Add weekly timeframe for longer-term trends
            weekly_indicators = {}
            if not stock_history_weekly.empty:
                weekly_indicators = self._calculate_technical_indicators(
                    stock_history_weekly, ticker_symbol)
                daily_indicators.update(
                    {f"weekly_{k}": v for k, v in weekly_indicators.items()})

            # Determine multi-timeframe trend confluence
            trend_confluence = "neutral"
            trend_confidence = 0

            # Extract trend signals from different timeframes
            daily_trend = daily_indicators.get('trend', 'neutral')
            hourly_trend = hourly_indicators.get('trend', 'neutral')
            weekly_trend = weekly_indicators.get('trend', 'neutral')

            # Count bullish and bearish signals
            bullish_signals = sum(1 for trend in [daily_trend, hourly_trend, weekly_trend]
                                  if trend in ['bullish', 'moderately_bullish'])
            bearish_signals = sum(1 for trend in [daily_trend, hourly_trend, weekly_trend]
                                  if trend in ['bearish', 'moderately_bearish'])

            # Weighting: weekly > daily > hourly
            if weekly_trend in ['bullish', 'moderately_bullish']:
                trend_confidence += 3
            elif weekly_trend in ['bearish', 'moderately_bearish']:
                trend_confidence -= 3

            if daily_trend in ['bullish', 'moderately_bullish']:
                trend_confidence += 2
            elif daily_trend in ['bearish', 'moderately_bearish']:
                trend_confidence -= 2

            if hourly_trend in ['bullish', 'moderately_bullish']:
                trend_confidence += 1
            elif hourly_trend in ['bearish', 'moderately_bearish']:
                trend_confidence -= 1

            # Determine overall trend confluence
            if trend_confidence >= 4:
                trend_confluence = "strongly_bullish"
            elif trend_confidence >= 2:
                trend_confluence = "bullish"
            elif trend_confidence <= -4:
                trend_confluence = "strongly_bearish"
            elif trend_confidence <= -2:
                trend_confluence = "bearish"

            # Add confluence to indicators
            daily_indicators['trend_confluence'] = trend_confluence
            daily_indicators['trend_confidence'] = trend_confidence

            # Build enhanced stock data
            stock_data = {
                "ticker": ticker_symbol,
                "regularMarketPrice": current_price,
                "open": open_price,
                "dayHigh": day_high,
                "dayLow": day_low,
                "previousClose": stock_history_daily['Close'].iloc[-2] if len(stock_history_daily) > 1 else None,
                "regularMarketVolume": stock_history_daily['Volume'].iloc[-1],
                "averageVolume": recent_volume,
                "fiftyTwoWeekHigh": stock_history_daily['High'].max(),
                "fiftyTwoWeekLow": stock_history_daily['Low'].min(),
                "price_history": {
                    "1d_change": ((current_price / stock_history_daily['Close'].iloc[-2]) - 1) * 100
                    if len(stock_history_daily) > 1 else 0,
                    "1w_change": ((current_price / stock_history_daily['Close'].iloc[-6]) - 1) * 100
                    if len(stock_history_daily) >= 6 else None,
                    "1m_change": ((current_price / stock_history_daily['Close'].iloc[-23]) - 1) * 100
                    if len(stock_history_daily) >= 23 else None
                }
            }

            # Add multi-timeframe analysis results
            enhanced_result = {
                "info": stock_data,
                "technical": daily_indicators,
                "short_term_indicators": hourly_indicators,
                "long_term_indicators": weekly_indicators,
                "history": {
                    "daily_data_available": not stock_history_daily.empty,
                    "hourly_data_available": not stock_history_hourly.empty,
                    "weekly_data_available": not stock_history_weekly.empty,
                    "daily_data_points": len(stock_history_daily),
                    "hourly_data_points": len(stock_history_hourly),
                    "weekly_data_points": len(stock_history_weekly)
                }
            }

            # Cache the result if enabled
            if self.use_cache:
                self.cache.set("get_stock_analysis",
                               enhanced_result, ticker_symbol)

            return enhanced_result

        except Exception as e:
            logger.error(f"Error fetching {ticker_symbol} stock analysis: {e}")
            logger.exception(e)
            return None

    def _calculate_technical_indicators(self, data: pd.DataFrame, ticker_symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for a DataFrame

        Args:
            data: DataFrame with OHLCV data
            ticker_symbol: Stock ticker symbol

        Returns:
            Dictionary with technical indicators
        """
        try:
            # First try using the utility's calculate_all_indicators
            try:
                enriched_data = calculate_all_indicators(data)
            except Exception as utility_error:
                logger.warning(
                    f"Error using calculate_all_indicators: {utility_error}. Falling back to hook.")
                enriched_data = None

            # If that doesn't work, fall back to the hook's implementation
            if enriched_data is None or 'indicators_calculated' not in enriched_data.attrs or not enriched_data.attrs['indicators_calculated']:
                logger.info(
                    f"Using technical_indicators_hook for {ticker_symbol}")
                enriched_data = enrich_with_technical_indicators(
                    data, ticker_symbol)

            # Extract key indicators to a dictionary
            result = {}

            # Get the most recent values
            latest = enriched_data.iloc[-1]

            # Extract moving averages
            for col in enriched_data.columns:
                if isinstance(col, str) and (col.startswith('ema') or col.startswith('sma')):
                    result[col] = latest[col]

            # Extract momentum indicators
            if 'rsi' in enriched_data.columns:
                result['rsi'] = latest['rsi']
            if 'macd' in enriched_data.columns:
                result['macd'] = latest['macd']
            if 'macd_signal' in enriched_data.columns:
                result['macd_signal'] = latest['macd_signal']

            # Extract volatility indicators
            if 'atr' in enriched_data.columns:
                result['atr'] = latest['atr']
            if 'atr_percent' in enriched_data.columns:
                result['atr_percent'] = latest['atr_percent']
            if 'bollinger_upper' in enriched_data.columns:
                result['bollinger_upper'] = latest['bollinger_upper']
                result['bollinger_lower'] = latest['bollinger_lower']

            # Determine trend based on moving averages
            if 'ema9' in result and 'ema21' in result:
                if result['ema9'] > result['ema21']:
                    result['trend'] = 'bullish'
                else:
                    result['trend'] = 'bearish'

            # Add support and resistance levels
            if hasattr(enriched_data, 'attrs') and 'support_levels' in enriched_data.attrs:
                result['support'] = enriched_data.attrs['support_levels']

            if hasattr(enriched_data, 'attrs') and 'resistance_levels' in enriched_data.attrs:
                result['resistance'] = enriched_data.attrs['resistance_levels']

            # Determine if price is near support or resistance
            if 'support' in result and 'resistance' in result:
                current_price = latest['Close']
                support = result['support']
                resistance = result['resistance']

                # Calculate distance to support/resistance as percentage
                support_distance = (
                    (current_price - support) / current_price) * 100
                resistance_distance = (
                    (resistance - current_price) / current_price) * 100

                result['support_distance'] = support_distance
                result['resistance_distance'] = resistance_distance

                # Determine if price is near support or resistance (within 2%)
                result['near_support'] = support_distance <= 2.0
                result['near_resistance'] = resistance_distance <= 2.0

            return result

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            logger.exception(e)
            return {
                'trend': 'neutral',
                'error': str(e)
            }
