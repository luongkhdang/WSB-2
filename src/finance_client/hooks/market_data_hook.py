import logging
import pandas as pd
from typing import Dict, Any, Optional, List
import re

# Import utilities
from src.finance_client.utilities.yfinance_api import download_ticker_data
from src.finance_client.utilities.data_cache import DataCache
from src.finance_client.utilities.technical_indicators import calculate_all_indicators

logger = logging.getLogger(__name__)


class MarketDataHook:
    """Hook for fetching and analyzing market data"""

    def __init__(self, use_cache: bool = True, cache_dir: str = "./data_cache"):
        """
        Initialize the market data hook

        Args:
            use_cache: Whether to use caching for API calls
            cache_dir: Directory to store cached data
        """
        logger.info("Initializing MarketDataHook")
        self.use_cache = use_cache

        if use_cache:
            self.cache = DataCache(cache_dir)

        # Common market indices to track
        self.indices = {
            "SPY": "SPY",   # S&P 500 Index ETF
            "QQQ": "QQQ",   # Nasdaq 100 ETF (Tech-heavy)
            "IWM": "IWM",   # Russell 2000 Small Cap ETF
            "VTV": "VTV",   # Vanguard Value ETF
            "VGLT": "VGLT",  # Vanguard Long-Term Treasury ETF
            "VIX": "^VIX",  # Volatility Index
            "DIA": "DIA",   # Dow Jones Industrial Average ETF
            "BND": "BND",   # Total Bond Market ETF
        }

    def get_market_data(self) -> Dict[str, Any]:
        """
        Get comprehensive market data including major indices

        Returns:
            Dictionary containing market data for major indices
        """
        try:
            # Dictionary to store all market data
            market_data = {}

            # Check cache first if enabled
            if self.use_cache:
                cached_data = self.cache.get(
                    "get_market_data", max_age_hours=4)
                if cached_data is not None:
                    return cached_data

            # Get historical data for each index
            for index_name, ticker in self.indices.items():
                history = download_ticker_data(
                    ticker, period="5d", interval="1h")

                if len(history) == 0:
                    logger.error(f"Failed to retrieve {ticker} data")
                    continue

                # Calculate basic metrics
                current_price = history['Close'].iloc[-1].item() if len(
                    history) > 0 else None
                open_price = history['Open'].iloc[-1].item() if len(
                    history) > 0 else None
                day_high = history['High'].iloc[-1].item() if len(history) > 0 else None
                day_low = history['Low'].iloc[-1].item() if len(history) > 0 else None
                previous_close = history['Close'].iloc[-2].item() if len(
                    history) > 1 else None

                # Calculate daily change and percentage
                daily_change = current_price - \
                    previous_close if current_price and previous_close else None
                daily_pct_change = (
                    daily_change / previous_close * 100) if daily_change and previous_close else None

                # Calculate 50-day average if we have enough data
                fifty_day_avg = None
                if len(history) >= 50:
                    fifty_day_avg = history['Close'].rolling(
                        window=50).mean().iloc[-1].item()

                # Calculate 200-day average if we have enough data
                two_hundred_day_avg = None
                if len(history) >= 200:
                    two_hundred_day_avg = history['Close'].rolling(
                        window=200).mean().iloc[-1].item()

                # Calculate 9-day and 21-day EMA for trend analysis
                ema9 = history['Close'].ewm(span=9, adjust=False).mean(
                ).iloc[-1].item() if len(history) > 0 else None
                ema21 = history['Close'].ewm(span=21, adjust=False).mean(
                ).iloc[-1].item() if len(history) > 0 else None

                # Basic market data
                index_data = {
                    "regularMarketPrice": current_price,
                    "open": open_price,
                    "dayHigh": day_high,
                    "dayLow": day_low,
                    "previousClose": previous_close,
                    "regularMarketVolume": history['Volume'].iloc[-1].item() if len(history) > 0 else None,
                    "averageVolume": history['Volume'].mean() if len(history) > 0 else None,
                    "fiftyDayAverage": fifty_day_avg,
                    "twoHundredDayAverage": two_hundred_day_avg,
                    "ema9": ema9,
                    "ema21": ema21,
                    "dailyChange": daily_change,
                    "dailyPctChange": daily_pct_change
                }

                market_data[index_name] = {
                    "info": index_data,
                    "history": history
                }

            # If we couldn't get SPY data (our primary index), return None
            if "SPY" not in market_data:
                logger.error(
                    "Failed to retrieve SPY data, which is required for market analysis")
                return None

            # Add technical indicators to SPY data if available
            if "SPY" in market_data and market_data["SPY"]["history"] is not None:
                spy_daily = download_ticker_data(
                    "SPY", period="3mo", interval="1d")
                if len(spy_daily) > 0:
                    spy_indicators = calculate_all_indicators(spy_daily)
                    market_data["SPY"]["technical"] = self._extract_technical_indicators(
                        spy_indicators)

            # Cache the result if enabled
            if self.use_cache:
                self.cache.set("get_market_data", market_data)

            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def get_volatility_filter(self) -> Dict[str, Any]:
        """
        Get VIX data for market volatility filtering

        Returns:
            Dictionary containing VIX analysis and risk adjustment recommendations
        """
        try:
            # Check cache first if enabled
            if self.use_cache:
                cached_data = self.cache.get(
                    "get_volatility_filter", max_age_hours=4)
                if cached_data is not None:
                    return cached_data

            vix_data = download_ticker_data("^VIX", period="7d", interval="1d")

            if vix_data.empty or len(vix_data) < 2:
                logger.error("Failed to retrieve VIX data")
                logger.info("Using fallback VIX data to continue workflow")
                # Provide fallback data
                fallback_data = {
                    "price": 19.29,  # Moderate VIX level as fallback
                    "stability": "normal",
                    "risk_adjustment": 1.0,
                    "adjustment_note": "Standard position sizing (fallback data)",
                    "is_fallback": True,
                    "trend": "unknown"
                }
                return fallback_data

            # Extract the last VIX price
            try:
                vix_price = float(vix_data['Close'].iloc[-1])
                vix_open = float(vix_data['Open'].iloc[-1])
                vix_prev_close = float(
                    vix_data['Close'].iloc[-2]) if len(vix_data) >= 2 else vix_open

                # Calculate day change and trend
                vix_change = vix_price - vix_prev_close
                vix_change_pct = (vix_change / vix_prev_close) * \
                    100 if vix_prev_close > 0 else 0

                # Calculate 3-day trend
                vix_3d_trend = "flat"
                if len(vix_data) >= 4:
                    vix_3d_ago = float(vix_data['Close'].iloc[-4])
                    vix_3d_change = (
                        (vix_price - vix_3d_ago) / vix_3d_ago) * 100
                    if vix_3d_change > 10:
                        vix_3d_trend = "sharply_rising"
                    elif vix_3d_change > 5:
                        vix_3d_trend = "rising"
                    elif vix_3d_change < -10:
                        vix_3d_trend = "sharply_falling"
                    elif vix_3d_change < -5:
                        vix_3d_trend = "falling"

                logger.info(
                    f"Current VIX price: {vix_price}, Change: {vix_change_pct:.2f}%, 3-day trend: {vix_3d_trend}")
            except Exception as e:
                logger.error(f"Error extracting VIX price from data: {e}")
                logger.info("Using fallback VIX data to continue workflow")
                fallback_data = {
                    "price": 19.29,  # Moderate VIX level as fallback
                    "stability": "normal",
                    "risk_adjustment": 1.0,
                    "adjustment_note": "Standard position sizing (fallback data)",
                    "is_fallback": True,
                    "trend": "unknown"
                }
                return fallback_data

            # Calculate market stability score based on Rule Book
            # Below 15: Very stable, 15-20: Stable, 20-25: Normal, 25-35: Elevated, Above 35: Extreme
            if vix_price < 15:
                stability = "very_stable"
                risk_adjustment = 1.0  # No adjustment
                adjustment_note = "Standard position sizing (1-2% of account)"
            elif vix_price < 20:
                stability = "stable"
                risk_adjustment = 1.0  # No adjustment
                adjustment_note = "Standard position sizing with +5 to Market Trend score"
            elif vix_price < 25:
                stability = "normal"
                risk_adjustment = 1.0  # No adjustment
                adjustment_note = "Standard position sizing"
            elif vix_price < 35:
                stability = "elevated"
                risk_adjustment = 0.5  # Halve position size per Rule Book
                adjustment_note = "Reduce position size by 50% (-5 to score unless size halved)"
            else:
                stability = "extreme"
                risk_adjustment = 0.0  # Stay in cash per Rule Book
                adjustment_note = "Skip unless justified by high Gamble Score (>80)"

            # Add trend information
            trend = "stable"
            if vix_change_pct > 10:
                trend = "sharply_rising"
                adjustment_note += " | Caution: VIX spiking, consider waiting"
            elif vix_change_pct > 5:
                trend = "rising"
                adjustment_note += " | VIX rising, increased market uncertainty"
            elif vix_change_pct < -10:
                trend = "sharply_falling"
                adjustment_note += " | VIX collapsing, potential for new bull trend"
            elif vix_change_pct < -5:
                trend = "falling"
                adjustment_note += " | VIX declining, decreasing market uncertainty"

            # Add context from more historical VIX data
            vix_context = {}
            try:
                vix_monthly = download_ticker_data(
                    "^VIX", period="1mo", interval="1d")
                if not vix_monthly.empty and len(vix_monthly) > 5:
                    # Calculate scalar values properly
                    high_max = vix_monthly['High'].max()
                    low_min = vix_monthly['Low'].min()
                    close_mean = vix_monthly['Close'].mean()

                    vix_context = {
                        "1m_high": float(high_max),
                        "1m_low": float(low_min),
                        "1m_avg": float(close_mean),
                        "percentile": float(((vix_price - low_min) / (high_max - low_min)) * 100)
                        if (high_max - low_min) > 0 else 50
                    }
            except Exception as e:
                logger.warning(f"Error retrieving historical VIX context: {e}")

            result = {
                "price": vix_price,
                "stability": stability,
                "risk_adjustment": risk_adjustment,
                "adjustment_note": adjustment_note,
                "change": vix_change,
                "change_pct": vix_change_pct,
                "trend": trend,
                "is_fallback": False,
                "context": vix_context
            }

            # Cache the result if enabled
            if self.use_cache:
                self.cache.set("get_volatility_filter", result)

            return result

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            logger.info("Using fallback VIX data to continue workflow")
            return {
                "price": 22.5,  # Moderate VIX level as fallback
                "stability": "normal",
                "risk_adjustment": 1.0,
                "adjustment_note": "Standard position sizing (fallback data)",
                "is_fallback": True,
                "trend": "unknown"
            }

    def _extract_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key technical indicators from a DataFrame with indicators

        Args:
            data: DataFrame with technical indicators calculated

        Returns:
            Dictionary with key technical indicators
        """
        try:
            # Get the most recent values
            latest = data.iloc[-1]

            # Extract moving averages
            moving_averages = {}
            for col in data.columns:
                if col.startswith('ema') or col.startswith('sma'):
                    moving_averages[col] = latest[col]

            # Extract RSI, MACD
            momentum = {}
            if 'rsi' in data.columns:
                momentum['rsi'] = latest['rsi']
            if 'macd' in data.columns:
                momentum['macd'] = latest['macd']
            if 'macd_signal' in data.columns:
                momentum['macd_signal'] = latest['macd_signal']
            if 'macd_histogram' in data.columns:
                momentum['macd_histogram'] = latest['macd_histogram']

            # Extract volatility indicators
            volatility = {}
            if 'atr' in data.columns:
                volatility['atr'] = latest['atr']
            if 'atr_percent' in data.columns:
                volatility['atr_percent'] = latest['atr_percent']
            if 'bollinger_upper' in data.columns:
                volatility['bollinger_upper'] = latest['bollinger_upper']
            if 'bollinger_lower' in data.columns:
                volatility['bollinger_lower'] = latest['bollinger_lower']
            if 'bollinger_width' in data.columns:
                volatility['bollinger_width'] = latest['bollinger_width']

            # Get support/resistance
            support_resistance = {}
            if hasattr(data, 'attrs') and 'support_levels' in data.attrs:
                support_resistance['support'] = data.attrs['support_levels']
            if hasattr(data, 'attrs') and 'resistance_levels' in data.attrs:
                support_resistance['resistance'] = data.attrs['resistance_levels']

            # Compile results
            result = {
                'moving_averages': moving_averages,
                'momentum': momentum,
                'volatility': volatility,
                'support_resistance': support_resistance
            }

            return result

        except Exception as e:
            logger.error(f"Error extracting technical indicators: {e}")
            return {}
