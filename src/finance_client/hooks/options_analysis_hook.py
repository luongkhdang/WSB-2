import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple

# Import utilities
from src.finance_client.utilities.yfinance_api import get_option_chain
from src.finance_client.utilities.data_cache import DataCache
from src.finance_client.utilities.options_greeks import (
    calculate_call_greeks,
    calculate_put_greeks,
    calculate_iv_surface,
    calculate_historical_volatility,
    get_risk_free_rate
)

logger = logging.getLogger(__name__)


class OptionsAnalysisHook:
    """Hook for fetching and analyzing options data"""

    def __init__(self, use_cache: bool = True, cache_dir: str = "./data_cache"):
        """
        Initialize the options analysis hook

        Args:
            use_cache: Whether to use caching for API calls
            cache_dir: Directory to store cached data
        """
        logger.info("Initializing OptionsAnalysisHook")
        self.use_cache = use_cache

        if use_cache:
            self.cache = DataCache(cache_dir)

    def get_options_sentiment(self, symbol="SPY", expiration_date=None) -> Dict[str, Any]:
        """
        Get options sentiment data

        Args:
            symbol: Ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)

        Returns:
            Dictionary containing options sentiment analysis
        """
        try:
            # Check cache first if enabled (shorter cache time for options data)
            if self.use_cache:
                cached_data = self.cache.get("get_options_sentiment", symbol, expiration_date,
                                             max_age_hours=4)  # Options data changes frequently
                if cached_data is not None:
                    return cached_data

            # Get options data
            options_data = get_option_chain(symbol, expiration_date)

            if not options_data:
                logger.warning(f"No options data available for {symbol}")
                return self._get_fallback_sentiment_data(expiration_date)

            # Extract chains and expiration date
            calls = options_data.get('calls')
            puts = options_data.get('puts')
            expiration_date = options_data.get('expiration_date')

            if calls.empty or puts.empty:
                logger.warning(f"Empty options data for {symbol}")
                return self._get_fallback_sentiment_data(expiration_date)

            # Get current price from options data or estimate it if not available
            current_price = options_data.get('current_price')

            if current_price is None:
                logger.warning(
                    f"Current price not available in options data for {symbol}, estimating from options chain")
                # Estimate current price from ATM options as fallback
                if 'lastPrice' in calls.columns and not calls['lastPrice'].empty:
                    strikes = sorted(calls['strike'].unique())
                    middle_index = len(strikes) // 2
                    current_price = strikes[middle_index]
                else:
                    # Use a default value if we can't estimate
                    current_price = 100
                    logger.warning(
                        f"Could not estimate current price for {symbol}")
            else:
                logger.info(
                    f"Using provided current price {current_price} for {symbol}")

            # Calculate volume ratios
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()

            # Avoid division by zero
            call_put_volume_ratio = total_call_volume / \
                total_put_volume if total_put_volume > 0 else 1.5
            call_put_oi_ratio = total_call_oi / total_put_oi if total_put_oi > 0 else 1.5

            # Enhanced analysis - filter for near-the-money options
            atm_range = 0.05  # 5% from current price

            atm_calls = calls[(calls['strike'] >= current_price * (1 - atm_range)) &
                              (calls['strike'] <= current_price * (1 + atm_range))]

            atm_puts = puts[(puts['strike'] >= current_price * (1 - atm_range)) &
                            (puts['strike'] <= current_price * (1 + atm_range))]

            # Calculate ATM ratios (these are more significant)
            atm_call_volume = atm_calls['volume'].sum(
            ) if not atm_calls.empty else 0
            atm_put_volume = atm_puts['volume'].sum(
            ) if not atm_puts.empty else 0
            atm_call_oi = atm_calls['openInterest'].sum(
            ) if not atm_calls.empty else 0
            atm_put_oi = atm_puts['openInterest'].sum(
            ) if not atm_puts.empty else 0

            atm_call_put_volume_ratio = atm_call_volume / \
                atm_put_volume if atm_put_volume > 0 else 1.5
            atm_call_put_oi_ratio = atm_call_oi / atm_put_oi if atm_put_oi > 0 else 1.5

            # Calculate IV metrics
            call_iv_avg = calls['impliedVolatility'].mean(
            ) if 'impliedVolatility' in calls.columns else 0.25
            put_iv_avg = puts['impliedVolatility'].mean(
            ) if 'impliedVolatility' in puts.columns else 0.25

            # Calculate IV skew (especially for ATM options)
            iv_skew = 0
            if 'impliedVolatility' in calls.columns and 'impliedVolatility' in puts.columns:
                atm_call_iv_avg = atm_calls['impliedVolatility'].mean(
                ) if not atm_calls.empty else call_iv_avg
                atm_put_iv_avg = atm_puts['impliedVolatility'].mean(
                ) if not atm_puts.empty else put_iv_avg
                iv_skew = atm_put_iv_avg - atm_call_iv_avg
            else:
                iv_skew = put_iv_avg - call_iv_avg

            # Analyze sentiment based on ratios and IV skew
            sentiment = "neutral"
            sentiment_factors = []

            # Volume ratio analysis
            if call_put_volume_ratio > 1.5:
                sentiment_factors.append("high_call_volume")
            elif call_put_volume_ratio < 0.7:
                sentiment_factors.append("high_put_volume")

            # Open interest analysis
            if call_put_oi_ratio > 1.5:
                sentiment_factors.append("high_call_interest")
            elif call_put_oi_ratio < 0.7:
                sentiment_factors.append("high_put_interest")

            # IV skew analysis
            if iv_skew < -0.05:
                # Higher call IV = defensive positioning
                sentiment_factors.append("call_skew")
            elif iv_skew > 0.05:
                # Higher put IV = risk of downside
                sentiment_factors.append("put_skew")

            # Determine overall sentiment
            bullish_factors = ["high_call_volume",
                               "high_call_interest", "put_skew"]
            bearish_factors = ["high_put_volume",
                               "high_put_interest", "call_skew"]

            bullish_count = sum(
                1 for factor in sentiment_factors if factor in bullish_factors)
            bearish_count = sum(
                1 for factor in sentiment_factors if factor in bearish_factors)

            if bullish_count > bearish_count + 1:
                sentiment = "strongly_bullish"
            elif bullish_count > bearish_count:
                sentiment = "bullish"
            elif bearish_count > bullish_count + 1:
                sentiment = "strongly_bearish"
            elif bearish_count > bullish_count:
                sentiment = "bearish"

            # Extra analysis for premium direction
            premium_direction = "neutral"
            if atm_call_volume > 100 and atm_put_volume > 100:
                if 'lastPrice' in atm_calls.columns and 'lastPrice' in atm_puts.columns:
                    avg_call_premium = atm_calls['lastPrice'].mean(
                    ) if not atm_calls.empty else 0
                    avg_put_premium = atm_puts['lastPrice'].mean(
                    ) if not atm_puts.empty else 0

                    premium_ratio = avg_call_premium / avg_put_premium if avg_put_premium > 0 else 1.0

                    if premium_ratio > 1.3:
                        premium_direction = "call_premium"  # More paid for calls = bullish
                    elif premium_ratio < 0.7:
                        premium_direction = "put_premium"  # More paid for puts = bearish

            result = {
                "expiration_date": expiration_date,
                "call_put_volume_ratio": round(call_put_volume_ratio, 2),
                "call_put_oi_ratio": round(call_put_oi_ratio, 2),
                "atm_call_put_volume_ratio": round(atm_call_put_volume_ratio, 2),
                "atm_call_put_oi_ratio": round(atm_call_put_oi_ratio, 2),
                "call_iv_avg": round(float(call_iv_avg), 2),
                "put_iv_avg": round(float(put_iv_avg), 2),
                "iv_skew": round(float(iv_skew), 2),
                "sentiment": sentiment,
                "sentiment_factors": sentiment_factors,
                "premium_direction": premium_direction,
                "total_call_volume": int(total_call_volume),
                "total_put_volume": int(total_put_volume),
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
                "is_fallback": False
            }

            # Cache the result if enabled
            if self.use_cache:
                self.cache.set("get_options_sentiment",
                               result, symbol, expiration_date)

            return result

        except Exception as e:
            logger.error(f"Error getting options sentiment for {symbol}: {e}")
            logger.exception(e)
            return self._get_fallback_sentiment_data(expiration_date)

    def get_options_for_spread(self, symbol, expiration_date=None) -> Optional[Dict[str, Any]]:
        """
        Get options data formatted for credit spread analysis

        Args:
            symbol: Ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)

        Returns:
            Dictionary containing credit spread analysis
        """
        try:
            # Skip options analysis for assets that typically don't have standard options
            no_options_assets = ["VIX", "^VIX",
                                 "BTC-USD", "ETH-USD", "XRP-USD"]
            if symbol in no_options_assets:
                logger.info(
                    f"Skipping options analysis for {symbol} as it typically doesn't have standard options data")
                return {
                    "ticker": symbol,
                    "current_price": 0.0,
                    "expiration_date": None,
                    "bull_put_spreads": [],
                    "bear_call_spreads": [],
                    "no_options_available": True,
                    "reason": f"{symbol} doesn't have standard options data available"
                }

            # Check cache first if enabled
            if self.use_cache:
                cached_data = self.cache.get("get_options_for_spread", symbol, expiration_date,
                                             max_age_hours=4)
                if cached_data is not None:
                    return cached_data

            # Get options chain
            options_data = get_option_chain(symbol, expiration_date)

            # Check if options data is available
            if not options_data or "error" in options_data:
                logger.warning(f"No options data available for {symbol}")
                return None

            # Extract chains and expiration date
            calls = options_data.get('calls')
            puts = options_data.get('puts')
            expiration_date = options_data.get('expiration_date')

            # Get current price from options data or estimate it if not available
            current_price = options_data.get('current_price')

            if current_price is None:
                logger.warning(
                    f"Current price not available in options data for {symbol}, estimating from options chain")
                # Estimate current price from ATM options as fallback
                if 'lastPrice' in calls.columns and not calls['lastPrice'].empty:
                    strikes = sorted(calls['strike'].unique())
                    middle_index = len(strikes) // 2
                    current_price = strikes[middle_index]
                else:
                    logger.warning(
                        f"Could not estimate current price for {symbol}")
                    return None
            else:
                logger.info(
                    f"Using provided current price {current_price} for {symbol}")

            # Calculate days to expiration for more accurate Greeks
            today = datetime.now()
            try:
                expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
                days_to_expiration = (expiry - today).days
                t = days_to_expiration / 365  # Time to expiry in years
            except Exception as e:
                logger.warning(
                    f"Error calculating days to expiration: {e}. Using default of 30 days.")
                days_to_expiration = 30
                t = 30 / 365

            # Get risk-free rate
            r = get_risk_free_rate()

            # Calculate historical volatility for more accurate calculations
            historical_vol = 0.3  # Default value

            # Calculate Greeks for more accurate spread analysis
            calls = calculate_call_greeks(
                calls, current_price, t, r, historical_vol)
            puts = calculate_put_greeks(
                puts, current_price, t, r, historical_vol)

            # Fix put delta sign (puts should have negative delta)
            if 'delta' in puts.columns:
                puts['delta'] = -abs(puts['delta'])

            # Filter by delta ranges for better spread selection
            # For bull put spreads, we want OTM puts with deltas between -0.10 and -0.40
            # For bear call spreads, we want OTM calls with deltas between 0.10 and 0.40
            bull_put_candidates = puts[(
                puts['delta'] >= -0.40) & (puts['delta'] <= -0.10)]
            bear_call_candidates = calls[(
                calls['delta'] >= 0.10) & (calls['delta'] <= 0.40)]

            # Calculate bull put spreads (for bullish outlook)
            bull_put_spreads = []

            # Sort puts by strike price descending (higher to lower)
            sorted_puts = bull_put_candidates.sort_values(
                'strike', ascending=False)

            # Process each short put option
            for i, short_put in sorted_puts.iterrows():
                short_strike = short_put.strike
                short_premium = short_put.lastPrice
                short_delta = short_put.delta if 'delta' in short_put and not pd.isna(
                    short_put.delta) else -0.3
                short_iv = short_put.impliedVolatility if 'impliedVolatility' in short_put and not pd.isna(
                    short_put.impliedVolatility) else 0.3

                # Calculate distance from current price
                distance_pct = (
                    (short_strike - current_price) / current_price) * 100

                # Skip if the strike price is too far OTM or ITM
                if distance_pct < -25 or distance_pct > 5:
                    continue

                # Find potential long puts with strikes below the short put
                potential_long_puts = sorted_puts[sorted_puts['strike']
                                                  < short_strike]

                # Find long puts that are 1-5 strikes away (typical width)
                for _, long_put in potential_long_puts.iterrows():
                    long_strike = long_put.strike
                    long_premium = long_put.lastPrice
                    long_delta = long_put.delta if 'delta' in long_put and not pd.isna(
                        long_put.delta) else -0.15

                    # Width in points
                    width = short_strike - long_strike

                    # Skip spreads that are too narrow or too wide
                    if width < 1 or width > 10:
                        continue

                    # Calculate credit received
                    credit = short_premium - long_premium
                    max_risk = width - credit

                    # Skip spreads with no credit or negligible credit
                    if credit <= 0.05:
                        continue

                    # Calculate reward-to-risk ratio and probability metrics
                    reward_risk_ratio = credit / max_risk if max_risk > 0 else 0
                    return_on_capital = (credit / width) * \
                        100 if width > 0 else 0
                    probability_of_profit = (
                        1 + short_delta) * 100 if short_delta else 70

                    # Only include spreads with a reasonable reward-to-risk ratio
                    if reward_risk_ratio >= 0.15 and return_on_capital >= 8:
                        bull_put_spreads.append({
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'width': width,
                            'credit': credit,
                            'max_risk': max_risk,
                            'return_on_risk': reward_risk_ratio,
                            'return_on_capital': return_on_capital,
                            'short_delta': float(short_delta),
                            'long_delta': float(long_delta),
                            'short_iv': float(short_iv),
                            'distance_pct': distance_pct,
                            'probability_of_profit': probability_of_profit
                        })

            # Calculate bear call spreads (for bearish outlook)
            bear_call_spreads = []

            # Sort calls by strike price ascending (lower to higher)
            sorted_calls = bear_call_candidates.sort_values(
                'strike', ascending=True)

            # Process each short call option
            for i, short_call in sorted_calls.iterrows():
                short_strike = short_call.strike
                short_premium = short_call.lastPrice
                short_delta = short_call.delta if 'delta' in short_call and not pd.isna(
                    short_call.delta) else 0.3
                short_iv = short_call.impliedVolatility if 'impliedVolatility' in short_call and not pd.isna(
                    short_call.impliedVolatility) else 0.3

                # Calculate distance from current price
                distance_pct = (
                    (short_strike - current_price) / current_price) * 100

                # Skip if the strike price is too far OTM or ITM
                if distance_pct < -5 or distance_pct > 25:
                    continue

                # Find potential long calls with strikes above the short call
                potential_long_calls = sorted_calls[sorted_calls['strike']
                                                    > short_strike]

                # Find long calls that are 1-5 strikes away (typical width)
                for _, long_call in potential_long_calls.iterrows():
                    long_strike = long_call.strike
                    long_premium = long_call.lastPrice
                    long_delta = long_call.delta if 'delta' in long_call and not pd.isna(
                        long_call.delta) else 0.15

                    # Width in points
                    width = long_strike - short_strike

                    # Skip spreads that are too narrow or too wide
                    if width < 1 or width > 10:
                        continue

                    # Calculate credit received
                    credit = short_premium - long_premium
                    max_risk = width - credit

                    # Skip spreads with no credit or negligible credit
                    if credit <= 0.05:
                        continue

                    # Calculate reward-to-risk ratio and probability metrics
                    reward_risk_ratio = credit / max_risk if max_risk > 0 else 0
                    return_on_capital = (credit / width) * \
                        100 if width > 0 else 0
                    probability_of_profit = (
                        1 - short_delta) * 100 if short_delta else 70

                    # Only include spreads with a reasonable reward-to-risk ratio
                    if reward_risk_ratio >= 0.15 and return_on_capital >= 8:
                        bear_call_spreads.append({
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'width': width,
                            'credit': credit,
                            'max_risk': max_risk,
                            'return_on_risk': reward_risk_ratio,
                            'return_on_capital': return_on_capital,
                            'short_delta': float(short_delta),
                            'long_delta': float(long_delta),
                            'short_iv': float(short_iv),
                            'distance_pct': distance_pct,
                            'probability_of_profit': probability_of_profit
                        })

            # Sort spreads by return on capital (better metric than return on risk)
            bull_put_spreads.sort(
                key=lambda x: x['return_on_capital'], reverse=True)
            bear_call_spreads.sort(
                key=lambda x: x['return_on_capital'], reverse=True)

            # Analyze IV skew to identify favorable spread setups
            # Calculate IV skew (positive means puts are more expensive than calls at equivalent deltas)
            puts_30_delta = puts[(puts['delta'] >= -0.35)
                                 & (puts['delta'] <= -0.25)]
            calls_30_delta = calls[(calls['delta'] >= 0.25)
                                   & (calls['delta'] <= 0.35)]

            avg_30d_put_iv = puts_30_delta['impliedVolatility'].mean(
            ) if not puts_30_delta.empty and 'impliedVolatility' in puts_30_delta.columns else None
            avg_30d_call_iv = calls_30_delta['impliedVolatility'].mean(
            ) if not calls_30_delta.empty and 'impliedVolatility' in calls_30_delta.columns else None

            iv_skew = None
            if avg_30d_put_iv is not None and avg_30d_call_iv is not None and not pd.isna(avg_30d_put_iv) and not pd.isna(avg_30d_call_iv):
                iv_skew = float(avg_30d_put_iv - avg_30d_call_iv)

            # Determine if one spread type is favored by IV skew
            skew_recommendation = "neutral"
            if iv_skew is not None:
                if iv_skew > 0.05:
                    # Put IV is higher, selling puts may be more profitable
                    skew_recommendation = "bull_put_favored"
                elif iv_skew < -0.05:
                    # Call IV is higher, selling calls may be more profitable
                    skew_recommendation = "bear_call_favored"

            result = {
                'ticker': symbol,
                'expiration_date': expiration_date,
                'current_price': current_price,
                'days_to_expiration': days_to_expiration,
                'bull_put_spreads': bull_put_spreads,
                'bear_call_spreads': bear_call_spreads,
                'iv_skew': iv_skew,
                'skew_recommendation': skew_recommendation,
                'historical_volatility': historical_vol
            }

            # Cache the result if enabled
            if self.use_cache:
                self.cache.set("get_options_for_spread",
                               result, symbol, expiration_date)

            return result

        except Exception as e:
            logger.error(
                f"Error getting options for spread analysis for {symbol}: {e}")
            logger.exception(e)
            return None

    def _get_fallback_sentiment_data(self, expiration_date=None) -> Dict[str, Any]:
        """
        Generate fallback options sentiment data when API fails

        Args:
            expiration_date: Optional expiration date to include in fallback data

        Returns:
            Dictionary with fallback options sentiment data
        """
        logger.warning("Using fallback options sentiment data")

        if not expiration_date:
            now = datetime.now()
            target_date = now + timedelta(days=14)
            expiration_date = target_date.strftime('%Y-%m-%d')

        return {
            "expiration_date": expiration_date,
            "call_put_volume_ratio": 1.05,
            "call_put_oi_ratio": 1.03,
            "atm_call_put_volume_ratio": 1.08,
            "atm_call_put_oi_ratio": 1.05,
            "call_iv_avg": 0.25,
            "put_iv_avg": 0.26,
            "iv_skew": -0.01,
            "sentiment": "neutral",
            "sentiment_factors": [],
            "premium_direction": "neutral",
            "total_call_volume": 150000,
            "total_put_volume": 143000,
            "total_call_oi": 150000,
            "total_put_oi": 143000,
            "is_fallback": True,
            "note": "This is fallback data. Real options data could not be retrieved."
        }
