import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import copy
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yfinance_client')

class YFinanceClient:
    def __init__(self):
        logger.info("Initializing YFinanceClient")
        try:
            # We'll avoid creating Ticker objects in __init__ to simplify things
            self.tickers = {
                "SPY": "SPY",
                "VIX": "^VIX",
                "TSLA": "TSLA"
            }
            logger.info("YFinanceClient initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing YFinanceClient: {e}")
            raise
    
    def _get_ticker_data(self, symbol, period="1d", interval="1h"):
        """Get ticker data using yf.download instead of Ticker objects"""
        try:
            # Use yf.download which should be more stable than Ticker objects
            data = yf.download(symbol, period=period, interval=interval)
            logger.info(f"Downloaded data for {symbol}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()  # Return empty dataframe on error
    
    def get_market_data(self):
        """Get SPY market trend and direction data"""
        try:
            # Get historical data
            spy_history = self._get_ticker_data("SPY", period="5d", interval="1h")
            
            if spy_history.empty:
                logger.error("Failed to retrieve SPY data")
                return None
            
            # Calculate basic metrics
            current_price = spy_history['Close'].iloc[-1].item() if not spy_history.empty else None
            open_price = spy_history['Open'].iloc[-1].item() if not spy_history.empty else None
            day_high = spy_history['High'].iloc[-1].item() if not spy_history.empty else None
            day_low = spy_history['Low'].iloc[-1].item() if not spy_history.empty else None
            
            # Calculate 50-day average if we have enough data
            fifty_day_avg = None
            if len(spy_history) >= 50:
                fifty_day_avg = spy_history['Close'].rolling(window=50).mean().iloc[-1].item()
            
            # Calculate 200-day average if we have enough data
            two_hundred_day_avg = None
            if len(spy_history) >= 200:
                two_hundred_day_avg = spy_history['Close'].rolling(window=200).mean().iloc[-1].item()
            
            # Basic market data
            spy_data = {
                "regularMarketPrice": current_price,
                "open": open_price,
                "dayHigh": day_high,
                "dayLow": day_low,
                "previousClose": spy_history['Close'].iloc[-2].item() if len(spy_history) > 1 else None,
                "regularMarketVolume": spy_history['Volume'].iloc[-1].item() if not spy_history.empty else None,
                "averageVolume": spy_history['Volume'].mean() if not spy_history.empty else None,
                "fiftyDayAverage": fifty_day_avg,
                "twoHundredDayAverage": two_hundred_day_avg
            }
            
            return {
                "info": spy_data,
                "history": spy_history
            }
        except Exception as e:
            logger.error(f"Error fetching SPY market data: {e}")
            return None
    
    def get_options_sentiment(self, expiration_date=None):
        """Get options sentiment analysis data based on SPY options"""
        logger.info("Fetching options sentiment data from SPY options")
        try:
            # Use yfinance Ticker directly for options data
            from yfinance import Ticker
            ticker = Ticker("SPY")
            
            # Get current price for reference
            current_price = ticker.history(period='1d')['Close'].iloc[-1].item()
            logger.info(f"Current SPY price: {current_price}")
            
            # Get available expiration dates if none provided
            if not expiration_date:
                try:
                    expirations = ticker.options
                    if not expirations:
                        logger.warning("No options expiration dates found for SPY")
                        return self._get_fallback_sentiment_data(expiration_date)
                    
                    # Find nearest expiration date (ideally 7-14 days out for highest liquidity)
                    from datetime import datetime, timedelta
                    now = datetime.now()
                    target_date = now + timedelta(days=14)
                    
                    # Convert expirations to datetime objects for comparison
                    exp_dates = []
                    for exp in expirations:
                        try:
                            exp_date = datetime.strptime(exp, '%Y-%m-%d')
                            exp_dates.append((exp_date, exp))
                        except ValueError:
                            continue
                    
                    # Sort by absolute difference from target date
                    exp_dates.sort(key=lambda x: abs((x[0] - target_date).days))
                    
                    # Select closest date
                    if exp_dates:
                        expiration_date = exp_dates[0][1]
                    else:
                        logger.warning("Could not determine optimal expiration date")
                        return self._get_fallback_sentiment_data(expiration_date)
                        
                except Exception as e:
                    logger.error(f"Error getting options expiration dates: {e}")
                    return self._get_fallback_sentiment_data(expiration_date)
            
            # Get option chain for the selected expiration
            try:
                logger.info(f"Fetching option chain for expiration date: {expiration_date}")
                chain = ticker.option_chain(expiration_date)
                
                if not hasattr(chain, 'calls') or not hasattr(chain, 'puts'):
                    logger.warning(f"Incomplete options chain for SPY at {expiration_date}")
                    return self._get_fallback_sentiment_data(expiration_date)
                
                # Calculate call/put volume ratio
                total_call_volume = chain.calls['volume'].sum() if 'volume' in chain.calls.columns else 0
                total_put_volume = chain.puts['volume'].sum() if 'volume' in chain.puts.columns else 0
                
                if total_put_volume == 0:  # Avoid division by zero
                    call_put_volume_ratio = 1.5  # Default bullish if no put volume
                else:
                    call_put_volume_ratio = total_call_volume / total_put_volume
                
                # Calculate call/put open interest ratio
                total_call_oi = chain.calls['openInterest'].sum() if 'openInterest' in chain.calls.columns else 0
                total_put_oi = chain.puts['openInterest'].sum() if 'openInterest' in chain.puts.columns else 0
                
                if total_put_oi == 0:  # Avoid division by zero
                    call_put_oi_ratio = 1.5  # Default bullish if no put open interest
                else:
                    call_put_oi_ratio = total_call_oi / total_put_oi
                
                # Calculate average IV for ATM options (within 5% of current price)
                atm_range_upper = current_price * 1.05
                atm_range_lower = current_price * 0.95
                
                atm_calls = chain.calls[
                    (chain.calls['strike'] >= atm_range_lower) & 
                    (chain.calls['strike'] <= atm_range_upper)
                ]
                
                atm_puts = chain.puts[
                    (chain.puts['strike'] >= atm_range_lower) & 
                    (chain.puts['strike'] <= atm_range_upper)
                ]
                
                call_iv_avg = atm_calls['impliedVolatility'].mean() if 'impliedVolatility' in atm_calls.columns and not atm_calls.empty else 0.25
                put_iv_avg = atm_puts['impliedVolatility'].mean() if 'impliedVolatility' in atm_puts.columns and not atm_puts.empty else 0.28
                
                # Calculate IV skew (negative means calls more expensive, bullish)
                iv_skew = put_iv_avg - call_iv_avg
                
                # Analyze sentiment based on ratios
                sentiment = "neutral"
                if call_put_volume_ratio > 1.2 and iv_skew < -0.02:
                    sentiment = "strongly_bullish"
                elif call_put_volume_ratio > 1.1 or iv_skew < -0.01:
                    sentiment = "bullish"
                elif call_put_volume_ratio < 0.8 and iv_skew > 0.02:
                    sentiment = "strongly_bearish"
                elif call_put_volume_ratio < 0.9 or iv_skew > 0.01:
                    sentiment = "bearish"
                
                return {
                    "expiration_date": expiration_date,
                    "call_put_volume_ratio": round(call_put_volume_ratio, 2),
                    "call_put_oi_ratio": round(call_put_oi_ratio, 2),
                    "call_iv_avg": round(float(call_iv_avg), 2),
                    "put_iv_avg": round(float(put_iv_avg), 2),
                    "iv_skew": round(float(iv_skew), 2),
                    "sentiment": sentiment,
                    "total_call_volume": int(total_call_volume),
                    "total_put_volume": int(total_put_volume)
                }
                
            except Exception as e:
                logger.error(f"Error analyzing options chain: {e}")
                return self._get_fallback_sentiment_data(expiration_date)
            
        except Exception as e:
            logger.error(f"Error in get_options_sentiment: {e}")
            return self._get_fallback_sentiment_data(expiration_date)
    
    def _get_fallback_sentiment_data(self, expiration_date=None):
        """Generate fallback options sentiment data when API fails"""
        logger.warning("Using fallback options sentiment data")
        
        from datetime import datetime, timedelta
        if not expiration_date:
            now = datetime.now()
            target_date = now + timedelta(days=14)
            expiration_date = target_date.strftime('%Y-%m-%d')
            
        return {
            "expiration_date": expiration_date,
            "call_put_volume_ratio": 1.05,
            "call_put_oi_ratio": 1.03,
            "call_iv_avg": 0.25,
            "put_iv_avg": 0.26,
            "iv_skew": -0.01,
            "sentiment": "neutral",
            "total_call_volume": 150000,
            "total_put_volume": 143000,
            "note": "This is fallback data. Real options data could not be retrieved."
        }
    
    def get_volatility_filter(self):
        """Get VIX data for market volatility filtering"""
        try:
            vix_data = self._get_ticker_data("^VIX", period="1d")
            
            if vix_data.empty:
                logger.error("Failed to retrieve VIX data")
                logger.info("Using fallback VIX data to continue workflow")
                # Provide fallback data
                return {
                    "price": 19.29,  # Moderate VIX level as fallback
                    "stability": "normal",
                    "risk_adjustment": 1.0,
                    "adjustment_note": "Standard position sizing (fallback data)"
                }
            
            # Extract the last VIX price
            try:
                vix_price = float(vix_data['Close'].iloc[-1].item())
                logger.info(f"Current VIX price: {vix_price}")
            except Exception as e:
                logger.error(f"Error extracting VIX price from data: {e}")
                logger.info("Using fallback VIX data to continue workflow")
                return {
                    "price": 19.29,  # Moderate VIX level as fallback
                    "stability": "normal",
                    "risk_adjustment": 1.0,
                    "adjustment_note": "Standard position sizing (fallback data)"
                }
            
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
            
            return {
                "price": vix_price,
                "stability": stability,
                "risk_adjustment": risk_adjustment,
                "adjustment_note": adjustment_note
            }
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            logger.info("Using fallback VIX data to continue workflow")
            return {
                "price": 22.5,  # Moderate VIX level as fallback
                "stability": "normal",
                "risk_adjustment": 1.0,
                "adjustment_note": "Standard position sizing (fallback data)"
            }
    

    def get_stock_analysis(self, ticker_symbol="TSLA"):
        """Get stock analysis data for specified ticker (default: TSLA)"""
        try:
            # Get stock data - short term (5d hourly) and longer term (3mo daily)
            stock_history = self._get_ticker_data(ticker_symbol, period="5d", interval="1h")
            stock_history_3_month = self._get_ticker_data(ticker_symbol, period="3mo", interval="1d")

            if stock_history.empty:
                logger.error(f"Failed to retrieve data for {ticker_symbol}")
                return None

            # Basic stock data
            current_price = stock_history['Close'].iloc[-1].item()
            open_price = stock_history['Open'].iloc[-1].item()
            day_high = stock_history['High'].iloc[-1].item()
            day_low = stock_history['Low'].iloc[-1].item()
            recent_volume = stock_history['Volume'].iloc[-5:].mean() if len(stock_history) >= 5 else None

            # Support and resistance
            try:
                lows_array = stock_history['Low'].tail(20).to_numpy()
                highs_array = stock_history['High'].tail(20).to_numpy()
                lows_array.sort()
                highs_array = -np.sort(-highs_array)
                support = np.mean(lows_array[:3])
                resistance = np.mean(highs_array[:3])
                support_distance = ((current_price - support) / current_price) * 100
                resistance_distance = ((resistance - current_price) / current_price) * 100
                near_support = support_distance <= 2.0
                near_resistance = resistance_distance <= 2.0
            except Exception as e:
                logger.error(f"Error calculating support/resistance: {e}")
                support = current_price * 0.95
                resistance = current_price * 1.05
                near_support = False
                near_resistance = False

            stock_data = {
                "ticker": ticker_symbol,
                "regularMarketPrice": current_price,
                "open": open_price,
                "dayHigh": day_high,
                "dayLow": day_low,
                "previousClose": stock_history['Close'].iloc[-2].item() if len(stock_history) > 1 else None,
                "regularMarketVolume": stock_history['Volume'].iloc[-1].item(),
                "averageVolume": recent_volume,
                "support": support,
                "resistance": resistance,
                "near_support": bool(near_support),
                "near_resistance": bool(near_resistance),
                "fiftyTwoWeekHigh": stock_history['High'].max().item(),
                "fiftyTwoWeekLow": stock_history['Low'].min().item()
            }

            # ATR calculation
            atr = None
            atr_percent = None
            volatility = "unknown"
            volatility_note = "Unable to determine volatility"
            try:
                high_low = stock_history['High'] - stock_history['Low']
                high_close = abs(stock_history['High'] - stock_history['Close'].shift())
                low_close = abs(stock_history['Low'] - stock_history['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1].item() if len(true_range) >= 14 else None

                if atr is not None and current_price is not None:
                    atr_percent = (atr / current_price) * 100
                    if atr_percent < 1.0:
                        volatility = "low"
                        volatility_note = "Stable stock (+5 to Risk)"
                    elif atr_percent < 2.0:
                        volatility = "normal"
                        volatility_note = "Normal volatility"
                    else:
                        volatility = "high"
                        volatility_note = "Volatile, tighten stop (-5 unless Gamble Score high)"
            except Exception as e:
                logger.error(f"Error calculating ATR: {e}")
                atr = None
                atr_percent = None
                volatility = "unknown"
                volatility_note = "Error calculating volatility"

            # EMA/trend analysis
            current_ema9 = None
            current_ema21 = None
            trend = "unknown"
            trend_note = "Insufficient data for EMA calculation"

            try:
                if not stock_history_3_month.empty and len(stock_history_3_month) >= 21:
                    logger.info(f"Using 3-month data for EMA calculation, shape: {stock_history_3_month.shape}")
                    ema9 = stock_history_3_month['Close'].ewm(span=9, adjust=False).mean()
                    ema21 = stock_history_3_month['Close'].ewm(span=21, adjust=False).mean()
                    current_ema9 = ema9.iloc[-1].item()
                    current_ema21 = ema21.iloc[-1].item()
                else:
                    logger.warning("Falling back to short-term data for EMA")
                    ema9 = stock_history['Close'].ewm(span=9, adjust=False).mean()
                    ema21 = stock_history['Close'].ewm(span=21, adjust=False).mean()
                    current_ema9 = ema9.iloc[-1].item()
                    current_ema21 = ema21.iloc[-1].item()

                if current_ema9 is not None and current_ema21 is not None:
                    if current_price > current_ema9 and current_ema9 > current_ema21:
                        trend = "bullish"
                        trend_note = "Price > EMA9 > EMA21 (+10 to Technicals)"
                    elif current_price < current_ema9 and current_ema9 < current_ema21:
                        trend = "bearish"
                        trend_note = "Price < EMA9 < EMA21 (+10 to Technicals if bearish setup)"
                    elif current_ema9 > current_ema21:
                        trend = "moderately_bullish"
                        trend_note = "EMA9 > EMA21 but price position unclear"
                    elif current_ema9 < current_ema21:
                        trend = "moderately_bearish"
                        trend_note = "EMA9 < EMA21 but price position unclear"
                    else:
                        trend = "neutral"
                        trend_note = "No clear EMA trend"
            except Exception as e:
                logger.error(f"Error determining trend: {e}")
                trend = "unknown"
                trend_note = "Error in trend calculation"
                current_ema9 = None
                current_ema21 = None

            return {
                "info": stock_data,
                "history": stock_history.to_dict(),
                "history_3_month": stock_history_3_month.to_dict(),
                "technical": {
                    "atr": float(atr) if atr is not None else None,
                    "atr_percent": float(atr_percent) if atr_percent is not None else None,
                    "volatility": volatility,
                    "volatility_note": volatility_note,
                    "ema9": current_ema9,
                    "ema21": current_ema21,
                    "trend": trend,
                    "trend_note": trend_note,
                    "support": float(support),
                    "resistance": float(resistance),
                    "near_support": bool(near_support),
                    "near_resistance": bool(near_resistance)
                }
            }

        except Exception as e:
            logger.error(f"Error fetching {ticker_symbol} stock analysis: {e}")
            return None

    
    def get_options_for_spread(self, ticker_symbol):
        """
        Get options data optimized for finding credit spread opportunities
        """
        logger.info(f"Getting options data for credit spreads on {ticker_symbol}")
        
        try:
            # Get stock info
            ticker = yf.Ticker(ticker_symbol)
            stock_info = ticker.info
            current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
            
            if not current_price or current_price <= 0:
                logger.warning(f"Invalid current price for {ticker_symbol}: {current_price}")
                return None
            
            logger.info(f"{ticker_symbol} current price: ${current_price}")
            
            # Get options expirations
            expirations = ticker.options
            if not expirations or len(expirations) == 0:
                logger.warning(f"No options expirations found for {ticker_symbol}")
                return None
            
            # Sort expirations and find one that's 7-15 days out
            expirations = sorted(expirations)
            selected_expiration = None
            today = datetime.now().date()
            
            for exp_date in expirations:
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
                days_to_exp = (exp_date_obj - today).days
                
                if 7 <= days_to_exp <= 21:  # Allow a bit wider range than the ideal 7-15
                    selected_expiration = exp_date
                    break
            
            # If no ideal expiration found, take the closest one
            if not selected_expiration and expirations:
                selected_expiration = expirations[0]
            
            # Instead of fetching options directly, use our method with greeks
            options_with_greeks = self.get_option_chain_with_greeks(ticker_symbol, weeks=2)
            
            # Filter for selected expiration
            if selected_expiration and selected_expiration in options_with_greeks:
                greeks_data = options_with_greeks[selected_expiration]
            else:
                # If selected_expiration not found in greeks data, take the first available
                if options_with_greeks:
                    selected_expiration = list(options_with_greeks.keys())[0]
                    greeks_data = options_with_greeks[selected_expiration]
                else:
                    greeks_data = {}
            
            # Get options chain for the selected expiration
            try:
                calls = ticker.option_chain(selected_expiration).calls
                puts = ticker.option_chain(selected_expiration).puts
                
                logger.info(f"Analyzing {len(calls)} calls and {len(puts)} puts for {ticker_symbol} expiring {selected_expiration}")
            except Exception as e:
                logger.error(f"Error getting option chain for {ticker_symbol} expiring {selected_expiration}: {e}")
                return None
            
            # Define delta ranges for short and long options
            short_delta_min, short_delta_max = 0.20, 0.35  # ~65-80% OTM probability
            long_delta_min, long_delta_max = 0.10, 0.20    # Further OTM for protection
            
            # Calculate bull put spreads (for bullish outlook)
            bull_put_spreads = []
            
            # Sort puts by strike price descending (higher to lower)
            sorted_puts = puts.sort_values('strike', ascending=False)
            
            for i, put in sorted_puts.iterrows():
                strike_price = put.strike
                
                # Skip if strike is too far ITM or OTM
                if strike_price > current_price * 1.2 or strike_price < current_price * 0.5:
                    continue
                
                # Calculate estimated delta if not available (roughly)
                # Higher is more ITM, lower is more OTM
                delta_estimation = (current_price - strike_price) / current_price
                
                # Find puts that could be good short legs
                if (0.9 <= strike_price / current_price <= 0.97) or short_delta_min <= -delta_estimation <= short_delta_max:
                    short_put = put
                    short_strike = short_put.strike
                    
                    # Find a long put 5-10 points lower
                    long_put = None
                    ideal_long_strike_min = short_strike - 10
                    ideal_long_strike_max = short_strike - 3
                    
                    long_candidates = sorted_puts[
                        (sorted_puts.strike < short_strike) & 
                        (sorted_puts.strike >= ideal_long_strike_min) & 
                        (sorted_puts.strike <= ideal_long_strike_max)
                    ]
                    
                    if not long_candidates.empty:
                        long_put = long_candidates.iloc[0]
                        
                        # Calculate credit received
                        credit = short_put.lastPrice - long_put.lastPrice
                        max_risk = short_strike - long_put.strike - credit
                        
                        # Only include if credit is meaningful
                        if credit > 0.1:
                            bull_put_spreads.append({
                                'short_strike': short_strike,
                                'long_strike': long_put.strike,
                                'credit': credit,
                                'max_risk': max_risk,
                                'return_on_risk': credit / max_risk if max_risk > 0 else 0,
                                'short_delta': -delta_estimation
                            })
            
            # Calculate bear call spreads (for bearish outlook)
            bear_call_spreads = []
            
            # Sort calls by strike price ascending (lower to higher)
            sorted_calls = calls.sort_values('strike', ascending=True)
            
            for i, call in sorted_calls.iterrows():
                strike_price = call.strike
                
                # Skip if strike is too far ITM or OTM
                if strike_price < current_price * 0.8 or strike_price > current_price * 1.5:
                    continue
                
                # Calculate estimated delta if not available (roughly)
                delta_estimation = (strike_price - current_price) / current_price
                
                # Find calls that could be good short legs
                if (1.03 <= strike_price / current_price <= 1.1) or short_delta_min <= delta_estimation <= short_delta_max:
                    short_call = call
                    short_strike = short_call.strike
                    
                    # Find a long call 5-10 points higher
                    long_call = None
                    ideal_long_strike_min = short_strike + 3
                    ideal_long_strike_max = short_strike + 10
                    
                    long_candidates = sorted_calls[
                        (sorted_calls.strike > short_strike) & 
                        (sorted_calls.strike >= ideal_long_strike_min) & 
                        (sorted_calls.strike <= ideal_long_strike_max)
                    ]
                    
                    if not long_candidates.empty:
                        long_call = long_candidates.iloc[0]
                        
                        # Calculate credit received
                        credit = short_call.lastPrice - long_call.lastPrice
                        max_risk = long_call.strike - short_strike - credit
                        
                        # Only include if credit is meaningful
                        if credit > 0.1:
                            bear_call_spreads.append({
                                'short_strike': short_strike,
                                'long_strike': long_call.strike,
                                'credit': credit,
                                'max_risk': max_risk,
                                'return_on_risk': credit / max_risk if max_risk > 0 else 0,
                                'short_delta': delta_estimation
                            })
            
            # Sort spreads by risk/reward
            bull_put_spreads = sorted(bull_put_spreads, key=lambda x: x['return_on_risk'], reverse=True)
            bear_call_spreads = sorted(bear_call_spreads, key=lambda x: x['return_on_risk'], reverse=True)
            
            logger.info(f"Found {len(bull_put_spreads)} bull put spreads and {len(bear_call_spreads)} bear call spreads for {ticker_symbol}")
            
            return {
                "ticker": ticker_symbol,
                "current_price": current_price,
                "expiration_date": selected_expiration,
                "bull_put_spreads": bull_put_spreads,
                "bear_call_spreads": bear_call_spreads,
                "options_with_greeks": greeks_data  # Include the greeks data
            }
            
        except Exception as e:
            logger.error(f"Error getting options data for {ticker_symbol}: {str(e)}")
            return None
    
    def get_complete_analysis(self, ticker_symbol="TSLA"):
        """
        Get comprehensive analysis for trading decision following the Rule Book sequence:
        1. Analyze SPY: Determine general market trend
        2. Analyze Options of SPY: Assess general market direction
        3. Analyze Underlying Stock: Evaluate fundamental data 
        4. Analyze Credit Spreads: Identify profiting opportunities
        
        This now returns a structured dataset for AI analysis rather than performing the analysis itself.
        """
        logger.info(f"Gathering comprehensive data for {ticker_symbol} analysis")
        
        # Step a: Get SPY Market Data & VIX for Market Trend Analysis
        market_data = self.get_market_data()
        volatility_filter = self.get_volatility_filter()
        options_sentiment = self.get_options_sentiment()
        
        # Step b: Get stock data
        stock_data = self.get_stock_analysis(ticker_symbol)
        
        # Step c: Get options data for spreads
        options_spread_data = self.get_options_for_spread(ticker_symbol)
        
        # Prepare market context
        market_context = {
            "spy_price": market_data.get("info", {}).get("regularMarketPrice") if market_data else None,
            "spy_ema_data": {
                "50d_avg": market_data.get("info", {}).get("fiftyDayAverage") if market_data else None,
                "200d_avg": market_data.get("info", {}).get("twoHundredDayAverage") if market_data else None
            },
            "vix": {
                "price": volatility_filter.get("price") if volatility_filter else None,
                "stability": volatility_filter.get("stability") if volatility_filter else None,
                "risk_adjustment": volatility_filter.get("risk_adjustment") if volatility_filter else None
            },
            "options_sentiment": {
                "call_put_ratio": options_sentiment.get("call_put_volume_ratio") if options_sentiment else None,
                "iv_skew": options_sentiment.get("iv_skew") if options_sentiment else None
            }
        }
        
        # Prepare stock analysis data
        stock_analysis_data = None
        if stock_data:
            stock_analysis_data = {
                "price": stock_data.get("info", {}).get("regularMarketPrice"),
                "support": stock_data.get("technical", {}).get("support"),
                "resistance": stock_data.get("technical", {}).get("resistance"),
                "near_support": stock_data.get("technical", {}).get("near_support"),
                "near_resistance": stock_data.get("technical", {}).get("near_resistance"),
                "ema9": stock_data.get("technical", {}).get("ema9"),
                "ema21": stock_data.get("technical", {}).get("ema21"),
                "trend": stock_data.get("technical", {}).get("trend"),
                "atr": stock_data.get("technical", {}).get("atr"),
                "atr_percent": stock_data.get("technical", {}).get("atr_percent"),
                "volatility": stock_data.get("technical", {}).get("volatility")
            }
        
        # Prepare options data
        options_data = None
        if options_spread_data:
            # Extract just what's needed for the analysis
            bull_put_spreads = options_spread_data.get("bull_put_spreads", [])
            bear_call_spreads = options_spread_data.get("bear_call_spreads", [])
            
            options_data = {
                "current_price": options_spread_data.get("current_price"),
                "expiration_date": options_spread_data.get("expiration_date"),
                "bull_put_spreads": bull_put_spreads[:3] if bull_put_spreads else [],  # Just keep top 3
                "bear_call_spreads": bear_call_spreads[:3] if bear_call_spreads else []  # Just keep top 3
            }
        
        # Assemble structured dataset for AI analysis
        complete_analysis = {
            "ticker": ticker_symbol,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "market_context": market_context,
            "stock_analysis": stock_analysis_data,
            "options_data": options_data,
            
            # Keep the full detailed data for reference
            "full_data": {
            "market_data": market_data,
            "volatility_filter": volatility_filter,
                "options_sentiment": options_sentiment,
                "stock_data": stock_data,
                "options_spread_data": options_spread_data
            }
        }
        
        logger.info(f"Comprehensive data gathering complete for {ticker_symbol}")
        return complete_analysis
    
    def export_to_excel(self, data, filename="yfinance_analysis.xlsx"):
        """Export analysis data to Excel for further processing"""
        try:
            with pd.ExcelWriter(filename) as writer:
                # Create summary sheet
                summary_data = data.get("summary", {})
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name="Summary")
                
                # Create SPY sheet
                if data.get("market_data") and data["market_data"].get("info"):
                    spy_df = pd.DataFrame([data["market_data"]["info"]])
                    spy_df.to_excel(writer, sheet_name="SPY")
                
                # Create VIX sheet
                if data.get("volatility_filter"):
                    vix_df = pd.DataFrame([data["volatility_filter"]])
                    vix_df.to_excel(writer, sheet_name="VIX")
                
                # Create TSLA sheet
                if data.get("stock_analysis") and data["stock_analysis"].get("info"):
                    tsla_df = pd.DataFrame([data["stock_analysis"]["info"]])
                    tsla_tech_df = pd.DataFrame([data["stock_analysis"]["technical"]])
                    tsla_df.to_excel(writer, sheet_name="TSLA")
                    tsla_tech_df.to_excel(writer, sheet_name="TSLA_Technical")
                
                # Create Options sheet if available
                if data.get("options_spread"):
                    options_df = pd.DataFrame([{
                        "ticker": data["options_spread"]["ticker"],
                        "expiration": data["options_spread"]["expiration_date"]
                    }])
                    options_df.to_excel(writer, sheet_name="Options")
            
            logger.info(f"Data exported to {filename}")
            return f"Data exported to {filename}"
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return None
    
    def export_to_json(self, data, filename="yfinance_analysis.json"):
        """Export analysis data to JSON for Notion import"""
        try:
            # Deep copy to avoid modifying original data
            data_copy = copy.deepcopy(data)
            
            # Convert pandas dataframes and timestamps for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                if isinstance(obj, pd.Timestamp):
                    return str(obj)
                return obj
            
            data_json = convert_for_json(data_copy)
            
            with open(filename, 'w') as f:
                json.dump(data_json, f, indent=4, default=str)
            
            logger.info(f"Data exported to {filename}")
            return f"Data exported to {filename}"
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return None

    def get_option_chain_with_greeks(self, ticker_symbol, weeks=4):
        """
        Get option chain data for the specified ticker for the next several weeks
        with Black-Scholes greeks calculations.
        
        Args:
            ticker_symbol (str): The stock ticker symbol
            weeks (int): Number of weeks to look ahead
            
        Returns:
            dict: Option chain data by expiration date with calculated greeks
        """
        logger.info(f"Fetching {weeks} weeks of option chains with greeks for {ticker_symbol}")
        
        try:
            # Use yfinance Ticker directly
            from yfinance import Ticker
            ticker = Ticker(ticker_symbol)
            
            # Get current price for reference
            current_price = ticker.history(period='1d')['Close'].iloc[-1].item()
            logger.info(f"Current price of {ticker_symbol}: ${current_price:.2f}")
            
            # Get all available expiration dates
            try:
                expirations = ticker.options
                if not expirations:
                    logger.warning(f"No options expiration dates found for {ticker_symbol}")
                    return {}
                    
                logger.info(f"Found {len(expirations)} expiration dates for {ticker_symbol}")
            except Exception as e:
                logger.error(f"Error fetching expiration dates: {e}")
                return {}
            
            # Calculate target dates for the next specified number of weeks
            from datetime import datetime, timedelta
            today = datetime.now()
            target_dates = [(today + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, weeks+1)]
            
            # Find the closest expiration dates to our target dates
            selected_expirations = []
            for target in target_dates:
                closest = None
                min_diff = float('inf')
                target_date = datetime.strptime(target, '%Y-%m-%d')
                
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    diff = abs((exp_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest = exp
                
                if closest and closest not in selected_expirations:
                    selected_expirations.append(closest)
            
            # Fetch option chains for selected expiration dates
            option_chains = {}
            for exp_date in selected_expirations:
                logger.info(f"Fetching option chain for expiration date: {exp_date}")
                
                try:
                    opt = ticker.option_chain(exp_date)
                    
                    # Make sure we have both calls and puts
                    if not hasattr(opt, 'calls') or not hasattr(opt, 'puts'):
                        logger.warning(f"Incomplete option chain for {ticker_symbol} at {exp_date}")
                        continue
                        
                    logger.info(f"Retrieved {len(opt.calls)} calls and {len(opt.puts)} puts")
                    
                    # Calculate days to expiration for Black-Scholes
                    days_to_expiration = (datetime.strptime(exp_date, '%Y-%m-%d') - today).days
                    
                    # Calculate time to expiry in years for Black-Scholes
                    t = days_to_expiration / 365
                    
                    # Risk-free rate (assuming 5% in 2025)
                    r = 0.05
                    
                    # Extract and process call options
                    calls_df = opt.calls.copy()
                    
                    # Extract and process put options
                    puts_df = opt.puts.copy()
                    
                    # Add distance from current price (% OTM/ITM)
                    calls_df['distance_pct'] = ((calls_df['strike'] - current_price) / current_price) * 100
                    puts_df['distance_pct'] = ((puts_df['strike'] - current_price) / current_price) * 100
                    
                    # Calculate Black-Scholes greeks for calls
                    calls_df['delta'] = 0.0
                    calls_df['gamma'] = 0.0
                    calls_df['theta'] = 0.0
                    calls_df['vega'] = 0.0
                    
                    for idx, row in calls_df.iterrows():
                        # Use implied volatility from yfinance if available
                        sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']) else 0.3
                        
                        # Calculate greeks
                        calls_df.at[idx, 'delta'] = self._call_delta(current_price, row['strike'], t, r, sigma)
                        calls_df.at[idx, 'gamma'] = self._gamma(current_price, row['strike'], t, r, sigma)
                        calls_df.at[idx, 'theta'] = self._call_theta(current_price, row['strike'], t, r, sigma)
                        calls_df.at[idx, 'vega'] = self._vega(current_price, row['strike'], t, r, sigma)
                    
                    # Calculate Black-Scholes greeks for puts
                    puts_df['delta'] = 0.0
                    puts_df['gamma'] = 0.0
                    puts_df['theta'] = 0.0
                    puts_df['vega'] = 0.0
                    
                    for idx, row in puts_df.iterrows():
                        # Use implied volatility from yfinance if available
                        sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']) else 0.3
                        
                        # Calculate greeks
                        puts_df.at[idx, 'delta'] = self._put_delta(current_price, row['strike'], t, r, sigma)
                        puts_df.at[idx, 'gamma'] = self._gamma(current_price, row['strike'], t, r, sigma)
                        puts_df.at[idx, 'theta'] = self._put_theta(current_price, row['strike'], t, r, sigma)
                        puts_df.at[idx, 'vega'] = self._vega(current_price, row['strike'], t, r, sigma)
                    
                    # Calculate implied volatility skew metrics
                    atm_calls = calls_df[abs(calls_df['distance_pct']) < 5]
                    atm_puts = puts_df[abs(puts_df['distance_pct']) < 5]
                    
                    avg_call_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0
                    avg_put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0
                    
                    iv_skew = avg_put_iv - avg_call_iv
                    
                    # Store the option chain data
                    option_chains[exp_date] = {
                        'calls': calls_df,
                        'puts': puts_df,
                        'days_to_expiration': days_to_expiration,
                        'time_to_expiry_years': t,
                        'current_price': current_price,
                        'avg_call_iv': avg_call_iv,
                        'avg_put_iv': avg_put_iv,
                        'iv_skew': iv_skew
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing option chain for {exp_date}: {e}")
            
            logger.info(f"Successfully fetched {len(option_chains)} option chains for {ticker_symbol}")
            return option_chains
            
        except Exception as e:
            logger.error(f"Error in get_option_chain_with_greeks: {e}")
            return {}
    
    # Black-Scholes helper functions for Greeks calculation
    def _call_delta(self, S, K, t, r, sigma):
        """Calculate call option delta using Black-Scholes"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return 1.0 if S > K else 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        return norm.cdf(d1)
    
    def _put_delta(self, S, K, t, r, sigma):
        """Calculate put option delta using Black-Scholes"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return -1.0 if S < K else 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        return norm.cdf(d1) - 1
    
    def _gamma(self, S, K, t, r, sigma):
        """Calculate option gamma using Black-Scholes (same for calls and puts)"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        return norm.pdf(d1) / (S * sigma * math.sqrt(t))
    
    def _call_theta(self, S, K, t, r, sigma):
        """Calculate call option theta using Black-Scholes"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        
        theta = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(t)) - r * K * math.exp(-r * t) * norm.cdf(d2)
        return theta / 365  # Convert to daily theta
    
    def _put_theta(self, S, K, t, r, sigma):
        """Calculate put option theta using Black-Scholes"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        
        theta = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(t)) + r * K * math.exp(-r * t) * norm.cdf(-d2)
        return theta / 365  # Convert to daily theta
    
    def _vega(self, S, K, t, r, sigma):
        """Calculate option vega using Black-Scholes (same for calls and puts)"""
        from scipy.stats import norm
        import math
        
        if sigma <= 0 or t <= 0:
            return 0.0
            
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        return S * math.sqrt(t) * norm.pdf(d1) * 0.01  # Multiply by 0.01 to get vega per 1% change in IV


# Example usage function
def test_client():
    client = YFinanceClient()
    print("Fetching comprehensive analysis...")
    analysis = client.get_complete_analysis()
    
    # Print summary information
    if analysis and analysis.get("summary"):
        print("\n=== ANALYSIS SUMMARY ===")
        for key, value in analysis["summary"].items():
            print(f"{key}: {value}")
    
    # Print VIX information
    if analysis and analysis.get("volatility_filter"):
        print("\n=== VIX INFORMATION ===")
        print(f"VIX Price: {analysis['volatility_filter'].get('price')}")
        print(f"Market Stability: {analysis['volatility_filter'].get('stability')}")
        print(f"Risk Adjustment: {analysis['volatility_filter'].get('risk_adjustment')}")
    
    # Export data
    print("\nExporting data...")
    client.export_to_excel(analysis)
    client.export_to_json(analysis)
    print("Export complete!")


if __name__ == "__main__":
    test_client() 