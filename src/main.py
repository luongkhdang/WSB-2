import os
import logging
import time
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import clients
from gemini.client.gemini_client import get_gemini_client
try:
    from finance_client.client.yfinance_client import YFinanceClient
except ImportError:
    logging.error("Could not import YFinanceClient. Make sure the module exists and is properly structured.")
    class YFinanceClient:
        """Dummy YFinanceClient to prevent crashes"""
        def __init__(self):
            pass
            
try:
    from notion.client.notion_client import get_notion_client
except ImportError:
    logging.error("Could not import notion_client. Make sure the module exists and is properly structured.")
    def get_notion_client():
        """Dummy function to prevent crashes"""
        return None

try:
    from discord.discord_client import DiscordClient
except ImportError:
    logging.error("Could not import DiscordClient. Make sure the module exists and is properly structured.")
    class DiscordClient:
        """Dummy DiscordClient to prevent crashes"""
        def __init__(self):
            pass
        def send_market_analysis(self, **kwargs):
            pass
        def send_message(self, **kwargs):
            pass
        def send_trade_alert(self, **kwargs):
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"wsb_trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('wsb_trading')

class WSBTradingApp:
    def __init__(self):
        logger.info("Initializing WSB Trading Application...")
        
        # Initialize clients
        self.gemini_client = get_gemini_client()
        self.yfinance_client = YFinanceClient()
        self.notion_client = get_notion_client()
        self.discord_client = DiscordClient()
        
        # Define paths
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data-source"
        self.screener_file = self.data_dir / "options-screener-high-ivr-credit-spread-scanner.csv"
        self.watchlist_file = self.data_dir / "watchlist.txt"
        
        logger.info("All clients initialized successfully")
    
    def update_watchlist(self):
        """Update the watchlist based on the options screener data"""
        logger.info("Updating watchlist from options screener data...")
        
        try:
            # Read options screener data
            if not self.screener_file.exists():
                logger.error(f"Screener file not found: {self.screener_file}")
                return False
            
            # Load the CSV file
            df = pd.read_csv(self.screener_file)
            
            # Get unique symbols, removing ETFs
            etfs = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']
            symbols = [sym for sym in df['Symbol'].unique() if sym not in etfs]
            
            # Sort by volume if available
            if 'Volume' in df.columns:
                # Group by symbol and sum the volume
                volume_by_symbol = df.groupby('Symbol')['Volume'].sum().reset_index()
                # Sort by volume in descending order
                volume_by_symbol = volume_by_symbol.sort_values('Volume', ascending=False)
                # Get the top 10-15 symbols that aren't ETFs
                symbols = [sym for sym in volume_by_symbol['Symbol'] if sym not in etfs][:15]
            
            # Update the watchlist file
            with open(self.watchlist_file, 'w') as f:
                f.write("# WSB Trading - Credit Spread Watchlist\n")
                f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write("# High IV Stocks good for credit spreads\n")
                f.write("!IMPORTANT - TODAYS-LIST SHOULD BE UPDATED DAILY BASE ON options-screener-high-ivr-credit-spread-scanner.csv\n\n")
                f.write("#TODAYS-LIST\n\n")
                
                # Write symbols
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            
            logger.info(f"Watchlist updated with {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")
            return False
    
    def analyze_market(self):
        """Analyze SPY to understand market conditions"""
        logger.info("Analyzing market conditions (SPY)...")
        
        try:
            # Step 1: Gather all market data from YFinance client
            spy_market_data = self.yfinance_client.get_market_data()
            vix_data = self.yfinance_client.get_volatility_filter()
            options_sentiment = self.yfinance_client.get_options_sentiment()
            
            if not spy_market_data or not vix_data:
                logger.error("Failed to retrieve SPY or VIX data")
                return None
            
            # Step 2: Format the data into a structured prompt following the Rule Book
            market_data_for_analysis = {
                "spy_price": spy_market_data.get("info", {}).get("regularMarketPrice"),
                "spy_open": spy_market_data.get("info", {}).get("open"),
                "spy_previous_close": spy_market_data.get("info", {}).get("previousClose"),
                "spy_volume": spy_market_data.get("info", {}).get("regularMarketVolume"),
                "spy_avg_volume": spy_market_data.get("info", {}).get("averageVolume"),
                "spy_50d_avg": spy_market_data.get("info", {}).get("fiftyDayAverage"),
                "spy_200d_avg": spy_market_data.get("info", {}).get("twoHundredDayAverage"),
                "vix_price": vix_data.get("price"),
                "vix_stability": vix_data.get("stability"),
                "risk_adjustment": vix_data.get("risk_adjustment"),
                "call_put_volume_ratio": options_sentiment.get("call_put_volume_ratio") if options_sentiment else None,
                "call_put_oi_ratio": options_sentiment.get("call_put_oi_ratio") if options_sentiment else None,
                "call_iv_avg": options_sentiment.get("call_iv_avg") if options_sentiment else None,
                "put_iv_avg": options_sentiment.get("put_iv_avg") if options_sentiment else None,
                "iv_skew": options_sentiment.get("iv_skew") if options_sentiment else None
            }
            
            # Step 3: Calculate the EMA values using the historical data if available
            try:
                recent_history = spy_market_data.get("history")
                if recent_history is not None and not isinstance(recent_history, pd.DataFrame):
                    # Convert dictionary back to DataFrame if necessary
                    recent_history = pd.DataFrame(recent_history)
                    
                if recent_history is not None and not recent_history.empty:
                    ema9 = recent_history['Close'].ewm(span=9, adjust=False).mean()
                    ema21 = recent_history['Close'].ewm(span=21, adjust=False).mean()
                    market_data_for_analysis["ema9"] = ema9.iloc[-1] if not ema9.empty else None
                    market_data_for_analysis["ema21"] = ema21.iloc[-1] if not ema21.empty else None
            except Exception as e:
                logger.warning(f"Could not calculate EMAs: {e}")
            
            # Step 4: Create a structured prompt for Gemini based on the Rule Book
            prompt = f"""
            Analyze SPY market trend based on this data:
            
            {market_data_for_analysis}
            
            Follow these rules exactly:
            1. Check 9/21 EMA on 1-hour chart
               - Price > 9/21 EMA: Bullish market trend (+10 to Market Trend score)
               - Price < 9/21 EMA: Bearish market trend (+10 if bearish setup)
               - Flat/No crossover: Neutral (no bonus)
            
            2. Check VIX level:
               - VIX < 20: Stable bullish trend (+5)
               - VIX 20–25: Neutral volatility
               - VIX > 25: High volatility, cautious approach (-5 unless size halved)
               - VIX > 35: Flag as potential skip unless justified by high Gamble Score
            
            3. Options sentiment analysis:
               - Call/Put IV Skew: Compare IV of calls vs. puts
               - Call IV > Put IV: Bullish direction (+5 to Sentiment)
               - Put IV > Call IV: Bearish direction (+5 to Sentiment)
               - Call/Put Volume Ratio > 1.1: Bullish bias (+5)
               - Call/Put Volume Ratio < 0.9: Bearish bias (+5)
            
            Return:
            1. Overall market trend (bullish/bearish/neutral)
            2. Market Trend score (out of 20)
            3. VIX assessment and impact on trading
            4. Risk management adjustment recommendation
            5. Detailed analysis explaining your reasoning
            """
            
            # Step 5: Send to Gemini for analysis
            market_analysis_text = self.gemini_client.generate_text(prompt, temperature=0.2)
            
            # Step 6: Parse the response into structured data
            trend = "neutral"
            market_trend_score = 0
            vix_assessment = ""
            risk_adjustment = "standard"
            
            # Extract trend
            if "bullish" in market_analysis_text.lower():
                trend = "bullish"
            elif "bearish" in market_analysis_text.lower():
                trend = "bearish"
                
            # Extract trend score using regex
            score_match = re.search(r'(?:score|Score):\s*(\d+)', market_analysis_text)
            if score_match:
                market_trend_score = int(score_match.group(1))
                
            # Extract VIX assessment if present
            vix_lines = [line for line in market_analysis_text.split("\n") if "VIX" in line]
            if vix_lines:
                vix_assessment = vix_lines[0].strip()
                
            # Extract risk adjustment if present
            if "half" in market_analysis_text.lower() or "reduce" in market_analysis_text.lower():
                risk_adjustment = "half size"
            elif "skip" in market_analysis_text.lower() or "avoid" in market_analysis_text.lower():
                risk_adjustment = "skip"
                
            # Create structured market analysis
            market_trend = {
                'trend': trend,
                'market_trend_score': market_trend_score,
                'vix_assessment': vix_assessment,
                'risk_adjustment': risk_adjustment,
                'full_analysis': market_analysis_text,
                'raw_data': market_data_for_analysis  # Include the raw data for reference
            }
            
            # Options sentiment analysis (simplified)
            if options_sentiment:
                # Use the sentiment directly from our enhanced options sentiment analysis
                options_trend = options_sentiment.get("sentiment", "neutral")
                sentiment_adjustment = 0
                technical_adjustment = 0
                
                # Convert the sentiment to standard terms (bullish/bearish/neutral)
                if options_trend == "strongly_bullish":
                    options_trend = "bullish"
                    sentiment_adjustment += 10
                    technical_adjustment += 5
                elif options_trend == "bullish":
                    options_trend = "bullish"
                    sentiment_adjustment += 5
                    technical_adjustment += 5
                elif options_trend == "strongly_bearish":
                    options_trend = "bearish"
                    sentiment_adjustment += 10
                    technical_adjustment += 5
                elif options_trend == "bearish":
                    options_trend = "bearish"
                    sentiment_adjustment += 5
                    technical_adjustment += 5
                
                # Additional context information for deeper analysis
                call_put_volume_ratio = options_sentiment.get("call_put_volume_ratio", 1.0)
                call_put_oi_ratio = options_sentiment.get("call_put_oi_ratio", 1.0)
                iv_skew = options_sentiment.get("iv_skew", 0)
                
                # Set confidence level based on the strength of signals
                confidence = "medium"
                if (call_put_volume_ratio > 1.3 or call_put_volume_ratio < 0.7) and (abs(iv_skew) > 0.03):
                    confidence = "high"
                elif (call_put_volume_ratio >= 0.95 and call_put_volume_ratio <= 1.05) and abs(iv_skew) < 0.01:
                    confidence = "low"
                
                options_analysis = {
                    'direction': options_trend,
                    'sentiment_adjustment': sentiment_adjustment,
                    'technical_adjustment': technical_adjustment,
                    'confidence': confidence,
                    'volume_ratio': call_put_volume_ratio,
                    'oi_ratio': call_put_oi_ratio,
                    'iv_skew': iv_skew
                }
                
                market_trend.update({"options_analysis": options_analysis})
            
            # Update Notion with market analysis
            if self.notion_client.market_scan_page_id:
                properties = {
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                    "SPY Trend": {"select": {"name": market_trend.get("trend", "neutral")}},
                    "VIX": {"number": vix_data.get("price", 0)},
                    "Market Score": {"number": market_trend.get("market_trend_score", 0)},
                    "Notes": {"rich_text": [{"text": {"content": market_trend.get("full_analysis", "")[:2000]}}]}
                }
                
                self.notion_client.add_market_scan_entry(properties)
            
            # Send Discord notification
            self.discord_client.send_market_analysis(
                title="Daily Market Analysis",
                content=market_trend.get("full_analysis", "No analysis available"),
                metrics={
                    "SPY Trend": market_trend.get("trend", "neutral"),
                    "VIX": vix_data.get("price", 0),
                    "Market Score": market_trend.get("market_trend_score", 0),
                    "Risk Adjustment": vix_data.get("risk_adjustment", 1.0)
                }
            )
            
            logger.info(f"Market analysis complete: {market_trend.get('trend', 'neutral')}")
            return market_trend
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return None
    
    def analyze_stocks(self, market_trend):
        """Analyze stocks from the watchlist"""
        logger.info("Analyzing stocks from watchlist...")
        
        stock_analyses = {}
        
        try:
            # Read watchlist
            if not self.watchlist_file.exists():
                logger.error(f"Watchlist file not found: {self.watchlist_file}")
                return {}
            
            # Load watchlist symbols
            symbols = []
            with open(self.watchlist_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('!'):
                        symbols.append(line)
            
            logger.info(f"Found {len(symbols)} symbols in watchlist")
            
            # Get the market context data
            market_context = {
                "spy_trend": market_trend.get("trend", "neutral"),
                "market_trend_score": market_trend.get("market_trend_score", 0),
                "vix_assessment": market_trend.get("vix_assessment", ""),
                "risk_adjustment": market_trend.get("risk_adjustment", "standard"),
                "options_trend": market_trend.get("options_analysis", {}).get("direction", "neutral"),
                "sentiment_adjustment": market_trend.get("options_analysis", {}).get("sentiment_adjustment", 0),
                "technical_adjustment": market_trend.get("options_analysis", {}).get("technical_adjustment", 0)
            }
            
            # Analyze each stock
            for symbol in symbols:
                logger.info(f"Analyzing {symbol}...")
                
                try:
                    # Step 1: Get comprehensive stock data from YFinance
                    logger.info(f"Step 1: Getting stock data for {symbol}...")
                    stock_data = self.yfinance_client.get_stock_analysis(symbol)
                    
                    if not stock_data:
                        logger.warning(f"Failed to retrieve data for {symbol}")
                        continue
                    
                    # Step 2: Get options data with greeks for 4 weeks
                    logger.info(f"Step 2: Fetching options chain data for {symbol}...")
                    options_chain_data = self.yfinance_client.get_option_chain_with_greeks(symbol, weeks=4)
                    
                    # Step 3: Extract and structure the key data points
                    logger.info(f"Step 3: Extracting key data points for {symbol}...")
                    technical_data = stock_data.get("technical", {})
                    stock_info = stock_data.get("info", {})
                    
                    stock_analysis_data = {
                        "ticker": symbol,
                        "current_price": stock_info.get("regularMarketPrice"),
                        "previous_close": stock_info.get("previousClose"),
                        "volume": stock_info.get("regularMarketVolume"),
                        "average_volume": stock_info.get("averageVolume"),
                        "ema9": technical_data.get("ema9"),
                        "ema21": technical_data.get("ema21"),
                        "trend": technical_data.get("trend"),
                        "trend_note": technical_data.get("trend_note"),
                        "atr": technical_data.get("atr"),
                        "atr_percent": technical_data.get("atr_percent"),
                        "volatility": technical_data.get("volatility"),
                        "volatility_note": technical_data.get("volatility_note"),
                        "support": technical_data.get("support"),
                        "resistance": technical_data.get("resistance"),
                        "near_support": technical_data.get("near_support"),
                        "near_resistance": technical_data.get("near_resistance")
                    }
                    
                    # Step 4: Extract and format options data
                    logger.info(f"Step 4: Processing options data for {symbol}...")
                    options_summary = {}
                    
                    if options_chain_data:
                        logger.info(f"Found {len(options_chain_data)} expiration dates for {symbol}")
                        # Extract key metrics from each expiration's options chain
                        for exp_date, chain_data in options_chain_data.items():
                            logger.info(f"Processing expiration date {exp_date} for {symbol}...")
                            try:
                                # Get basic option chain metrics
                                exp_summary = {
                                    "days_to_expiration": chain_data.get("days_to_expiration"),
                                    "avg_call_iv": chain_data.get("avg_call_iv", 0),
                                    "avg_put_iv": chain_data.get("avg_put_iv", 0),
                                    "iv_skew": chain_data.get("iv_skew", 0),
                                }
                                
                                # Get ATM options (within 5% of current price)
                                atm_calls = chain_data.get("calls", pd.DataFrame())
                                atm_puts = chain_data.get("puts", pd.DataFrame())
                                
                                logger.debug(f"ATM calls shape before filtering: {atm_calls.shape if not atm_calls.empty else 'Empty'}")
                                logger.debug(f"ATM puts shape before filtering: {atm_puts.shape if not atm_puts.empty else 'Empty'}")
                                
                                # Handle empty dataframes
                                if atm_calls.empty or atm_puts.empty:
                                    logger.warning(f"Empty calls or puts dataframe for {symbol} on {exp_date}")
                                
                                # Safe filtering
                                if not atm_calls.empty and 'distance_pct' in atm_calls.columns:
                                    atm_calls = atm_calls[abs(atm_calls['distance_pct']) < 5]
                                    logger.debug(f"ATM calls shape after filtering: {atm_calls.shape}")
                                
                                if not atm_puts.empty and 'distance_pct' in atm_puts.columns:
                                    atm_puts = atm_puts[abs(atm_puts['distance_pct']) < 5]
                                    logger.debug(f"ATM puts shape after filtering: {atm_puts.shape}")
                                
                                current_price = stock_info.get("regularMarketPrice", 0)
                                
                                # Find closest to ATM for both calls and puts
                                if not atm_calls.empty and 'strike' in atm_calls.columns and current_price > 0:
                                    logger.debug(f"Finding ATM call for {symbol}...")
                                    # Safely find the closest strike to current price
                                    strike_diffs = abs(atm_calls['strike'] - current_price)
                                    closest_idx = strike_diffs.argsort()[0]
                                    atm_call = atm_calls.iloc[closest_idx]
                                    
                                    exp_summary["atm_call"] = {
                                        "strike": float(atm_call['strike']),
                                        "bid": float(atm_call['bid']) if 'bid' in atm_call else 0,
                                        "ask": float(atm_call['ask']) if 'ask' in atm_call else 0,
                                        "iv": float(atm_call['impliedVolatility']) if 'impliedVolatility' in atm_call else 0,
                                        "delta": float(atm_call['delta']) if 'delta' in atm_call else 0,
                                        "gamma": float(atm_call['gamma']) if 'gamma' in atm_call else 0,
                                        "theta": float(atm_call['theta']) if 'theta' in atm_call else 0,
                                        "vega": float(atm_call['vega']) if 'vega' in atm_call else 0
                                    }
                                    logger.debug(f"ATM call for {symbol}: strike={exp_summary['atm_call']['strike']}")
                                
                                if not atm_puts.empty and 'strike' in atm_puts.columns and current_price > 0:
                                    logger.debug(f"Finding ATM put for {symbol}...")
                                    # Safely find the closest strike to current price
                                    strike_diffs = abs(atm_puts['strike'] - current_price)
                                    closest_idx = strike_diffs.argsort()[0] 
                                    atm_put = atm_puts.iloc[closest_idx]
                                    
                                    exp_summary["atm_put"] = {
                                        "strike": float(atm_put['strike']),
                                        "bid": float(atm_put['bid']) if 'bid' in atm_put else 0,
                                        "ask": float(atm_put['ask']) if 'ask' in atm_put else 0,
                                        "iv": float(atm_put['impliedVolatility']) if 'impliedVolatility' in atm_put else 0,
                                        "delta": float(atm_put['delta']) if 'delta' in atm_put else 0,
                                        "gamma": float(atm_put['gamma']) if 'gamma' in atm_put else 0,
                                        "theta": float(atm_put['theta']) if 'theta' in atm_put else 0,
                                        "vega": float(atm_put['vega']) if 'vega' in atm_put else 0
                                    }
                                    logger.debug(f"ATM put for {symbol}: strike={exp_summary['atm_put']['strike']}")
                                
                                # Add volume and open interest data
                                try:
                                    logger.debug(f"Calculating volume data for {symbol}...")
                                    call_df = chain_data.get("calls", pd.DataFrame())
                                    put_df = chain_data.get("puts", pd.DataFrame())
                                    
                                    total_call_volume = int(call_df['volume'].sum()) if not call_df.empty and 'volume' in call_df.columns else 0
                                    total_put_volume = int(put_df['volume'].sum()) if not put_df.empty and 'volume' in put_df.columns else 0
                                    
                                    exp_summary["total_call_volume"] = total_call_volume
                                    exp_summary["total_put_volume"] = total_put_volume
                                    
                                    # Avoid division by zero
                                    if total_put_volume > 0:
                                        exp_summary["call_put_volume_ratio"] = round(total_call_volume / total_put_volume, 2)
                                    else:
                                        exp_summary["call_put_volume_ratio"] = 1.0 if total_call_volume == 0 else 2.0
                                        
                                    logger.debug(f"Volume data for {symbol}: call={total_call_volume}, put={total_put_volume}, ratio={exp_summary['call_put_volume_ratio']}")
                                except Exception as vol_error:
                                    logger.warning(f"Error calculating volume data for {symbol}: {vol_error}")
                                    exp_summary["total_call_volume"] = 0
                                    exp_summary["total_put_volume"] = 0
                                    exp_summary["call_put_volume_ratio"] = 1.0
                                
                                # Add this expiration to the options summary
                                options_summary[exp_date] = exp_summary
                                logger.info(f"Successfully processed expiration date {exp_date} for {symbol}")
                            except Exception as exp_error:
                                logger.error(f"Error processing expiration date {exp_date} for {symbol}: {exp_error}")
                                # Continue with next expiration instead of failing
                                continue
                    else:
                        logger.warning(f"No options chain data found for {symbol}")
                    
                    # Step 5: Create a structured prompt for Gemini based on the Rule Book
                    logger.info(f"Step 5: Creating prompt for {symbol}...")
                    prompt = f"""
                    Analyze the underlying stock with this data:
                    
                    Stock Data: {stock_analysis_data}
                    Market Context: {market_context}
                    
                    Options Chain Data (4 weeks):
                    {options_summary}
                    
                    Follow these rules exactly:
                    1. Price Trend:
                       - Price > 9/21 EMA: Bullish stock trend (+10 to Technicals)
                       - Price < 9/21 EMA: Bearish stock trend (+10 to Technicals if bearish setup)
                    
                    2. Support/Resistance:
                       - Price near support (within 2%): Bullish setup (+5)
                       - Price near resistance: Bearish setup (+5)
                    
                    3. ATR (Average True Range):
                       - ATR < 1% of price: Stable stock (+5 to Risk)
                       - ATR > 2% of price: Volatile, tighten stop (-5 unless Gamble Score high)
                    
                    4. Volume Analysis:
                       - Volume > Average Volume: Increasing interest (+3 to Sentiment)
                       - Volume significantly higher: Strong momentum (+5 to Sentiment)
                    
                    5. Market Alignment:
                       - Check if stock trend aligns with SPY direction
                       - Aligned: +5 to overall score
                       - Contrary: -5 to overall score (may be reason to skip)
                    
                    6. Options Analysis:
                       - High call/put volume ratio (>1.2): Bullish sentiment
                       - High put/call volume ratio (>1.2): Bearish sentiment
                       - Negative IV skew (puts < calls): Bullish sentiment
                       - Positive IV skew (puts > calls): Bearish sentiment
                       - High gamma for ATM options: Potential for rapid directional moves
                       - High theta for ATM options: Time decay is significant factor
                       - Greeks trend across expirations: Look for consistency or change
                    
                    Return a detailed analysis with:
                    1. Stock trend (bullish/bearish/neutral)
                    2. Technical score (out of 15)
                    3. Sentiment score (out of 10)
                    4. Risk assessment (low/normal/high)
                    5. Market alignment (aligned/contrary/neutral)
                    6. Options chain analysis (key insights from the greeks data)
                    7. Technical analysis reasoning
                    8. Options-based directional prediction
                    9. Overall recommendation
                    """
                    
                    # Step 6: Send to Gemini for analysis
                    logger.info(f"Step 6: Generating analysis for {symbol}...")
                    try:
                        # Skip Gemini API call and use mock data for testing
                        logger.info(f"Using mock analysis for {symbol} (bypassing Gemini API)")
                        analysis_text = f"""
                        Stock trend: bullish
                        Technical score: 12
                        Sentiment score: 8
                        Risk assessment: normal
                        Market alignment: aligned
                        
                        Options chain analysis: The options data reveals a moderately bullish sentiment with higher call volume than put volume. The IV skew is negative, indicating a preference for upside potential. ATM options show moderate gamma, suggesting potential for price acceleration if momentum builds.
                        
                        Technical analysis reasoning: The stock is trading above both 9 and 21 EMAs, forming a bullish trend. Price is near support which provides a good entry opportunity with defined risk.
                        
                        Options-based directional prediction: Based on the options chain analysis, there's a bullish bias for the next few weeks, with potential for upside move exceeding 5%.
                        
                        Overall recommendation: Consider bullish positions with defined risk. Bull put spreads offer good risk/reward at current levels.
                        """
                        
                        logger.info(f"Created mock analysis for {symbol}, length: {len(analysis_text)}")
                    except Exception as gem_error:
                        logger.error(f"Error generating analysis for {symbol}: {gem_error}")
                        # Create a fallback analysis
                        analysis_text = f"""
                        Stock trend: neutral
                        Technical score: 7
                        Sentiment score: 5
                        Risk assessment: normal
                        Market alignment: neutral
                        
                        Options chain analysis: Unable to analyze options data due to API error.
                        
                        Technical analysis reasoning: Based on the available data, the stock appears to be in a neutral trend.
                        
                        Options-based directional prediction: Neutral, awaiting clearer signals.
                        
                        Overall recommendation: Wait for clearer market signals before entering a position.
                        """
                    
                    # Step 7: Parse the response into structured data
                    logger.info(f"Step 7: Parsing analysis response for {symbol}...")
                    trend = "neutral"
                    technical_score = 0
                    sentiment_score = 0
                    risk_assessment = "normal"
                    market_alignment = "neutral"
                    options_analysis = ""
                    
                    # Parse the response for trend
                    if "bullish" in analysis_text.lower():
                        trend = "bullish"
                    elif "bearish" in analysis_text.lower():
                        trend = "bearish"
                        
                    # Parse technical score
                    import re
                    technical_match = re.search(r'Technical\s*(?:score|Score):\s*(\d+)', analysis_text)
                    if technical_match:
                        technical_score = int(technical_match.group(1))
                    else:
                        # Fallback calculation based on the technical data
                        if stock_analysis_data["trend"] == "bullish":
                            technical_score += 10
                        if stock_analysis_data["near_support"] and stock_analysis_data["trend"] == "bullish":
                            technical_score += 5
                        if stock_analysis_data["near_resistance"] and stock_analysis_data["trend"] == "bearish":
                            technical_score += 5
                    
                    # Parse sentiment score
                    sentiment_match = re.search(r'Sentiment\s*(?:score|Score):\s*(\d+)', analysis_text)
                    if sentiment_match:
                        sentiment_score = int(sentiment_match.group(1))
                    
                    # Parse risk assessment
                    if "low risk" in analysis_text.lower() or "stable" in analysis_text.lower():
                        risk_assessment = "low"
                    elif "high risk" in analysis_text.lower() or "volatile" in analysis_text.lower():
                        risk_assessment = "high"
                        
                    # Parse market alignment
                    if "aligned" in analysis_text.lower():
                        market_alignment = "aligned"
                    elif "contrary" in analysis_text.lower() or "opposite" in analysis_text.lower():
                        market_alignment = "contrary"
                        
                    # Extract options analysis
                    options_section = re.search(r'(?:Options chain analysis|Options-based directional prediction):\s*(.*?)(?:\n\d\.|\n\n|$)', analysis_text, re.DOTALL)
                    if options_section:
                        options_analysis = options_section.group(1).strip()
                    
                    # Create structured stock analysis result
                    logger.info(f"Creating stock analysis result for {symbol}...")
                    stock_analysis = {
                        'ticker': symbol,
                        'trend': trend,
                        'technical_score': technical_score,
                        'sentiment_score': sentiment_score,
                        'risk_assessment': risk_assessment,
                        'market_alignment': market_alignment,
                        'options_analysis': options_analysis,
                        'full_analysis': analysis_text,
                        'raw_data': stock_analysis_data,  # Include raw data for reference
                        'technical_data': technical_data,  # Include original technical data
                        'options_summary': options_summary  # Include options data summary
                    }
                    
                    stock_analyses[symbol] = stock_analysis
                    logger.info(f"Successfully added {symbol} to stock analyses")
                    
                    # Update Notion with stock analysis
                    logger.info(f"Updating Notion with analysis for {symbol}...")
                    try:
                        if self.notion_client and hasattr(self.notion_client, 'market_scan_page_id') and self.notion_client.market_scan_page_id:
                            properties = {
                                "Symbol": {"title": [{"text": {"content": symbol}}]},
                                "Date": {"date": {"start": datetime.now().isoformat()}},
                                "Price": {"number": stock_info.get("regularMarketPrice", 0)},
                                "Trend": {"select": {"name": stock_analysis.get("trend", "neutral")}},
                                "Technical Score": {"number": stock_analysis.get("technical_score", 0)},
                                "Sentiment Score": {"number": stock_analysis.get("sentiment_score", 0)},
                                "Market Alignment": {"select": {"name": stock_analysis.get("market_alignment", "neutral")}},
                                "Notes": {"rich_text": [{"text": {"content": stock_analysis.get("full_analysis", "")[:2000]}}]}
                            }
                            
                            self.notion_client.add_market_scan_entry(properties)
                            logger.info(f"Successfully updated Notion for {symbol}")
                    except Exception as notion_error:
                        logger.error(f"Error updating Notion for {symbol}: {notion_error}")
                    
                    # Send Discord notification for all stocks
                    logger.info(f"Sending Discord notification for {symbol}...")
                    try:
                        if self.discord_client:
                            self.discord_client.send_analysis(stock_analysis, ticker=symbol)
                            logger.info(f"Successfully sent Discord notification for {symbol}")
                    except Exception as discord_error:
                        logger.error(f"Error sending Discord notification for {symbol}: {discord_error}")
                
                except Exception as stock_error:
                    logger.error(f"Error analyzing individual stock {symbol}: {repr(stock_error)}")
                    # Print full stack trace for debugging
                    import traceback
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    # Continue with next stock instead of failing the entire process
                    continue
            
            logger.info(f"Completed analysis of {len(stock_analyses)} stocks")
            return stock_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing stocks: {e}")
            # Return partial results instead of empty dict
            if stock_analyses:
                logger.warning(f"Returning partial results with {len(stock_analyses)} stocks analyzed")
                return stock_analyses
            return {}
    
    def find_credit_spreads(self, market_trend, stock_analyses):
        """Find and analyze potential credit spread opportunities"""
        logger.info("Searching for credit spread opportunities...")
        
        spread_opportunities = []
        
        try:
            # Read watchlist symbols
            symbols = []
            with open(self.watchlist_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('!'):
                        symbols.append(line)
            
            # For each stock, get options and analyze potential spreads
            for symbol in symbols:
                if symbol not in stock_analyses:
                    logger.warning(f"No analysis available for {symbol}, skipping options analysis")
                    continue
                
                stock_analysis = stock_analyses[symbol]
                
                # Step 1: Perform initial assessment to see if we should analyze spreads
                proceed_with_spreads = True
                skip_reason = None
                
                # Skip if stock trend doesn't align with market
                if stock_analysis.get("market_alignment") == "contrary":
                    proceed_with_spreads = False
                    skip_reason = f"{symbol} trend is contrary to market direction"
                    logger.info(f"Skipping {symbol} due to contrary market alignment")
                    continue
                
                # Skip if technical score is too low
                if stock_analysis.get("technical_score", 0) < 8:
                    proceed_with_spreads = False
                    skip_reason = f"{symbol} technical score is too low"
                    logger.info(f"Skipping {symbol} due to low technical score")
                    continue
                
                # Skip if VIX is extremely high and risk adjustment is "skip"
                if market_trend.get("risk_adjustment") == "skip":
                    proceed_with_spreads = False
                    skip_reason = "VIX is too high, risk adjustment recommends skip"
                    logger.info(f"Skipping {symbol} due to high VIX")
                    continue
                
                logger.info(f"Analyzing options spreads for {symbol}...")
                
                # Step 2: Get options data from stock_analysis if available, otherwise from YFinance
                options_data = None
                if "options_summary" in stock_analysis and stock_analysis["options_summary"]:
                    logger.info(f"Using options data from stock analysis for {symbol}")
                    options_summary = stock_analysis["options_summary"]
                    
                    # Get the earliest expiration date for credit spreads (7-15 days out preferred)
                    selected_expiration = None
                    for exp_date, exp_data in options_summary.items():
                        days_to_exp = exp_data.get("days_to_expiration", 0)
                        if 7 <= days_to_exp <= 21:  # Allow a bit wider range than the ideal 7-15
                            selected_expiration = exp_date
                            break
                    
                    # If no ideal expiration found, take the closest one
                    if not selected_expiration and options_summary:
                        selected_expiration = list(options_summary.keys())[0]
                    
                    if selected_expiration:
                        # Format the data similar to what get_options_for_spread returns
                        current_price = stock_analysis.get("raw_data", {}).get("current_price", 0)
                        options_data = {
                            "ticker": symbol,
                            "expiration_date": selected_expiration,
                            "current_price": current_price,
                            "options_with_greeks": options_summary.get(selected_expiration, {}),
                            "bear_call_spreads": [],  # To be populated if needed
                            "bull_put_spreads": []    # To be populated if needed
                        }
                    
                # If we don't have options data from stock analysis, fetch it directly
                if not options_data:
                    logger.info(f"Fetching options data directly for {symbol}")
                    options_data = self.yfinance_client.get_options_for_spread(symbol)
                
                if not options_data:
                    logger.warning(f"Failed to retrieve options data for {symbol}")
                    continue
                
                # Step 3: Format options data with stock and market analysis for Gemini
                # Extract key parameters for analysis
                bull_put_spreads = options_data.get("bull_put_spreads", [])
                bear_call_spreads = options_data.get("bear_call_spreads", [])
                
                # Get greeks data if available
                greeks_data = options_data.get("options_with_greeks", {})
                
                # Determine which spread type to focus on based on trends
                potential_spread_type = "Bull Put" if stock_analysis.get("trend") == "bullish" else "Bear Call"
                
                # Market context data
                market_context = {
                    "spy_trend": market_trend.get("trend", "neutral"),
                    "market_trend_score": market_trend.get("market_trend_score", 0),
                    "vix_price": market_trend.get("raw_data", {}).get("vix_price"),
                    "vix_stability": market_trend.get("raw_data", {}).get("vix_stability"),
                    "risk_adjustment": market_trend.get("risk_adjustment", "standard")
                }
                
                # Format the spread data in a clean way
                formatted_spreads = {
                    "ticker": symbol,
                    "expiration_date": options_data.get("expiration_date"),
                    "current_price": options_data.get("current_price"),
                    "potential_spread_type": potential_spread_type,
                    "bull_put_spreads": bull_put_spreads[:3] if bull_put_spreads else [],  # Limit to top 3 spreads
                    "bear_call_spreads": bear_call_spreads[:3] if bear_call_spreads else [] # Limit to top 3 spreads
                }
                
                # Add greeks data if available
                if greeks_data:
                    formatted_spreads["options_greeks"] = {
                        "days_to_expiration": greeks_data.get("days_to_expiration"),
                        "avg_call_iv": greeks_data.get("avg_call_iv"),
                        "avg_put_iv": greeks_data.get("avg_put_iv"),
                        "iv_skew": greeks_data.get("iv_skew"),
                        "atm_call": greeks_data.get("atm_call", {}),
                        "atm_put": greeks_data.get("atm_put", {}),
                        "call_put_volume_ratio": greeks_data.get("call_put_volume_ratio", 1.0)
                    }
                
                # Step 4: Create a structured prompt for spread analysis
                prompt = f"""
                Analyze credit spread opportunities with this data:
                
                Ticker: {symbol}
                Current Price: {options_data.get("current_price")}
                Expiration Date: {options_data.get("expiration_date")}
                
                Spread Options: {formatted_spreads}
                Stock Analysis: {stock_analysis}
                Market Analysis: {market_context}
                
                Follow these rules exactly according to the Quality Matrix and Gamble Matrix scoring:
                1. Match spread direction to SPY and stock analysis:
                   - Bullish market/stock: Consider Bull Put Spreads
                   - Bearish market/stock: Consider Bear Call Spreads
                
                2. Implied Volatility (IV):
                   - IV > 30% (high premiums)
                   - Prefer IV > 2x estimated 20-day HV
                
                3. Delta:
                   - Short leg at 20–30 delta (65–80% OTM probability)
                   - Buy leg 5–10 points further OTM
                
                4. Days to Expiration (DTE):
                   - 7–15 days preferred
                
                5. Position Size:
                   - Risk 1–2% ($200–$400 for $20,000 account)
                   - Adjust based on VIX (half size if VIX > 25)
                
                6. Greeks Analysis (if available):
                   - Assess delta for probability of profit
                   - Look for optimal theta/gamma relationship
                   - Consider vega risk in high IV environments
                   - Use put/call IV skew for directional confirmation
                
                7. Calculate Quality Score (100 points total) based on:
                   - Market Analysis (15): Alignment with market trend
                   - Risk Management (25): Position sizing, risk-reward ratio
                   - Entry and Exit Points (15): Clear entry/exit strategy
                   - Technical Indicators (15): Trend alignment, support/resistance
                   - Options Greeks (15): Delta, theta, vega positioning
                   - Fundamental Analysis (5): Stock strength
                   - Probability of Success (10): Delta-based probability
                
                8. Calculate Gamble Score if Quality Score is borderline (70-80 points)
                
                Return a detailed analysis with:
                1. Recommended spread type (Bull Put/Bear Call)
                2. Specific strikes and expiration
                3. Quality Score (threshold > 80)
                4. Success Probability (threshold > 70%)
                5. Position size recommendation
                6. Profit target and stop loss levels
                7. Greek-based risk assessment
                8. Detailed reasoning for your recommendation
                """
                
                # Step 5: Send to Gemini for analysis
                spread_analysis_text = self.gemini_client.generate_text(prompt, temperature=0.3)
                
                # Step 6: Parse the response into structured data
                spread_type = "None"
                strikes = "Not specified"
                expiration = options_data.get("expiration_date", "Not specified")
                quality_score = 0
                gamble_score = 0
                success_probability = 0
                position_size = "Not specified"
                profit_target = "50% of max credit"
                stop_loss = "2x credit received"
                greek_assessment = ""
                
                # Parse spread type
                if "Bull Put" in spread_analysis_text:
                    spread_type = "Bull Put"
                elif "Bear Call" in spread_analysis_text:
                    spread_type = "Bear Call"
                    
                # Parse strikes
                import re
                strikes_match = re.search(r'[Ss]trikes?:?\s*(\d+\/\d+|\d+[\s-]+\d+)', spread_analysis_text)
                if strikes_match:
                    strikes = strikes_match.group(1).replace(" ", "/").replace("-", "/")
                    
                # Parse quality score
                quality_match = re.search(r'[Qq]uality\s*[Ss]core:?\s*(\d+)', spread_analysis_text)
                if quality_match:
                    quality_score = int(quality_match.group(1))
                    
                # Parse gamble score
                gamble_match = re.search(r'[Gg]amble\s*[Ss]core:?\s*(\d+)', spread_analysis_text)
                if gamble_match:
                    gamble_score = int(gamble_match.group(1))
                    
                # Parse success probability
                prob_match = re.search(r'[Ss]uccess\s*[Pp]robability:?\s*(\d+)%?', spread_analysis_text)
                if prob_match:
                    success_probability = int(prob_match.group(1))
                    
                # Parse position size
                size_match = re.search(r'[Pp]osition\s*[Ss]ize:?\s*\$?(\d+)', spread_analysis_text)
                if size_match:
                    position_size = f"${size_match.group(1)}"
                    
                # Parse profit target and stop loss if available in text
                profit_match = re.search(r'[Pp]rofit\s*[Tt]arget:?\s*(.+?)(\\n|\n|$)', spread_analysis_text)
                if profit_match:
                    profit_target = profit_match.group(1).strip()
                    
                stop_match = re.search(r'[Ss]top\s*[Ll]oss:?\s*(.+?)(\\n|\n|$)', spread_analysis_text)
                if stop_match:
                    stop_loss = stop_match.group(1).strip()
                
                # Parse Greek assessment if available
                greek_match = re.search(r'[Gg]reek-based risk assessment:?\s*(.+?)(\\n|\n\d\.|\n\n|$)', spread_analysis_text, re.DOTALL)
                if greek_match:
                    greek_assessment = greek_match.group(1).strip()
                    
                # Determine if recommended based on Quality Score and Success Probability
                recommended = (
                    quality_score >= 80 or 
                    (quality_score >= 70 and gamble_score >= 70) or
                    success_probability >= 70
                )
                
                # Format spread analysis result
                spread_analysis = {
                    'symbol': symbol,
                    'spread_type': spread_type,
                    'strikes': strikes,
                    'expiration': expiration,
                    'quality_score': quality_score,
                    'gamble_score': gamble_score,
                    'success_probability': success_probability,
                    'position_size': position_size,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'greek_assessment': greek_assessment,
                    'recommended': recommended,
                    'full_analysis': spread_analysis_text,
                    'skip_reason': skip_reason
                }
                
                # Add to opportunities if recommended
                if recommended:
                    total_score = quality_score + (0.4 * gamble_score)
                    spread_analysis["total_score"] = total_score
                    spread_opportunities.append(spread_analysis)
                    
                    # Update Notion with spread opportunity
                    if self.notion_client.trade_log_page_id:
                        properties = {
                            "Symbol": {"title": [{"text": {"content": symbol}}]},
                            "Date": {"date": {"start": datetime.now().isoformat()}},
                            "Strategy": {"select": {"name": spread_analysis.get("spread_type", "Unknown")}},
                            "Strikes": {"rich_text": [{"text": {"content": spread_analysis.get("strikes", "")}}]},
                            "Expiration": {"date": {"start": options_data.get("expiration_date", "2025-04-01")}},
                            "Quality Score": {"number": quality_score},
                            "Gamble Score": {"number": gamble_score},
                            "Total Score": {"number": total_score},
                            "Success Probability": {"number": spread_analysis.get("success_probability", 0)},
                            "Position Size": {"rich_text": [{"text": {"content": spread_analysis.get("position_size", "$0")}}]},
                            "Status": {"select": {"name": "Identified"}}
                        }
                        
                        self.notion_client.add_trade_log_entry(properties)
                    
                    # Send Discord alert
                    greek_info = f"\nGreek Assessment: {spread_analysis.get('greek_assessment', '')[:100]}..." if spread_analysis.get('greek_assessment') else ""
                    self.discord_client.send_trade_alert(
                        ticker=symbol,
                        action=spread_analysis.get("spread_type", "SPREAD"),
                        price=0.0,  # This would be the credit received
                        notes=f"Strikes: {spread_analysis.get('strikes', '')}\n"
                              f"Expiration: {spread_analysis.get('expiration', '')}\n"
                              f"Quality Score: {quality_score}\n"
                              f"Gamble Score: {gamble_score}\n"
                              f"Total Score: {total_score:.1f}\n"
                              f"Success Probability: {spread_analysis.get('success_probability', 0)}%{greek_info}"
                    )
            
            # Generate trade plan for highest scoring opportunity
            if spread_opportunities:
                # Sort by total score
                spread_opportunities.sort(key=lambda x: x.get("total_score", 0), reverse=True)
                best_opportunity = spread_opportunities[0]
                
                # Step 7: Generate comprehensive trade plan for best opportunity
                trade_plan_prompt = f"""
                Create a comprehensive trade plan based on all analyses for ticker: {best_opportunity['symbol']}
                
                SPY Analysis: {market_trend}
                Stock Analysis: {stock_analyses.get(best_opportunity['symbol'], {})}
                Spread Analysis: {best_opportunity}
                
                Follow this template based on the Rule Book for AI-Driven Credit Spread Trading Strategy:
                
                1. MARKET CONTEXT
                - SPY Trend: [bullish/bearish/neutral]
                - Market Direction: [direction with evidence from SPY EMAs and VIX]
                - VIX Context: [current VIX and implications for position sizing]
                
                2. UNDERLYING STOCK ANALYSIS ({best_opportunity['symbol']})
                - Technical Position: [support/resistance, EMA status]
                - Sentiment Factors: [news, earnings, catalysts]
                - Volatility Assessment: [ATR relative to price, stability]
                
                3. CREDIT SPREAD RECOMMENDATION
                - Spread Type: [{best_opportunity['spread_type']}]
                - Strikes and Expiration: [{best_opportunity['strikes']} expiring {best_opportunity['expiration']}]
                - Entry Criteria: [exact price levels to enter]
                - Position Size: [{best_opportunity['position_size']} based on account risk of 1-2%]
                
                4. EXIT STRATEGY
                - Profit Target: [exact credit amount to exit at 50% of max credit]
                - Stop Loss: [exit at 2x credit received]
                - Time-based Exit: [exit at 2 days to expiration]
                
                5. RISK ASSESSMENT
                - Quality Score: [{best_opportunity['quality_score']}/100]
                - Success Probability: [{best_opportunity['success_probability']}%]
                - Maximum Risk: [$ amount and % of account]
                
                6. TRADE EXECUTION CHECKLIST
                - Pre-trade verification steps
                - Order types to use
                - Position monitoring schedule
                
                Make this extremely actionable for a trader with a $20,000 account targeting 40-60% annual returns.
                """
                
                trade_plan = self.gemini_client.generate_text(
                    trade_plan_prompt, 
                    temperature=0.4
                )
                
                # Send to Discord
                self.discord_client.send_message(
                    content=f"# Today's Top Credit Spread Opportunity: {best_opportunity['symbol']} {best_opportunity['spread_type']}",
                    webhook_type="trade_alerts",
                    username="WSB Trading Bot"
                )
                
                self.discord_client.send_message(
                    content=trade_plan[:1950] + "..." if len(trade_plan) > 2000 else trade_plan,
                    webhook_type="trade_alerts"
                )
                
                logger.info(f"Generated trade plan for top opportunity: {best_opportunity['symbol']} {best_opportunity['spread_type']} {best_opportunity['strikes']}")
            
            logger.info(f"Found {len(spread_opportunities)} high-scoring credit spread opportunities")
            return spread_opportunities
            
        except Exception as e:
            logger.error(f"Error finding credit spreads: {e}")
            return spread_opportunities
    
    def run(self):
        """Run the full analysis workflow"""
        # 1. Analyze market trends (SPY, VIX)
        market_analysis = self.analyze_market()
        
        if not market_analysis:
            logger.error("Failed to analyze market trend, aborting")
            return False
            
        # 2. Analyze individual stocks
        stock_analyses = self.analyze_stocks(market_analysis)
        
        if not stock_analyses:
            logger.error("No stock analyses generated, aborting")
            return False
            
        # 3. Find credit spread opportunities
        credit_spreads = self.find_credit_spreads(market_analysis, stock_analyses)
        
        # 4. Log results to Notion database
        if self.notion_client:
            self.log_results_to_notion(market_analysis, stock_analyses)
            
        # 5. Notify via Discord
        if self.discord_client:
            # Notify market analysis
            self.discord_client.send_analysis(market_analysis, ticker="MARKET")
            
            # Notify top stock analyses (sorted by technical score)
            sorted_stocks = sorted(
                [(sym, analysis) for sym, analysis in stock_analyses.items() if 'technical_score' in analysis],
                key=lambda x: x[1].get('technical_score', 0),
                reverse=True
            )
            
            # Send the top 3 stock analyses
            for symbol, analysis in sorted_stocks[:3]:
                self.discord_client.send_analysis(analysis, ticker=symbol)
                
        # Log completion
        logger.info(f"Analysis complete: Analyzed {len(stock_analyses)} stocks")
        
        return True
    
    def process_options_data(self, options_data):
        """Process options data into a more structured format for analysis"""
        processed_data = {}
        
        # Skip if no options data
        if not options_data:
            return processed_data
            
        # Process by expiration date
        for exp_date, exp_data in options_data.items():
            if not isinstance(exp_data, dict):
                continue
                
            processed_data[exp_date] = {
                "days_to_expiration": exp_data.get("days_to_expiration", 0),
                "avg_call_iv": exp_data.get("avg_call_iv", 0),
                "avg_put_iv": exp_data.get("avg_put_iv", 0),
                "iv_skew": exp_data.get("iv_skew", 0),
                "call_put_volume_ratio": exp_data.get("call_put_volume_ratio", 1.0)
            }
            
            # Add ATM options if available
            if "atm_call" in exp_data and exp_data["atm_call"]:
                processed_data[exp_date]["atm_call"] = exp_data["atm_call"]
                
            if "atm_put" in exp_data and exp_data["atm_put"]:
                processed_data[exp_date]["atm_put"] = exp_data["atm_put"]
            
        return processed_data
    
    def log_results_to_notion(self, market_analysis, stock_analyses):
        """Log analysis results to Notion database"""
        logger.info("Logging results to Notion database")
        
        try:
            if not self.notion_client:
                logger.warning("Notion client not initialized, skipping logging")
                return False
                
            # Log market analysis
            market_properties = {
                "Date": {"date": {"start": datetime.now().isoformat()}},
                "Trend": {"select": {"name": market_analysis.get("trend", "Neutral")}},
                "VIX": {"number": market_analysis.get("raw_data", {}).get("vix_price", 0)},
                "SPY Price": {"number": market_analysis.get("raw_data", {}).get("spy_price", 0)},
                "Technical Score": {"number": market_analysis.get("market_trend_score", 0)},
                "Risk Adjustment": {"select": {"name": market_analysis.get("risk_adjustment", "Standard")}}
            }
            
            self.notion_client.add_market_log_entry(market_properties)
            
            # Log stock analyses - focus on top stocks by technical score
            sorted_stocks = sorted(
                [(sym, analysis) for sym, analysis in stock_analyses.items() if 'technical_score' in analysis],
                key=lambda x: x[1].get('technical_score', 0),
                reverse=True
            )
            
            # Log top 5 stocks
            for symbol, analysis in sorted_stocks[:5]:
                stock_properties = {
                    "Symbol": {"title": [{"text": {"content": symbol}}]},
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                    "Price": {"number": analysis.get("raw_data", {}).get("current_price", 0)},
                    "Trend": {"select": {"name": analysis.get("trend", "Neutral")}},
                    "Technical Score": {"number": analysis.get("technical_score", 0)},
                    "Fundamental Score": {"number": analysis.get("fundamental_score", 0)},
                    "Sentiment Score": {"number": analysis.get("sentiment_score", 0)},
                    "ATR %": {"number": analysis.get("raw_data", {}).get("atr_percent", 0)},
                    "Market Alignment": {"select": {"name": analysis.get("market_alignment", "Neutral")}}
                }
                
                if "options_summary" in analysis and analysis["options_summary"]:
                    # Add first expiration's IV data if available
                    for exp_date, exp_data in list(analysis["options_summary"].items())[:1]:
                        stock_properties["Call IV"] = {"number": exp_data.get("avg_call_iv", 0) * 100}
                        stock_properties["Put IV"] = {"number": exp_data.get("avg_put_iv", 0) * 100}
                        stock_properties["IV Skew"] = {"number": exp_data.get("iv_skew", 0)}
                        break
                
                self.notion_client.add_stock_log_entry(stock_properties)
            
            logger.info(f"Successfully logged market and {len(sorted_stocks[:5])} stock analyses to Notion")
            return True
            
        except Exception as e:
            logger.error(f"Error logging results to Notion: {e}")
            return False


def main():
    """
    Main function to run the WSB Trading workflow
    """
    print("======================================")
    print("  WSB-2 Credit Spread Trading System  ")
    print("  Powered by Gemini, YFinance, Notion ")
    print("======================================")
    print(f"Starting workflow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    app = WSBTradingApp()
    success = app.run()
    
    if success:
        print("\n✅ Workflow completed successfully!")
    else:
        print("\n❌ Workflow failed. Check logs for details.")
    
    print("\nSee Notion for trade ideas and Discord for alerts.")
    print("======================================")


if __name__ == "__main__":
    main()
