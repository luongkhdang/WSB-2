import os
import logging
import time
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import inspect

# Set up logging with normal level 
logging.basicConfig(
    level=logging.INFO,  # Changed back from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wsb_trading.log')  # Removed debug-specific log file
    ]
)

logger = logging.getLogger(__name__)

# Import clients
from gemini.client.gemini_client import get_gemini_client
from finance_client.client.yfinance_client import YFinanceClient
from notion.client.notion_client import get_notion_client
from discord.discord_client import DiscordClient

# Import prompt hooks
from gemini.hooks import (
    get_market_trend_prompt,
    get_spy_options_prompt,
    get_market_data_prompt,
    get_stock_analysis_prompt,
    get_stock_options_prompt,
    get_trade_plan_prompt
)

class WSBTradingApp:
    def __init__(self):
        logger.info("Initializing WSB Trading Application...")
        
        # Define key market indices to always include in analysis
        self.key_indices = ['SPY', 'QQQ', 'IWM', 'VTV', 'VGLT', 'VIX', 'DIA', 'BND', 'BTC-USD']
        
        # List of ETFs (kept for reference but not used for exclusion)
        self.etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTV', 'VGLT', 'GLD', 'SLV', 'USO', 'XLF', 'XLE', 'XLI', 'XLK', 'XLV', 'XLP', 'XLU', 'XLB', 'XLY', 'XLC', 'XLRE', 'BND']
        
        # Constant Big Players to include in every watchlist
        self.constant_players = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSTR', 'MSFT', 'META', 'TSM', 'LLY', 'MU', 'PLTR', 'HOOD']
        
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
    
    def get_watchlist_symbols(self):
        """Get list of symbols from watchlist file."""
        try:
            if not self.watchlist_file.exists():
                logger.error(f"Watchlist file not found: {self.watchlist_file}")
                return []
            
            symbols = []
            with open(self.watchlist_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('!'):
                        symbols.append(line)
            
            return symbols
        except Exception as e:
            logger.error(f"Error reading watchlist: {e}")
            return []
    
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
             
            # Include all unique symbols from the screener (no longer filtering out ETFs)
            symbols = list(df['Symbol'].unique())
            
            # Sort by volume if available
            if 'Volume' in df.columns:
                # Group by symbol and sum the volume
                volume_by_symbol = df.groupby('Symbol')['Volume'].sum().reset_index()
                # Sort by volume in descending order
                volume_by_symbol = volume_by_symbol.sort_values('Volume', ascending=False)
                # Get the top 15 symbols
                symbols = list(volume_by_symbol['Symbol'])[:15]
            
            # Ensure key indices are included
            for index in self.key_indices:
                if index not in symbols:
                    symbols.append(index)
                    
            # Add constant big players to the watchlist
            for player in self.constant_players:
                if player not in symbols:
                    symbols.append(player)
            
            # Update the watchlist file
            with open(self.watchlist_file, 'w') as f:
                f.write("# WSB Trading - Credit Spread Watchlist\n")
                f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write("# Stocks and key indices for analysis\n")
                f.write("!IMPORTANT - TODAYS-LIST SHOULD BE UPDATED DAILY BASED ON options-screener-high-ivr-credit-spread-scanner.csv\n\n")
                f.write("# KEY INDICES (Always included)\n")
                
                # Write key indices first
                for index in self.key_indices:
                    f.write(f"{index}\n")
                
                f.write("\n# CONSTANT BIG PLAYERS (Always included)\n")
                
                # Write constant big players
                for player in self.constant_players:
                    if player not in self.key_indices:  # Avoid duplicates
                        f.write(f"{player}\n")
                
                f.write("\n# STOCKS FOR CREDIT SPREADS\n\n")
                
                # Write other symbols
                for symbol in symbols:
                    if symbol not in self.key_indices and symbol not in self.constant_players:  # Avoid duplicates
                        f.write(f"{symbol}\n")
            
            logger.info(f"Watchlist updated with {len(symbols)} symbols including key indices and constant big players")
            return True
            
        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")
            return False
    
    def analyze_market(self):
        """Analyze market conditions using multiple indices"""
        logger.info("Analyzing comprehensive market conditions...")
        
        try:
            # Step 1: Gather all market data from YFinance client
            market_indices_data = self.yfinance_client.get_market_data()
            vix_data = self.yfinance_client.get_volatility_filter()
            options_sentiment = self.yfinance_client.get_options_sentiment()
            
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
            daily_changes = {k: v for k, v in daily_changes.items() if v is not None}
            
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
            prompt = get_market_trend_prompt(market_data_for_analysis)
            
            # Step 5: Send to Gemini for analysis
            market_analysis_text = self.gemini_client.generate_text(prompt, temperature=0.2)
            
            # Step 6: Parse the response into structured data
            trend = "neutral"
            market_trend_score = 0
            vix_assessment = ""
            risk_adjustment = "standard"
            sector_rotation_analysis = ""
            
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
                
            # Extract sector rotation analysis if present
            sector_rotation_match = re.search(r'Sector Rotation:[\s\n]*(.*?)(?:[\n][\n]|$)', market_analysis_text, re.DOTALL)
            if sector_rotation_match:
                sector_rotation_analysis = sector_rotation_match.group(1).strip()
            
            # Create structured market analysis
            market_trend = {
                'trend': trend,
                'market_trend_score': market_trend_score,
                'vix_assessment': vix_assessment,
                'risk_adjustment': risk_adjustment,
                'sector_rotation': sector_rotation_analysis if sector_rotation_analysis else market_data_for_analysis.get("sector_rotation", ""),
                'full_analysis': market_analysis_text,
                'raw_data': market_data_for_analysis  # Include the raw data for reference
            }
            
            # Options sentiment analysis (simplified)
            if options_sentiment:
                # Get the options sentiment prompt from hooks
                options_prompt = get_spy_options_prompt(options_sentiment)
                options_analysis_text = self.gemini_client.generate_text(options_prompt, temperature=0.2)
                
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
                title="Comprehensive Market Analysis",
                content=market_trend.get("full_analysis", "No analysis available"),
                metrics={
                    "Market Trend": market_trend.get("trend", "neutral"),
                    "SPY": market_data_for_analysis.get("spy_price", 0),
                    "QQQ": market_data_for_analysis.get("qqq_price", 0),
                    "IWM": market_data_for_analysis.get("iwm_price", 0),
                    "VTV": market_data_for_analysis.get("vtv_price", 0),
                    "VGLT": market_data_for_analysis.get("vglt_price", 0),
                    "DIA": market_data_for_analysis.get("dia_price", 0),
                    "BND": market_data_for_analysis.get("bnd_price", 0),
                    "BTC-USD": market_data_for_analysis.get("btc_price", 0),
                    "VIX": market_data_for_analysis.get("vix_price", 0),
                    "Market Score": market_trend.get("market_trend_score", 0),
                    "Risk Adjustment": market_trend.get("risk_adjustment", "standard")
                }
            )
            
            logger.info(f"Comprehensive market analysis complete: {market_trend.get('trend', 'neutral')}")
            return market_trend
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return None
    
    def analyze_stocks(self, market_analysis):
        """Analyze individual stocks in the watchlist."""
        logger.info("Starting stock analysis...")
        
        # Check if get_stock_analysis_prompt is defined with correct signature
        try:
            sig = inspect.signature(get_stock_analysis_prompt)
            params = list(sig.parameters.keys())
            logger.info(f"get_stock_analysis_prompt signature: {sig} with params: {params}")
            if len(params) != 2:
                logger.warning(f"WARNING: get_stock_analysis_prompt has {len(params)} parameters instead of 2!")
                logger.warning(f"Parameters are: {params}")
        except Exception as e:
            logger.error(f"Error checking function signature: {e}")
        
        stock_analyses = {}
        
        # Get watchlist symbols
        symbols = self.get_watchlist_symbols()
        if not symbols:
            logger.error("No symbols in watchlist")
            return {}
            
        logger.info(f"Found {len(symbols)} symbols in watchlist")
        
        # Analyze each stock
        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")
            
            try:
                # Step 1: Get comprehensive stock data from YFinance
                logger.info(f"Step 1: Getting stock data for {symbol}...")
                stock_data = self.yfinance_client.get_stock_analysis(symbol)
                
                if not stock_data:
                    logger.error(f"Failed to get stock data for {symbol}")
                    continue
                    
                stock_info = stock_data.get("info", {})
                technical_data = stock_data.get("technical", {})
                
                # Step 2: Get options chain data with greeks
                logger.info(f"Step 2: Getting options chain data for {symbol}...")
                options_chain_data = self.yfinance_client.get_option_chain_with_greeks(symbol, weeks=4)
                
                if not options_chain_data:
                    logger.warning(f"No options chain data found for {symbol}")
                
                # Step 3: Prepare stock analysis data
                logger.info(f"Step 3: Preparing analysis data for {symbol}...")
                stock_analysis_data = {
                    "ticker": symbol,
                    "current_price": stock_info.get("regularMarketPrice"),
                    "previous_close": stock_info.get("previousClose"),
                    "volume": stock_info.get("regularMarketVolume"),
                    "average_volume": stock_info.get("averageVolume"),
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
                            # Validate chain_data structure
                            if not isinstance(chain_data, dict):
                                logger.warning(f"Invalid chain data format for {symbol} on {exp_date} - not a dictionary")
                                continue
                                
                            # Get basic option chain metrics
                            exp_summary = {
                                "days_to_expiration": chain_data.get("days_to_expiration"),
                                "avg_call_iv": chain_data.get("avg_call_iv", 0),
                                "avg_put_iv": chain_data.get("avg_put_iv", 0),
                                "iv_skew": chain_data.get("iv_skew", 0),
                            }
                            
                            # Get ATM options (within 5% of current price)
                            try:
                                atm_calls = chain_data.get("calls", pd.DataFrame())
                                atm_puts = chain_data.get("puts", pd.DataFrame())
                                
                                logger.debug(f"ATM calls shape before filtering: {atm_calls.shape if not atm_calls.empty else 'Empty'}")
                                logger.debug(f"ATM puts shape before filtering: {atm_puts.shape if not atm_puts.empty else 'Empty'}")
                                
                                # Handle empty dataframes
                                if atm_calls.empty or atm_puts.empty:
                                    logger.warning(f"Empty calls or puts dataframe for {symbol} on {exp_date}")
                                    continue
                                
                                # Verify required columns exist
                                required_columns = ['strike', 'bid', 'ask', 'impliedVolatility', 'delta', 'gamma', 'theta', 'vega', 'distance_pct']
                                missing_call_columns = [col for col in required_columns if col not in atm_calls.columns]
                                missing_put_columns = [col for col in required_columns if col not in atm_puts.columns]
                                
                                if missing_call_columns:
                                    logger.warning(f"Missing columns in calls dataframe for {symbol}: {missing_call_columns}")
                                if missing_put_columns:
                                    logger.warning(f"Missing columns in puts dataframe for {symbol}: {missing_put_columns}")
                                
                                if missing_call_columns and missing_put_columns:
                                    logger.warning(f"Missing required columns for both calls and puts, skipping {exp_date}")
                                    continue
                                
                                # Safe filtering
                                if not atm_calls.empty and 'distance_pct' in atm_calls.columns:
                                    atm_calls = atm_calls[abs(atm_calls['distance_pct']) < 5]
                                    logger.debug(f"ATM calls shape after filtering: {atm_calls.shape}")
                                
                                if not atm_puts.empty and 'distance_pct' in atm_puts.columns:
                                    atm_puts = atm_puts[abs(atm_puts['distance_pct']) < 5]
                                    logger.debug(f"ATM puts shape after filtering: {atm_puts.shape}")
                                
                                current_price = stock_info.get("regularMarketPrice", 0)
                                if current_price <= 0:
                                    logger.warning(f"Invalid current price {current_price} for {symbol}")
                                    current_price = chain_data.get("current_price", 0)
                                    if current_price <= 0:
                                        logger.warning(f"Still invalid current price, skipping {exp_date}")
                                        continue
                                
                                # Skip if no valid options found after filtering
                                if atm_calls.empty and atm_puts.empty:
                                    logger.warning(f"No valid ATM options found for {symbol} on {exp_date}")
                                    continue
                            except Exception as df_error:
                                logger.error(f"Error processing dataframes for {symbol} on {exp_date}: {repr(df_error)}")
                                continue
                            
                            # Process ATM calls
                            try:
                                if not atm_calls.empty and 'strike' in atm_calls.columns and current_price > 0:
                                    logger.debug(f"Finding ATM call for {symbol}...")
                                    # Safely find the closest strike to current price
                                    strike_diffs = abs(atm_calls['strike'] - current_price)
                                    if not strike_diffs.empty:
                                        # Sort by difference and get the first row
                                        closest_idx = strike_diffs.nsmallest(1).index[0]
                                        atm_call = atm_calls.loc[closest_idx]
                                        
                                        exp_summary["atm_call"] = {
                                            "strike": float(atm_call.get('strike', 0)),
                                            "bid": float(atm_call.get('bid', 0)),
                                            "ask": float(atm_call.get('ask', 0)),
                                            "iv": float(atm_call.get('impliedVolatility', 0)),
                                            "delta": float(atm_call.get('delta', 0)),
                                            "gamma": float(atm_call.get('gamma', 0)),
                                            "theta": float(atm_call.get('theta', 0)),
                                            "vega": float(atm_call.get('vega', 0))
                                        }
                                        logger.debug(f"ATM call for {symbol}: strike={exp_summary['atm_call']['strike']}")
                                    else:
                                        logger.warning(f"No valid strikes found for ATM calls")
                            except Exception as call_error:
                                logger.error(f"Error processing ATM call for {symbol} on {exp_date}: {repr(call_error)}")
                                logger.debug(f"ATM calls shape: {atm_calls.shape}, columns: {list(atm_calls.columns)}")
                            
                            # Process ATM puts
                            try:
                                if not atm_puts.empty and 'strike' in atm_puts.columns and current_price > 0:
                                    logger.debug(f"Finding ATM put for {symbol}...")
                                    # Safely find the closest strike to current price
                                    strike_diffs = abs(atm_puts['strike'] - current_price)
                                    if not strike_diffs.empty:
                                        # Sort by difference and get the first row
                                        closest_idx = strike_diffs.nsmallest(1).index[0]
                                        atm_put = atm_puts.loc[closest_idx]
                                        
                                        exp_summary["atm_put"] = {
                                            "strike": float(atm_put.get('strike', 0)),
                                            "bid": float(atm_put.get('bid', 0)),
                                            "ask": float(atm_put.get('ask', 0)),
                                            "iv": float(atm_put.get('impliedVolatility', 0)),
                                            "delta": float(atm_put.get('delta', 0)),
                                            "gamma": float(atm_put.get('gamma', 0)),
                                            "theta": float(atm_put.get('theta', 0)),
                                            "vega": float(atm_put.get('vega', 0))
                                        }
                                        logger.debug(f"ATM put for {symbol}: strike={exp_summary['atm_put']['strike']}")
                                    else:
                                        logger.warning(f"No valid strikes found for ATM puts")
                            except Exception as put_error:
                                logger.error(f"Error processing ATM put for {symbol} on {exp_date}: {repr(put_error)}")
                                logger.debug(f"ATM puts shape: {atm_puts.shape}, columns: {list(atm_puts.columns)}")
                            
                            # Add volume and open interest data
                            try:
                                logger.debug(f"Calculating volume data for {symbol}...")
                                call_df = chain_data.get("calls", pd.DataFrame())
                                put_df = chain_data.get("puts", pd.DataFrame())
                                
                                if 'volume' not in call_df.columns and 'volume' not in put_df.columns:
                                    logger.warning(f"Volume data not available for {symbol} on {exp_date}")
                                    exp_summary["total_call_volume"] = 0
                                    exp_summary["total_put_volume"] = 0
                                    exp_summary["call_put_volume_ratio"] = 1.0
                                else:
                                    total_call_volume = int(call_df['volume'].sum()) if not call_df.empty and 'volume' in call_df.columns else 0
                                    total_put_volume = int(put_df['volume'].sum()) if not put_df.empty and 'volume' in put_df.columns else 0
                                    
                                    exp_summary["total_call_volume"] = total_call_volume
                                    exp_summary["total_put_volume"] = total_put_volume
                                    
                                    # Calculate volume ratios
                                    if total_put_volume > 0:
                                        exp_summary["call_put_volume_ratio"] = total_call_volume / total_put_volume
                                    else:
                                        exp_summary["call_put_volume_ratio"] = 1.5  # Default bullish if no put volume
                                    
                                    logger.debug(f"Volume data for {symbol}: calls={total_call_volume}, puts={total_put_volume}")
                            except Exception as volume_error:
                                logger.error(f"Error calculating volume data for {symbol}: {repr(volume_error)}")
                                # Set default values if volume calculation fails
                                exp_summary["total_call_volume"] = 0
                                exp_summary["total_put_volume"] = 0
                                exp_summary["call_put_volume_ratio"] = 1.0
                            
                            # Only add the expiration summary if we have valid data
                            if "atm_call" in exp_summary or "atm_put" in exp_summary:
                                options_summary[exp_date] = exp_summary
                                logger.info(f"Successfully processed expiration {exp_date} for {symbol}")
                            else:
                                logger.warning(f"No valid options data found for {symbol} on {exp_date}")
                            
                        except Exception as exp_error:
                            logger.error(f"Error processing expiration {exp_date} for {symbol}: {repr(exp_error)}")
                            import traceback
                            logger.debug(f"Traceback: {traceback.format_exc()}")
                            continue
                
                # Step 5: Generate analysis using prompt hook
                logger.info(f"Step 5: Generating analysis for {symbol}...")
                try:
                    # Prepare market context data for the stock analysis prompt
                    market_context = {
                        "spy_trend": market_analysis.get("trend", "neutral"),
                        "market_trend_score": market_analysis.get("market_trend_score", 0),
                        "vix_price": market_analysis.get("raw_data", {}).get("vix_price"),
                        "vix_stability": market_analysis.get("raw_data", {}).get("vix_stability"),
                        "risk_adjustment": market_analysis.get("risk_adjustment", "standard"),
                        "sector_rotation": market_analysis.get("sector_rotation", ""),
                        "overall_market": "The market trend is " + market_analysis.get("trend", "neutral")
                    }
                    
                    # DEBUG: Log the arguments being passed to get_stock_analysis_prompt
                    logger.debug(f"Calling get_stock_analysis_prompt for {symbol} with args: stock_analysis_data, market_context")
                    
                    # Get the stock analysis prompt from hooks
                    # IMPORTANT: Always use exactly 2 arguments for this function
                    prompt = get_stock_analysis_prompt(stock_analysis_data, market_context)
                    analysis_text = self.gemini_client.generate_text(prompt)
                    
                    # Parse the response
                    trend = "neutral"
                    technical_score = 0
                    sentiment_score = 0
                    risk_assessment = "normal"
                    market_alignment = "neutral"
                    options_analysis = ""
                    
                    # Parse trend
                    if "bullish" in analysis_text.lower():
                        trend = "bullish"
                    elif "bearish" in analysis_text.lower():
                        trend = "bearish"
                    
                    # Parse technical score
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
                    market_alignment_match = re.search(r'Market Alignment:\s*(.*?)(?:\n\d\.|\n\n|$)', analysis_text, re.DOTALL)
                    if market_alignment_match:
                        market_alignment = market_alignment_match.group(1).strip().lower()
                        logger.info(f"Market alignment for {symbol}: {market_alignment}")
                    else:
                        logger.warning(f"No explicit market alignment found for {symbol}, falling back to trend comparison")
                        if market_analysis.get("trend", "neutral") == trend:
                            market_alignment = "aligned"
                        else:
                            market_alignment = "contrary"
                        logger.info(f"Market alignment for {symbol} (fallback): {market_alignment}")
                    
                    # Extract options analysis section if present
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
                        'raw_data': stock_analysis_data,
                        'technical_data': technical_data,
                        'options_summary': options_summary
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
                        self.discord_client.send_analysis(stock_analysis, symbol)
                        logger.info(f"Successfully sent Discord notification for {symbol}")
                    except Exception as discord_error:
                        logger.error(f"Error sending Discord notification for {symbol}: {discord_error}")
                
                except Exception as stock_error:
                    logger.error(f"Error analyzing individual stock {symbol}: {repr(stock_error)}")
                    # Print full stack trace for debugging
                    import traceback
                    logger.error(f"Stack trace: {traceback.format_exc()}")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"Completed analysis of {len(stock_analyses)} stocks")
        return stock_analyses
    
    def find_credit_spreads(self, market_trend, stock_analyses):
        """Find and analyze potential credit spread opportunities"""
        logger.info("Searching for credit spread opportunities...")
        
        spread_opportunities = []
        
        # We'll use this to track API calls and add delays
        api_call_count = 0
        
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
                
                # Apply rate limiting to prevent API overload
                api_call_count += 1
                if api_call_count % 5 == 0:  # Add delay every 5 API calls
                    logger.info(f"Added delay to respect API rate limits...")
                    time.sleep(30)  # 30 second delay
                
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
                
                # Step 4: Get the stock options prompt from hooks
                prompt = get_stock_options_prompt(formatted_spreads, stock_analysis, market_context)
                
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
                
                # Report the top 5 highest scored credit spread positions
                top_opportunities = spread_opportunities[:5]
                
                # Send a summary of top 5 opportunities to Discord
                summary_content = "# Today's Top Credit Spread Opportunities\n\n"
                
                for idx, opportunity in enumerate(top_opportunities, 1):
                    symbol = opportunity['symbol']
                    spread_type = opportunity['spread_type']
                    strikes = opportunity['strikes']
                    expiration = opportunity['expiration']
                    total_score = opportunity.get('total_score', 0)
                    quality_score = opportunity.get('quality_score', 0)
                    success_probability = opportunity.get('success_probability', 0)
                    
                    summary_content += f"## {idx}. {symbol} {spread_type} {strikes}\n"
                    summary_content += f"- **Expiration:** {expiration}\n"
                    summary_content += f"- **Total Score:** {total_score:.1f}\n"
                    summary_content += f"- **Quality Score:** {quality_score}\n"
                    summary_content += f"- **Success Probability:** {success_probability}%\n\n"
                
                self.discord_client.send_message(
                    content=summary_content,
                    webhook_type="trade_alerts",
                    username="WSB Trading Bot"
                )
                
                # Add a delay before generating detailed trade plans
                logger.info("Adding delay before generating detailed trade plans...")
                time.sleep(30)  # 30 second delay
                
                # Generate detailed trade plans for the top 3 opportunities
                top_3_opportunities = top_opportunities[:3]
                
                logger.info(f"Generating detailed trade plans for top {len(top_3_opportunities)} opportunities")
                
                for idx, opportunity in enumerate(top_3_opportunities, 1):
                    symbol = opportunity['symbol']
                    spread_type = opportunity['spread_type']
                    strikes = opportunity['strikes']
                    
                    # Generate comprehensive trade plan for this opportunity using hooks
                    trade_plan = get_trade_plan_prompt(
                        market_trend,
                        market_trend.get("options_analysis", {}),
                        stock_analyses.get(symbol, {}),
                        opportunity,
                        symbol
                    )
                    
                    trade_plan_text = self.gemini_client.generate_text(trade_plan, temperature=0.4)
                    
                    # Send detailed plan for this opportunity
                    self.discord_client.send_message(
                        content=f"# Detailed Plan #{idx}: {symbol} {spread_type} {strikes}",
                        webhook_type="trade_alerts"
                    )
                    
                    # Split the trade plan text into chunks if needed to avoid exceeding character limit
                    trade_plan_chunks = [trade_plan_text[i:i+1900] for i in range(0, len(trade_plan_text), 1900)]
                    for i, chunk in enumerate(trade_plan_chunks):
                        chunk_title = "" if i > 0 else "## Trade Plan Details:\n\n"
                        chunk_suffix = "..." if i < len(trade_plan_chunks)-1 else ""
                        
                        self.discord_client.send_message(
                            content=f"{chunk_title}{chunk}{chunk_suffix}",
                            webhook_type="trade_alerts"
                        )
                    
                    logger.info(f"Generated trade plan for opportunity #{idx}: {symbol} {spread_type} {strikes}")
                    
                    # Add delay between each trade plan generation
                    if idx < len(top_3_opportunities):
                        logger.info("Adding delay between trade plan generations...")
                        time.sleep(20)  # 20 second delay
                
                logger.info(f"Reported top {min(5, len(spread_opportunities))} credit spread opportunities with detailed plans for top {len(top_3_opportunities)}")
            else:
                # When no high-scoring opportunities, still report the top 5 available ones
                # We need to collect all spread analyses, even those that weren't "recommended"
                all_spread_analyses = []
                
                # Send a summary message about no high-scoring opportunities
                summary_content = "# Today's Credit Spread Opportunities\n\n"
                summary_content += "⚠️ **Warning: No high-scoring opportunities found today. The following are the best available options but did not meet quality thresholds.**\n\n"
                
                # Sort by total score
                all_spread_analyses.sort(key=lambda x: x.get("total_score", 0), reverse=True)
                
                # Take top 5
                top_opportunities = all_spread_analyses[:5]
                
                # Send a summary of top 5 opportunities to Discord
                self.discord_client.send_message(
                    content=summary_content,
                    webhook_type="trade_alerts",
                    username="WSB Trading Bot"
                )
                
                logger.info(f"Reported top 5 credit spread opportunities (none high-scoring)")
            
            logger.info(f"Found {len(spread_opportunities)} high-scoring credit spread opportunities")
            return spread_opportunities
            
        except Exception as e:
            logger.error(f"Error finding credit spreads: {e}")
            return spread_opportunities
    
    def run(self):
        """Run the WSB trading application."""
        logger.info("Starting WSB Trading Application...")
        
        try:
            # 1. Analyze market conditions
            market_analysis = self.analyze_market()
            if not market_analysis:
                logger.error("Failed to analyze market conditions")
                return False
            
            # 2. Analyze individual stocks
            stock_analyses = self.analyze_stocks(market_analysis)
            if not stock_analyses:
                logger.error("No stock analyses generated, aborting")
                return False
            
            # 3. Find credit spread opportunities
            credit_spreads = self.find_credit_spreads(market_analysis, stock_analyses)
            
            # 4. Log results to Notion
            self.log_results_to_notion(market_analysis, stock_analyses, credit_spreads)
            
            # 5. Send final notification - removed duplicate market analysis notification
            logger.info("Workflow completed successfully")
            logger.info("Check Notion for trade ideas and Discord for alerts")
            return True
            
        except Exception as e:
            logger.error(f"Error in main workflow: {e}")
            return False
    
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
    
    def log_results_to_notion(self, market_analysis, stock_analyses, credit_spreads=None):
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
            
            # Log top credit spread opportunities if available
            if credit_spreads and len(credit_spreads) > 0:
                # Sort by total score and take top 5
                sorted_spreads = sorted(credit_spreads, key=lambda x: x.get("total_score", 0), reverse=True)[:5]
                
                for spread in sorted_spreads:
                    if not self.notion_client.trade_log_page_id:
                        logger.warning("No trade log page ID configured, skipping credit spread logging")
                        break
                        
                    properties = {
                        "Symbol": {"title": [{"text": {"content": spread.get('symbol', '')}}]},
                        "Date": {"date": {"start": datetime.now().isoformat()}},
                        "Strategy": {"select": {"name": spread.get("spread_type", "Unknown")}},
                        "Strikes": {"rich_text": [{"text": {"content": spread.get("strikes", "")}}]},
                        "Expiration": {"date": {"start": spread.get("expiration", "2025-04-01")}},
                        "Quality Score": {"number": spread.get("quality_score", 0)},
                        "Gamble Score": {"number": spread.get("gamble_score", 0)},
                        "Total Score": {"number": spread.get("total_score", 0)},
                        "Success Probability": {"number": spread.get("success_probability", 0)},
                        "Status": {"select": {"name": "Identified"}}
                    }
                    
                    self.notion_client.add_trade_log_entry(properties)
                
                logger.info(f"Logged {len(sorted_spreads)} credit spread opportunities to Notion")
            
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
