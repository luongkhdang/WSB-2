#!/usr/bin/env python3
"""
WSB-2 Trading System Main Application (src/main.py)
---------------------------------------------------
Core orchestration module for the WSB-2 Credit Spread Trading System.

Arguments:
  --pretrain TICKER           - Run pretraining for a specific ticker
  --batch-pretrain TICKERS    - Run batch pretraining (comma-separated tickers)
  --evaluate TICKER           - Evaluate ticker prediction accuracy
  --days DAYS                 - Lookback days (default: 30)
  --update-watchlist          - Update watchlist from screener data
  --run                       - Run full trading workflow
  --extended-lookback DAYS    - Set extended pretraining lookback period

Required files:
  - data-source/options-screener-high-ivr-credit-spread-scanner.csv
  - data-source/watchlist.txt (will be created if doesn't exist)

Dependencies:
  - src.main_hooks.* - Core trading logic modules
  - src.gemini.* - Gemini AI integration
  - src.discord.* - Discord notification system
  - src.finance_client.* - Financial data providers

Author: WSB-2 Team
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from src.main_hooks.discord_hooks import create_pretraining_message_hook
from src.main_hooks.pretraining import pretrain_analyzer, batch_pretrain_analyzer, evaluate_pretraining_predictions
from src.main_hooks.credit_spreads import find_credit_spreads
from src.main_hooks.stock_analysis import analyze_stocks
from src.main_hooks.market_analysis import analyze_market, analyze_market_with_pretraining
from src.main_utilities.file_operations import get_watchlist_symbols, update_watchlist
from src.gemini.hooks import (
    get_market_trend_prompt,
    get_spy_options_prompt,
    get_market_data_prompt,
    get_stock_analysis_prompt,
    get_stock_options_prompt,
    get_trade_plan_prompt,
    get_trade_plan_prompt_from_context,
    get_pretraining_prompt,
    generate_pretraining_context
)
from src.discord.discord_client import DiscordClient
from src.finance_client.client.yfinance_client import YFinanceClient
from src.gemini.client.gemini_client import get_gemini_client
from src.main_hooks.pretraining_integration import (
    integrated_pretrain_analyzer,
    integrated_batch_pretrain_analyzer,
    use_optimized_pretraining
)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wsb_trading.log"),
        logging.StreamHandler()
    ]
)

# Set specific logger levels
logging.getLogger('yfinance_client').setLevel(logging.DEBUG)
logging.getLogger('data_processor').setLevel(logging.DEBUG)
logging.getLogger('file_operations').setLevel(logging.DEBUG)
logging.getLogger('pretraining').setLevel(logging.DEBUG)
logging.getLogger('six_step_pretraining').setLevel(logging.DEBUG)

# Lower the level for urllib3 and other noisy libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('yfinance').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import clients

# Import prompt hooks

# Import utilities and hooks


class WSBTradingApp:
    def __init__(self):
        logger.info("Initializing WSB Trading Application...")

        # Define key market indices to always include in analysis
        self.key_indices = ['SPY', 'QQQ', 'IWM', 'VTV',
                            'VGLT', 'DIA', 'BND']

        # List of ETFs (kept for reference but not used for exclusion)
        self.etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTV', 'VGLT', 'GLD', 'SLV', 'USO', 'XLF',
                     'XLE', 'XLI', 'XLK', 'XLV', 'XLP', 'XLU', 'XLB', 'XLY', 'XLC', 'XLRE', 'BND']

        # Constant Big Players to include in every watchlist
        self.constant_players = ['NVDA', 'TSLA', 'META']

        # Initialize clients
        self.gemini_client = get_gemini_client()
        self.yfinance_client = YFinanceClient()
        self.discord_client = DiscordClient()

        # Define paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data-source"

        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)

        self.screener_file = self.data_dir / \
            "options-screener-high-ivr-credit-spread-scanner.csv"
        self.watchlist_file = self.data_dir / "watchlist.txt"

        # Pretraining results storage - ensure parent directory exists first
        self.pretraining_dir = self.data_dir / "pretraining"
        self.pretraining_dir.mkdir(exist_ok=True)

        # Also create market-data directory for historical market context
        market_data_dir = self.data_dir / "market-data"
        market_data_dir.mkdir(exist_ok=True)

        # Log if using optimized pretraining
        if use_optimized_pretraining():
            logger.info("Using OPTIMIZED pretraining implementation")
        else:
            logger.info("Using traditional pretraining implementation")

        logger.info("All clients initialized successfully")

    def get_watchlist_symbols(self):
        """Get list of symbols from watchlist file."""
        return get_watchlist_symbols(self.watchlist_file)

    def update_watchlist(self):
        """Update the watchlist based on the options screener data"""
        return update_watchlist(self.screener_file, self.watchlist_file, self.key_indices, self.constant_players)

    def analyze_market(self):
        """Analyze market conditions using multiple indices"""
        return analyze_market(self.yfinance_client, self.gemini_client, get_market_trend_prompt)

    def analyze_stocks(self, market_analysis):
        """Analyze individual stocks based on market trend"""
        symbols = self.get_watchlist_symbols()
        return analyze_stocks(self.yfinance_client, self.gemini_client, market_analysis, symbols, get_stock_analysis_prompt)

    def find_credit_spreads(self, market_trend, stock_analyses):
        """Find credit spread opportunities based on analyses"""
        return find_credit_spreads(self.yfinance_client, self.gemini_client, market_trend, stock_analyses, get_trade_plan_prompt_from_context)

    def pretrain_analyzer(self, ticker, start_date=None, end_date=None, save_results=True, callback=None, discord_client=None):
        """Pretrain the AI Analyzer on historical stock data with optional callback for sending results"""
        return integrated_pretrain_analyzer(
            self.yfinance_client,
            self.gemini_client,
            self.pretraining_dir,
            ticker,
            get_pretraining_prompt,  # This will be ignored if using optimized version
            start_date,
            end_date,
            save_results,
            callback,
            discord_client=discord_client  # Pass discord_client for response tracking
        )

    def batch_pretrain_analyzer(self, tickers, start_date=None, end_date=None, save_results=True, callback=None, discord_client=None):
        """Pretrain the AI Analyzer on multiple tickers in batch"""
        return integrated_batch_pretrain_analyzer(
            self.yfinance_client,
            self.gemini_client,
            self.pretraining_dir,
            tickers,
            get_pretraining_prompt,  # This will be ignored if using optimized version
            discord_client=discord_client,  # Pass discord_client for response tracking
            start_date=start_date,
            end_date=end_date,
            save_results=save_results,
            callback=callback
        )

    def analyze_market_with_pretraining(self, ticker=None):
        """Analyze market with pretraining data for enhanced insights"""
        if not ticker:
            # Use first symbol with pretraining data
            for symbol in self.get_watchlist_symbols():
                ticker_dir = self.pretraining_dir / symbol
                latest_context_file = ticker_dir / "latest_context.txt"
                if latest_context_file.exists():
                    ticker = symbol
                    break

        if not ticker:
            logger.warning(
                "No pretraining data found, using standard market analysis")
            return self.analyze_market()

        return analyze_market_with_pretraining(
            self.yfinance_client,
            self.gemini_client,
            self.pretraining_dir,
            ticker,
            get_market_trend_prompt,
            get_spy_options_prompt
        )

    def evaluate_pretraining_predictions(self, ticker, lookback_days=30):
        """Evaluate the accuracy of pretraining predictions"""
        return evaluate_pretraining_predictions(self.yfinance_client, self.pretraining_dir, ticker, lookback_days)

    def run(self, extended_lookback=60):
        """Run the WSB trading application.

        Args:
            extended_lookback: Number of days to look back for pretraining (default: 60)
        """
        logger.info("Starting WSB Trading Application...")
        logger.info(
            f"Using extended lookback of {extended_lookback} days for pretraining")

        try:
            # Get watchlist symbols
            symbols = self.get_watchlist_symbols()

            # Make sure we have at least one pretraining ticker
            pretraining_ticker = None
            has_pretraining_data = False

            # First, check if any symbol already has pretraining data
            for symbol in symbols:
                ticker_dir = self.pretraining_dir / symbol
                latest_context_file = ticker_dir / "latest_context.txt"

                if latest_context_file.exists():
                    has_pretraining_data = True
                    pretraining_ticker = symbol
                    logger.info(
                        f"Found pretraining data for {symbol}, will use it in analysis")
                    break

            # If no pretraining data exists, select pretraining candidates
            pretraining_candidates = []
            if symbols:
                # Use all symbols in watchlist for pretraining to match --batch-pretrain behavior
                pretraining_candidates = symbols.copy()
                logger.info(
                    f"Will perform pretraining for all {len(pretraining_candidates)} symbols in watchlist")

            # Always run pretraining before analysis
            if pretraining_candidates:
                # Get date range for pretraining with extended lookback for technical indicators
                end_date = datetime.now() - timedelta(days=1)  # yesterday
                end_date_str = end_date.strftime("%Y-%m-%d")
                start_date = end_date - \
                    timedelta(days=extended_lookback)  # Extended lookback
                start_date_str = start_date.strftime("%Y-%m-%d")

                logger.info(
                    f"Running batch pretraining for {len(pretraining_candidates)} tickers from {start_date_str} to {end_date_str} ({extended_lookback}-day lookback for technical indicators)")

                try:
                    # Create a callback function that handles Discord notifications for multiple tickers
                    def batch_discord_callback(analysis):
                        # Get the ticker from the analysis
                        ticker = analysis.get(
                            "ticker", pretraining_candidates[0])

                        # Log the ticker for debugging
                        logger.info(
                            f"Processing pretraining callback for ticker: {ticker}")

                        # Ensure this ticker is one of our candidates
                        if ticker not in pretraining_candidates:
                            logger.warning(
                                f"Ticker {ticker} not in pretraining candidates {pretraining_candidates}, defaulting to {pretraining_candidates[0]}")
                            ticker = pretraining_candidates[0]

                        # Create a Discord hook specific to this ticker
                        discord_hook = create_pretraining_message_hook(
                            self.discord_client, ticker)

                        # Add ticker to analysis if missing (additional safeguard)
                        if "ticker" not in analysis:
                            analysis["ticker"] = ticker

                        # Call the discord hook with the analysis
                        logger.info(
                            f"Sending pretraining analysis for {ticker} to Discord")
                        discord_hook(analysis)

                    # Run batch pretraining
                    batch_results = self.batch_pretrain_analyzer(
                        pretraining_candidates,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        save_results=True,
                        # Pass the callback function to send results to Discord
                        callback=batch_discord_callback,
                        # Explicitly pass discord_client to ensure all Gemini responses are tracked
                        discord_client=self.discord_client
                    )

                    # Check results and send notifications
                    successful_tickers = []
                    for ticker, result in batch_results.items():
                        if "error" not in result:
                            successful_tickers.append(ticker)
                            logger.info(f"Pretraining completed for {ticker}")

                            # Send a notification about pretraining
                            if hasattr(self, 'discord_client'):
                                data_points = len(result.get("results", []))
                                pretraining_time = result.get(
                                    "processing_time", 0)

                                summary = f"Pretraining completed for {ticker}. Generated {data_points} data points across {(end_date - start_date).days + 1} days. This data will be used to enhance trading recommendations."

                                self.discord_client.send_pretraining(
                                    ticker=ticker,
                                    start_date=start_date_str,
                                    end_date=end_date_str,
                                    data_points=data_points,
                                    pretraining_time=pretraining_time,
                                    summary=summary
                                )
                                logger.info(
                                    f"Sent pretraining notification for {ticker} to Discord")
                        else:
                            logger.error(
                                f"Pretraining failed for {ticker}: {result.get('error')}")

                    # Use the first successful ticker for further analysis
                    if successful_tickers:
                        pretraining_ticker = successful_tickers[0]
                        has_pretraining_data = True
                    else:
                        # If all tickers failed pretraining, abort the entire workflow
                        error_msg = "CRITICAL ERROR: All tickers failed pretraining. Cannot proceed with analysis. Check YFinance data availability."
                        logger.critical(error_msg)
                        self.discord_client.send_error_alert(
                            title="Critical Error - Trading System Halted",
                            message=error_msg,
                            suggestions="Check data sources and API availability. Trading suspended until resolved."
                        )
                        return False

                except SystemExit as e:
                    # Propagate SystemExit from YFinance client
                    error_msg = f"CRITICAL ERROR: YFinance data retrieval failed during pretraining: {str(e)}"
                    logger.critical(error_msg)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Missing Financial Data",
                        message=error_msg,
                        suggestions="Check YFinance API status and data availability. Trading suspended."
                    )
                    raise  # Re-raise to terminate the application

                except Exception as e:
                    error_msg = f"CRITICAL ERROR during batch pretraining: {e}"
                    logger.critical(error_msg)
                    logger.exception(e)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Pretraining Failed",
                        message=error_msg,
                        suggestions="Check application logs for details. Trading suspended."
                    )
                    raise SystemExit(error_msg)
            elif pretraining_ticker:
                # Run single-ticker pretraining for the existing ticker to update it
                end_date = datetime.now() - timedelta(days=1)  # yesterday
                end_date_str = end_date.strftime("%Y-%m-%d")
                # 3 days before for update
                start_date = end_date - timedelta(days=3)
                start_date_str = start_date.strftime("%Y-%m-%d")

                logger.info(
                    f"Running update pretraining for {pretraining_ticker} from {start_date_str} to {end_date_str}")

                try:
                    # Create a hook for sending pretraining data to Discord
                    discord_hook = create_pretraining_message_hook(
                        self.discord_client, pretraining_ticker)

                    # Run pretraining with the hook
                    pretrain_result = self.pretrain_analyzer(
                        pretraining_ticker,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        save_results=True,
                        callback=discord_hook,
                        # Explicitly pass discord_client to ensure all Gemini responses are tracked
                        discord_client=self.discord_client
                    )

                    if "error" not in pretrain_result:
                        logger.info(
                            f"Pretraining update completed for {pretraining_ticker}")
                    else:
                        error_msg = f"CRITICAL ERROR: Pretraining update failed: {pretrain_result.get('error')}"
                        logger.critical(error_msg)
                        self.discord_client.send_error_alert(
                            title="Critical Error - Pretraining Update Failed",
                            message=error_msg,
                            suggestions="Check YFinance data availability. Trading suspended."
                        )
                        raise SystemExit(error_msg)
                except SystemExit as e:
                    # Propagate SystemExit from YFinance client
                    error_msg = f"CRITICAL ERROR: YFinance data retrieval failed during pretraining update: {str(e)}"
                    logger.critical(error_msg)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Missing Financial Data",
                        message=error_msg,
                        suggestions="Check YFinance API status and data availability. Trading suspended."
                    )
                    raise  # Re-raise to terminate the application
                except Exception as e:
                    error_msg = f"CRITICAL ERROR during pretraining update: {e}"
                    logger.critical(error_msg)
                    logger.exception(e)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Pretraining Update Failed",
                        message=error_msg,
                        suggestions="Check application logs for details. Trading suspended."
                    )
                    raise SystemExit(error_msg)

            # 1. Analyze market conditions with pretraining data if available
            try:
                if has_pretraining_data and pretraining_ticker:
                    logger.info(
                        f"Using pretraining context for {pretraining_ticker} in market analysis")
                    market_analysis = self.analyze_market_with_pretraining(
                        pretraining_ticker)
                else:
                    logger.warning(
                        "No pretraining data available, using standard market analysis")
                    market_analysis = self.analyze_market()

                if not market_analysis:
                    error_msg = "CRITICAL ERROR: Failed to analyze market conditions. No market data available."
                    logger.critical(error_msg)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Market Analysis Failed",
                        message=error_msg,
                        suggestions="Check YFinance data availability for market indices. Trading suspended."
                    )
                    raise SystemExit(error_msg)
            except SystemExit as e:
                # Propagate SystemExit from YFinance client
                error_msg = f"CRITICAL ERROR: YFinance data retrieval failed during market analysis: {str(e)}"
                logger.critical(error_msg)
                self.discord_client.send_error_alert(
                    title="Critical Error - Missing Market Data",
                    message=error_msg,
                    suggestions="Check YFinance API status for market indices. Trading suspended."
                )
                raise  # Re-raise to terminate the application
            except Exception as e:
                error_msg = f"CRITICAL ERROR during market analysis: {e}"
                logger.critical(error_msg)
                logger.exception(e)
                self.discord_client.send_error_alert(
                    title="Critical Error - Market Analysis Failed",
                    message=error_msg,
                    suggestions="Check application logs for details. Trading suspended."
                )
                raise SystemExit(error_msg)

            # Send market analysis to Discord
            try:
                market_trend = market_analysis.get('trend', 'neutral')
                market_title = f"Market Analysis: {market_trend.title()} Trend"

                # Extract metrics for the Discord message
                metrics = {
                    "Trend": market_trend,
                    "Score": market_analysis.get('market_trend_score', 0),
                    "Risk": market_analysis.get('risk_adjustment', 'standard'),
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }

                # Send to Discord
                self.discord_client.send_market_analysis(
                    title=market_title,
                    content=market_analysis.get(
                        'full_analysis', 'No detailed analysis available'),
                    metrics=metrics
                )
                logger.info("Sent market analysis to Discord")
            except Exception as e:
                logger.error(f"Error sending market analysis to Discord: {e}")
                # Non-critical error, continue workflow

            # 2. Analyze individual stocks
            try:
                stock_analyses = self.analyze_stocks(market_analysis)
                if not stock_analyses:
                    error_msg = "CRITICAL ERROR: No stock analyses generated. Cannot proceed without stock data."
                    logger.critical(error_msg)
                    self.discord_client.send_error_alert(
                        title="Critical Error - Stock Analysis Failed",
                        message=error_msg,
                        suggestions="Check YFinance data availability for watchlist stocks. Trading suspended."
                    )
                    raise SystemExit(error_msg)
            except SystemExit as e:
                # Propagate SystemExit from YFinance client
                error_msg = f"CRITICAL ERROR: YFinance data retrieval failed during stock analysis: {str(e)}"
                logger.critical(error_msg)
                self.discord_client.send_error_alert(
                    title="Critical Error - Missing Stock Data",
                    message=error_msg,
                    suggestions="Check YFinance API status for watchlist stocks. Trading suspended."
                )
                raise  # Re-raise to terminate the application
            except Exception as e:
                error_msg = f"CRITICAL ERROR during stock analysis: {e}"
                logger.critical(error_msg)
                logger.exception(e)
                self.discord_client.send_error_alert(
                    title="Critical Error - Stock Analysis Failed",
                    message=error_msg,
                    suggestions="Check application logs for details. Trading suspended."
                )
                raise SystemExit(error_msg)

            # Check for errors in individual stock analyses
            failed_stocks = {}
            for symbol, analysis in stock_analyses.items():
                if 'error' in analysis:
                    failed_stocks[symbol] = analysis['error']
                    logger.error(
                        f"Analysis failed for {symbol}: {analysis['error']}")

            # If ALL stocks failed analysis, abort the workflow
            if failed_stocks and len(failed_stocks) == len(stock_analyses):
                error_msg = f"CRITICAL ERROR: All stocks failed analysis. Cannot proceed with trading strategy."
                logger.critical(error_msg)
                error_details = "\n".join(
                    [f"{symbol}: {error}" for symbol, error in failed_stocks.items()])
                logger.critical(f"Error details:\n{error_details}")
                self.discord_client.send_error_alert(
                    title="Critical Error - All Stock Analyses Failed",
                    message=error_msg,
                    suggestions="Check YFinance data availability for watchlist stocks. Trading suspended."
                )
                raise SystemExit(error_msg)

            # Send stock analyses to Discord
            try:
                # Send each stock analysis to Discord
                for symbol, analysis in stock_analyses.items():
                    if 'error' not in analysis:
                        # Send analysis to full-analysis webhook
                        self.discord_client.send_analysis(
                            analysis=analysis,
                            ticker=symbol,
                            title=f"{symbol} Analysis: {analysis.get('trend', 'neutral').title()}"
                        )
                logger.info(
                    f"Sent {len(stock_analyses) - len(failed_stocks)} stock analyses to Discord")
            except Exception as e:
                logger.error(f"Error sending stock analyses to Discord: {e}")
                # Non-critical error, continue workflow

            # 3. Find credit spread opportunities
            try:
                credit_spreads = self.find_credit_spreads(
                    market_analysis, stock_analyses)
                if not credit_spreads and not failed_stocks:
                    logger.warning(
                        "No credit spread opportunities found, but workflow completed successfully")
            except SystemExit as e:
                # Propagate SystemExit from YFinance client
                error_msg = f"CRITICAL ERROR: YFinance data retrieval failed during credit spread analysis: {str(e)}"
                logger.critical(error_msg)
                self.discord_client.send_error_alert(
                    title="Critical Error - Missing Options Data",
                    message=error_msg,
                    suggestions="Check YFinance API status for options data. Trading suspended."
                )
                raise  # Re-raise to terminate the application
            except Exception as e:
                error_msg = f"CRITICAL ERROR during credit spread analysis: {e}"
                logger.critical(error_msg)
                logger.exception(e)
                self.discord_client.send_error_alert(
                    title="Critical Error - Credit Spread Analysis Failed",
                    message=error_msg,
                    suggestions="Check application logs for details. Trading suspended."
                )
                raise SystemExit(error_msg)

            # Send credit spread opportunities to Discord
            try:
                if credit_spreads:
                    # Send each credit spread to Discord as a trade alert
                    for spread in credit_spreads:
                        symbol = spread.get('symbol', 'UNKNOWN')
                        spread_type = spread.get('spread_type', 'unknown')
                        direction = spread.get('direction', 'neutral')
                        strikes = spread.get('strikes', 'N/A')
                        expiration = spread.get('expiration', 'N/A')

                        # Create a note with details
                        notes = (
                            f"Type: {spread_type}\n"
                            f"Strikes: {strikes}\n"
                            f"Expiration: {expiration}\n"
                            f"Premium: ${spread.get('premium', 0):.2f}\n"
                            f"Max Loss: ${spread.get('max_loss', 0):.2f}\n"
                            f"Probability: {spread.get('success_probability', 0)}%\n"
                            f"Total Score: {spread.get('total_score', 0)}"
                        )

                        # Send a trade alert
                        self.discord_client.send_trade_alert(
                            ticker=symbol,
                            action=direction.upper(),
                            price=spread.get('premium', 0),
                            notes=notes
                        )

                    logger.info(
                        f"Sent {len(credit_spreads)} credit spread alerts to Discord")
                else:
                    logger.info(
                        "No credit spread opportunities found to send to Discord")
            except Exception as e:
                logger.error(f"Error sending credit spreads to Discord: {e}")
                # Non-critical error for final step

            # 5. Send final notification
            logger.info("Workflow completed successfully")
            logger.info("Check Discord for alerts")
            return True

        except SystemExit as e:
            # This will catch any SystemExit exceptions raised throughout the workflow
            error_msg = f"Application terminated due to critical error: {str(e)}"
            logger.critical(error_msg)
            try:
                self.discord_client.send_error_alert(
                    title="Critical Error - Application Terminated",
                    message=error_msg,
                    suggestions="Check logs for details. This is likely a data availability issue."
                )
            except Exception as discord_err:
                logger.critical(f"Failed to send Discord alert: {discord_err}")

            # Re-raise the exception to properly terminate the application
            raise

        except Exception as e:
            error_msg = f"Unhandled exception in main workflow: {e}"
            logger.critical(error_msg)
            logger.exception(e)
            try:
                self.discord_client.send_error_alert(
                    title="Critical Error - Unexpected Exception",
                    message=error_msg,
                    suggestions="Check application logs for details. Trading suspended."
                )
            except Exception as discord_err:
                logger.critical(f"Failed to send Discord alert: {discord_err}")
            return False


def main():
    """
    Main function to run the WSB Trading workflow
    """
    print("======================================")
    print("  WSB-2 Credit Spread Trading System  ")
    print("  Powered by Gemini & YFinance        ")
    print("======================================")
    print(
        f"Starting workflow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WSB Trading Application')

    # Add arguments
    parser.add_argument('--pretrain', dest='pretrain_ticker',
                        help='Run pretraining for a specific ticker')
    parser.add_argument('--batch-pretrain', dest='batch_pretrain',
                        help='Run batch pretraining for comma-separated tickers (e.g., "SPY,AAPL,MSFT")')
    parser.add_argument('--evaluate', dest='evaluate_ticker',
                        help='Evaluate pretraining predictions for a specific ticker')
    parser.add_argument('--days', dest='lookback_days', type=int, default=30,
                        help='Lookback days for evaluation or pretraining (default: 30)')
    parser.add_argument('--update-watchlist', dest='update_watchlist', action='store_true',
                        help='Update the watchlist from options screener data')
    parser.add_argument('--run', dest='run_workflow', action='store_true',
                        help='Run the full trading workflow (includes batch pretraining for entire watchlist, market analysis and credit spread discovery)')
    parser.add_argument('--extended-lookback', dest='extended_lookback', type=int, default=60,
                        help='Set extended lookback period in days for pretraining (default: 60 days, recommended for technical indicators)')

    args = parser.parse_args()

    # Initialize the app
    app = WSBTradingApp()

    success = True

    # Handle commands
    if args.update_watchlist:
        print("Updating watchlist from options screener data...")
        success = app.update_watchlist()
        if success:
            print("✅ Watchlist updated successfully!")
        else:
            print("❌ Failed to update watchlist.")

    # Handle pretrain command
    if args.pretrain_ticker:
        print(f"Running pretraining for {args.pretrain_ticker}...")

        # Set dates
        end_date = datetime.now() - timedelta(days=1)  # yesterday
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Use extended lookback if specified
        if args.extended_lookback:
            print(
                f"Using extended lookback period of {args.extended_lookback} days for sufficient technical indicator data")
            start_date = end_date - timedelta(days=args.extended_lookback)
        else:
            # This should not happen with the default, but keeping as a fallback
            start_date = end_date - timedelta(days=args.lookback_days)
            print(
                f"Using lookback period of {args.lookback_days} days (may be insufficient for some technical indicators)")

        start_date_str = start_date.strftime("%Y-%m-%d")
        print(f"Pretraining period: {start_date_str} to {end_date_str}")

        # Create a hook for sending pretraining data to Discord
        discord_hook = create_pretraining_message_hook(
            app.discord_client, args.pretrain_ticker)

        # Run pretraining with progress output
        print("Starting pretraining process with enhanced pattern recognition...")
        result = app.pretrain_analyzer(
            args.pretrain_ticker,
            start_date=start_date_str,
            end_date=end_date_str,
            save_results=True,
            callback=lambda analysis: print(
                f"Processed analysis for {analysis.get('date', 'unknown')} - {analysis.get('analysis_type', 'unknown')}")
        )

        # Check for errors
        if "error" in result:
            print(f"❌ Pretraining failed: {result['error']}")
            success = False
        else:
            analyses_count = result.get("analyses_count", 0)
            processing_time = result.get("processing_time", 0)

            # Check for data quality warnings
            if any("quality_warning" in analysis for analysis in result.get("results", [])):
                print("⚠️ Pretraining completed with data quality warnings.")
                print(
                    "    Some pattern recognition features may be limited due to insufficient historical data.")
                print(
                    "    Try using --extended-lookback with a larger value for better results.")
            else:
                print("✅ Pretraining completed successfully!")

            print(
                f"Generated {analyses_count} analysis points in {processing_time:.2f} seconds")

            # Send to Discord
            try:
                if hasattr(app, 'discord_client'):
                    summary = None
                    for r in result.get("results", []):
                        if r.get("type") == "summary":
                            summary = r
                            break

                    app.discord_client.send_pretraining(
                        ticker=args.pretrain_ticker,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        data_points=analyses_count,
                        pretraining_time=processing_time,
                        summary=summary["full_analysis"] if summary else "No summary generated"
                    )
                    print("Sent pretraining summary to Discord")
            except Exception as e:
                print(f"Failed to send to Discord: {e}")

    if args.batch_pretrain:
        tickers = [t.strip() for t in args.batch_pretrain.split(',')]
        print(
            f"Running batch pretraining for {len(tickers)} tickers: {', '.join(tickers)}...")

        # Set dates using extended_lookback
        end_date = datetime.now() - timedelta(days=1)  # yesterday
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Use extended lookback to ensure sufficient data for technical indicators
        start_date = end_date - timedelta(days=args.extended_lookback)
        start_date_str = start_date.strftime("%Y-%m-%d")

        print(
            f"Pretraining period: {start_date_str} to {end_date_str} ({args.extended_lookback}-day lookback)")

        # Run batch pretraining with dates
        batch_results = app.batch_pretrain_analyzer(
            tickers,
            start_date=start_date_str,
            end_date=end_date_str
        )

        # Show results
        successful = 0
        for ticker, result in batch_results.items():
            if "error" not in result:
                successful += 1
                analyses_count = result.get('analyses_count', 0)
                print(
                    f"✅ {ticker}: Pretraining completed with {analyses_count} analyses")
            else:
                print(f"❌ {ticker}: Pretraining failed - {result.get('error')}")

        print(
            f"\n✅ Batch pretraining completed for {successful}/{len(tickers)} tickers")
        success = successful > 0

    if args.evaluate_ticker:
        print(
            f"Evaluating pretraining predictions for {args.evaluate_ticker} over the last {args.lookback_days} days...")
        result = app.evaluate_pretraining_predictions(
            args.evaluate_ticker, args.lookback_days)
        if "error" not in result:
            metrics = result.get("metrics", {})
            print(f"\n✅ Evaluation completed for {args.evaluate_ticker}:")
            print(
                f"• Predictions analyzed: {result.get('prediction_count', 0)}")
            for horizon, horizon_metrics in metrics.items():
                print(f"• {horizon.replace('_', ' ').title()} Horizon:")
                print(
                    f"  - Directional Accuracy: {horizon_metrics.get('directional_accuracy')}")
                print(
                    f"  - Avg. Magnitude Error: {horizon_metrics.get('avg_magnitude_error')}")
                print(f"  - Sample Size: {horizon_metrics.get('sample_size')}")

            success = True
        else:
            print(f"❌ Evaluation failed: {result.get('error')}")
            success = False

    if args.run_workflow or not (args.update_watchlist or args.pretrain_ticker or args.batch_pretrain or args.evaluate_ticker):
        # Run the full analysis workflow
        print(
            f"Running full market analysis workflow with batch pretraining for all watchlist tickers (using 60-day lookback)...")
        # Always use 60 days for --run regardless of user input
        success = app.run(60)

    if success:
        print("\n✅ Workflow completed successfully!")
    else:
        print("\n❌ Workflow failed. Check logs for details.")

    print("\nSee Discord for alerts.")
    print(
        f"Workflow completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return success


if __name__ == "__main__":
    main()
