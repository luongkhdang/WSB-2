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

        # Set the strict validation mode flag
        self.strict_validation = True

        logger.info("All clients initialized successfully")

    def validate_data_quality(self, ticker, min_data_points=60):
        """
        Validate data quality for a ticker before processing

        Args:
            ticker: Stock symbol to validate
            min_data_points: Minimum data points required

        Raises:
            ValueError: If critical data is missing or invalid
        """
        from src.main_utilities.data_processor import format_stock_data_for_analysis, check_for_fallbacks
        from src.main_hooks.six_step_pretraining import validate_data_quality

        logger.info(f"Validating data quality for {ticker}")

        try:
            # Calculate start date (500 days ago from today to ensure enough trading days for SMA200)
            # This provides ~350+ trading days accounting for weekends and holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=500)

            # Get historical data - use correct parameters
            historical_data = self.yfinance_client.get_historical_data(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval='1d'
            )

            # Validate there's enough data
            if historical_data is None or historical_data.empty or len(historical_data) < min_data_points:
                error_msg = f"Insufficient historical data for {ticker}: {len(historical_data) if historical_data is not None and not historical_data.empty else 0} points < {min_data_points} minimum required. Application stopped for debugging."
                logger.error(error_msg)
                exit_on_data_error(self, error_msg, ticker)

            # Format and validate the data
            today = datetime.now().strftime('%Y-%m-%d')
            formatted_data = format_stock_data_for_analysis(
                historical_data,
                ticker,
                today,
                min_data_points=min_data_points
            )

            # Ensure price_history field is populated before validation
            if 'price_history' not in formatted_data:
                # Create price history array from historical data (at least 60 days for validation requirements)
                lookback = min(max(60, min_data_points), len(historical_data))
                # Convert to numpy array first, then to list to avoid AttributeError
                price_data = historical_data['Close'].iloc[-lookback:].to_numpy().tolist()
                formatted_data['price_history'] = price_data
                logger.info(
                    f"Added price_history field with {len(price_data)} days of data for {ticker}")

            # Additional validation - use parameter names matching the function signature
            validate_data_quality(
                formatted_data, ticker, min_days_required=min_data_points, strict_validation=self.strict_validation)

            # Check for any fallback values that might have been used
            check_for_fallbacks(formatted_data, ticker,
                                strict_mode=self.strict_validation)

            logger.info(f"Data validation successful for {ticker}")
            return formatted_data

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Data validation error for {ticker}: {error_msg}")
            if self.strict_validation:
                exit_on_data_error(self, error_msg, ticker)
            raise
        except Exception as e:
            error_msg = f"Unexpected error during data validation for {ticker}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            exit_on_data_error(self, error_msg, ticker)

    def setup_data_validation(self):
        """
        Perform data validation for key indices and tickers before starting the app

        Raises:
            ValueError: If any key index fails validation
        """
        logger.info("Performing initial data validation for key market indices")

        # Validate key indices first
        for index in self.key_indices:
            try:
                self.validate_data_quality(index)
                logger.info(f"Validation successful for {index}")
            except ValueError as e:
                logger.error(f"CRITICAL ERROR in key index {index}: {str(e)}")
                logger.error(
                    "Application stopped for debugging - fix data issues before continuing")
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error validating {index}: {str(e)}", exc_info=True)
                logger.error(
                    "Application stopped for debugging - fix data issues before continuing")
                raise ValueError(
                    f"Unexpected error validating {index}: {str(e)}")

        # Additional setup validations
        logger.info("All key indices passed validation")
        return True

    def get_watchlist_symbols(self):
        """Get list of symbols from watchlist file."""
        return get_watchlist_symbols(self.watchlist_file)

    def update_watchlist(self):
        """Update the watchlist based on the options screener data"""
        return update_watchlist(self.screener_file, self.watchlist_file, self.key_indices, self.constant_players)

    def analyze_market(self):
        """Analyze market conditions using multiple indices"""
        # First validate market indices data
        for index in ['SPY', 'QQQ', 'IWM']:
            self.validate_data_quality(index)

        return analyze_market(self.yfinance_client, self.gemini_client, get_market_trend_prompt)

    def analyze_stocks(self, market_analysis):
        """Analyze individual stocks based on market trend"""
        symbols = self.get_watchlist_symbols()

        # Validate symbols data before analysis
        valid_symbols = []
        for symbol in symbols:
            try:
                self.validate_data_quality(symbol)
                valid_symbols.append(symbol)
            except ValueError as e:
                logger.warning(
                    f"Skipping {symbol} due to data validation failure: {str(e)}")

        if not valid_symbols:
            raise ValueError(
                "No valid symbols to analyze after data validation. Application stopped for debugging.")

        return analyze_stocks(self.yfinance_client, self.gemini_client, market_analysis, valid_symbols, get_stock_analysis_prompt)

    def find_credit_spreads(self, market_trend, stock_analyses):
        """Find credit spread opportunities based on analyses"""
        return find_credit_spreads(self.yfinance_client, self.gemini_client, market_trend, stock_analyses, get_trade_plan_prompt_from_context)

    def pretrain_analyzer(self, ticker, start_date=None, end_date=None, save_results=True, callback=None, discord_client=None):
        """Pretrain the AI Analyzer on historical stock data with optional callback for sending results"""
        # Validate data quality first
        self.validate_data_quality(ticker)

        return integrated_pretrain_analyzer(
            self.yfinance_client,
            self.gemini_client,
            self.pretraining_dir,
            ticker,
            get_pretraining_prompt,  # This will be ignored if using optimized version
            start_date=start_date,
            end_date=end_date,
            save_results=save_results,
            callback=callback,
            discord_client=discord_client  # Pass discord_client for response tracking
        )

    def batch_pretrain_analyzer(self, tickers, start_date=None, end_date=None, save_results=True, callback=None, discord_client=None):
        """Pretrain the AI Analyzer on multiple tickers in batch"""
        # Validate each ticker first
        valid_tickers = []
        for ticker in tickers:
            try:
                self.validate_data_quality(ticker)
                valid_tickers.append(ticker)
            except ValueError as e:
                logger.warning(
                    f"Skipping {ticker} in batch pretrain due to data validation failure: {str(e)}")

        if not valid_tickers:
            raise ValueError(
                "No valid tickers to pretrain after data validation. Application stopped for debugging.")

        # Pass strict_validation through kwargs
        kwargs = {
            'start_date': start_date,
            'end_date': end_date,
            'save_results': save_results,
            'callback': callback,
            'strict_validation': self.strict_validation
        }

        return integrated_batch_pretrain_analyzer(
            self.yfinance_client,
            self.gemini_client,
            self.pretraining_dir,
            valid_tickers,
            get_pretraining_prompt,  # This will be ignored if using optimized version
            discord_client=discord_client,  # Pass discord_client for response tracking
            **kwargs
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

        # Validate data quality for the selected ticker
        self.validate_data_quality(ticker)

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
        # Validate data quality for the ticker
        self.validate_data_quality(ticker)

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
            # Perform initial data validation for key market indices
            logger.info(
                "Performing initial data validation for key market indices")
            for ticker in self.key_indices:
                self.validate_data_quality(ticker)
                logger.info(f"Validation successful for {ticker}")

            logger.info("All key indices passed validation")

            # Get watchlist
            watchlist = get_watchlist_symbols(self.watchlist_file)
            logger.info(f"Found {len(watchlist)} symbols in watchlist")

            # Always run pretraining before analysis
            if self.gemini_client:
                # Get date range for pretraining with extended lookback for technical indicators
                # Use yesterday's date as the end date to ensure data availability
                end_date = datetime.now() - timedelta(days=1)
                # Ensure it's a weekday (Monday-Friday)
                while end_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                    end_date = end_date - timedelta(days=1)

                end_date_str = end_date.strftime("%Y-%m-%d")
                # Use the extended lookback parameter
                start_date = end_date - timedelta(days=extended_lookback)
                start_date_str = start_date.strftime("%Y-%m-%d")

                # Look for existing pretraining data for SPY
                pretraining_files = list(
                    self.pretraining_dir.glob("SPY/*.json"))
                if len(pretraining_files) > 0:
                    logger.info(
                        "Found pretraining data for SPY, will use it in analysis")
                else:
                    # No existing pretraining data, need to run pretraining for SPY
                    logger.info(
                        "No pretraining data for SPY, running pretraining process")
                    try:
                        integrated_pretrain_analyzer(
                            self.yfinance_client,
                            self.gemini_client,
                            self.pretraining_dir,
                            "SPY",
                            start_date=start_date_str,
                            end_date=end_date_str,
                            discord_client=self.discord_client
                        )
                    except Exception as e:
                        logger.error(f"Error during SPY pretraining: {e}")
                        # Critical - send alert
                        if self.discord_client:
                            self.discord_client.send_error_alert(
                                message=f"Critical Error - Pretraining Failure: Failed to pretrain SPY: {e}"
                            )
                        # Continue with other processing

                # Now we run pretraining for all tickers in the watchlist
                logger.info(
                    f"Will perform pretraining for all {len(watchlist)} symbols in watchlist")

                logger.info(
                    f"Running batch pretraining for {len(watchlist)} tickers from {start_date_str} to {end_date_str} ({extended_lookback}-day lookback for technical indicators)")
                try:
                    # Pass strict_validation through kwargs
                    batch_kwargs = {
                        'start_date': start_date_str,
                        'end_date': end_date_str,
                        'strict_validation': self.strict_validation
                    }

                    batch_results = integrated_batch_pretrain_analyzer(
                        self.yfinance_client,
                        self.gemini_client,
                        self.pretraining_dir,
                        watchlist,
                        discord_client=self.discord_client,
                        **batch_kwargs
                    )
                    logger.info(
                        f"Batch pretraining complete for {len(batch_results)} tickers")
                except Exception as e:
                    logger.error(f"Error during batch pretraining: {e}")
                    # Critical - send alert
                    if self.discord_client:
                        self.discord_client.send_error_alert(
                            message=f"Critical Error - Batch Pretraining Failure: Failed to complete batch pretraining: {e}"
                        )

            # Run SPY analysis with pretraining integration
            try:
                market_result = analyze_market_with_pretraining(
                    self.yfinance_client,
                    self.gemini_client,
                    self.pretraining_dir,
                    "SPY",  # Use SPY as the default ticker
                    get_market_trend_prompt,
                    get_spy_options_prompt
                )
                logger.info("Market analysis with pretraining complete")
            except Exception as e:
                logger.error(f"Error during market analysis: {e}")
                # Critical - send alert
                if self.discord_client:
                    self.discord_client.send_error_alert(
                        message=f"Critical Error - Market Analysis Failure: Failed to analyze market: {e}"
                    )
                # Stop here if market analysis fails
                raise ValueError(f"Critical error in market analysis: {e}")

            # Process and alert on pretraining results
            if self.discord_client:
                # Detailed pretraining reports
                for ticker in watchlist[:10]:  # Limit to first 10 to avoid spam
                    try:
                        # Check if pretraining file exists
                        ticker_files = list(
                            self.pretraining_dir.glob(f"{ticker}/*.json"))
                        if not ticker_files:
                            continue

                        pretraining_time = "< 1 minute"  # Default
                        data_points = "Unknown"  # Default

                        # Send pretraining summary to Discord
                        self.discord_client.send_pretraining(
                            ticker=ticker,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            data_points=data_points,
                            pretraining_time=pretraining_time
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error sending pretraining report for {ticker}: {e}")

            # Send pretraining complete message
            if self.discord_client:
                self.discord_client.send_message(
                    content=f"⚙️ Pretraining complete for {len(watchlist)} symbols",
                    webhook_type="alerts"
                )

            # Done!
            logger.info("Trading workflow complete")
            return True

        except Exception as e:
            logger.critical(
                f"Application terminated due to critical error: {str(e)}")
            logger.exception(e)

            # Send critical error alert
            if hasattr(self, 'discord_client') and self.discord_client:
                self.discord_client.send_error_alert(
                    message=f"Critical Error - Application Terminated: The WSB Trading System has been halted due to a critical error: {str(e)}"
                )
            return False


def exit_on_data_error(app, error_message, ticker=None):
    """
    Exit the application due to a critical data validation error

    Args:
        app: The WSBTradingApp instance
        error_message: The error message to log and send
        ticker: Optional ticker symbol related to the error

    Returns:
        Never returns, exits the application with code 1
    """
    logger.error(f"CRITICAL ERROR: {error_message}")
    logger.error("Application stopped for debugging.")

    # Construct a message for Discord
    message = f"**⚠️ CRITICAL DATA ERROR ⚠️**\n\n"
    if ticker:
        message += f"**Ticker:** {ticker}\n"
    message += f"**Error:** {error_message}\n\n"
    message += "**Action Required:** Application halted. Investigate data issues and fix before continuing.\n"
    message += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Send to Discord if available
    try:
        if hasattr(app, 'discord_client'):
            app.discord_client.send_message(message, "ERROR_ALERTS")
            logger.info("Error notification sent to Discord.")
    except Exception as e:
        logger.error(f"Failed to send error to Discord: {e}")

    # Exit with error code
    sys.exit(1)


def main():
    """Main entry point for the WSB Trading System application"""
    try:
        logger.info("Starting WSB Trading System")

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='WSB Trading System - Credit Spread Trading Platform')
        parser.add_argument('--pretrain', type=str,
                            help='Run pretraining for a specific ticker')
        parser.add_argument('--batch-pretrain', type=str,
                            help='Run batch pretraining (comma-separated tickers)')
        parser.add_argument('--evaluate', type=str,
                            help='Evaluate ticker prediction accuracy')
        parser.add_argument('--days', type=int, default=30,
                            help='Lookback days (default: 30)')
        parser.add_argument('--update-watchlist', action='store_true',
                            help='Update watchlist from screener data')
        parser.add_argument('--run', action='store_true',
                            help='Run full trading workflow')
        parser.add_argument('--extended-lookback', type=int, default=60,
                            help='Set extended pretraining lookback period')
        parser.add_argument('--strict-validation', action='store_true', default=True,
                            help='Enable strict data validation (default: True)')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        args = parser.parse_args()

        # Initialize the application
        app = WSBTradingApp()

        # Set strict validation mode based on argument
        app.strict_validation = args.strict_validation
        if args.strict_validation:
            logger.info(
                "Strict data validation enabled - app will fail fast on missing or invalid data")
        else:
            logger.warning(
                "Strict data validation disabled - app will attempt to use fallbacks for missing data")

        # Enable debug mode if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Update the watchlist if requested
        if args.update_watchlist:
            logger.info("Updating watchlist...")
            app.update_watchlist()

        # Run pretraining for a single ticker if specified
        if args.pretrain:
            logger.info(f"Running pretraining for {args.pretrain}...")

            # Create a callback to send messages to Discord
            discord_callback = create_pretraining_message_hook(
                app.discord_client, "PRETRAINING")

            # Run pretraining with Discord callback
            result = app.pretrain_analyzer(
                args.pretrain,
                start_date=None,  # Use default date calculation
                end_date=None,  # Use default date calculation
                callback=discord_callback,
                discord_client=app.discord_client
            )

            logger.info(f"Pretraining completed for {args.pretrain}")

        # Run batch pretraining if specified
        if args.batch_pretrain:
            logger.info(
                f"Running batch pretraining for {args.batch_pretrain}...")
            tickers = [t.strip() for t in args.batch_pretrain.split(',')]

            # Create a callback to send messages to Discord
            discord_callback = create_pretraining_message_hook(
                app.discord_client, "BATCH_PRETRAINING")

            # Run batch pretraining with Discord callback
            result = app.batch_pretrain_analyzer(
                tickers,
                start_date=None,  # Use default date calculation
                end_date=None,  # Use default date calculation
                callback=discord_callback,
                discord_client=app.discord_client
            )

            logger.info(
                f"Batch pretraining completed for {len(tickers)} tickers")

        # Evaluate pretraining predictions if specified
        if args.evaluate:
            logger.info(
                f"Evaluating pretraining predictions for {args.evaluate}...")
            result = app.evaluate_pretraining_predictions(
                args.evaluate, args.days)

            if result:
                accuracy = result.get('directional_accuracy', 0)
                logger.info(
                    f"Prediction accuracy for {args.evaluate}: {accuracy:.2f}%")

                # Send results to Discord
                content = f"**Pretraining Evaluation for {args.evaluate}**\n"
                content += f"Directional Accuracy: {accuracy:.2f}%\n"
                content += f"Total Predictions: {result.get('total_predictions', 0)}\n"
                content += f"Average Magnitude Error: {result.get('avg_magnitude_error', 0):.2f}%"

                app.discord_client.send_message(content, "ERROR_ALERTS")

        # Run full trading workflow if requested
        if args.run:
            logger.info("Running full trading workflow...")

            # First, validate data quality for key market indices
            try:
                app.setup_data_validation()
            except ValueError as e:
                error_msg = str(e)
                if "Application stopped for debugging" in error_msg:
                    # Use more detailed error handling for data validation errors
                    exit_on_data_error(app, error_msg)
                logger.error(f"Data validation failed, aborting workflow: {e}")
                return 1

            # Run the main trading app with extended lookback
            app.run(extended_lookback=args.extended_lookback)

        # If no action specified, print help
        if not (args.update_watchlist or args.pretrain or args.batch_pretrain or args.evaluate or args.run):
            parser.print_help()
            return 0

        return 0  # Successful execution

    except ValueError as e:
        error_msg = str(e)
        # Check if this is a critical data validation error
        if "Application stopped for debugging" in error_msg:
            # This is a critical data error that needs investigation
            # Extract ticker if present in the message
            ticker = None
            import re
            match = re.search(r"for\s+([A-Z]+)[\s\.\:]", error_msg)
            if match:
                ticker = match.group(1)
            # Use exit_on_data_error for a cleaner exit with notifications
            exit_on_data_error(app, error_msg, ticker)
        else:
            # For other ValueErrors, just log and exit with code 1
            logger.error(f"ERROR: {error_msg}")
            return 1
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        error_msg = f"Unhandled exception: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Try to send error to Discord if possible
        try:
            # Only attempt if we have an app instance
            if 'app' in locals() and hasattr(app, 'discord_client'):
                content = f"**❌ UNHANDLED ERROR ❌**\n\n"
                content += f"**Error:** {str(e)}\n\n"
                content += f"See logs for stack trace.\n"
                content += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                app.discord_client.send_message(content, "ERROR_ALERTS")
        except:
            # Silently ignore errors in error handling
            pass

        return 1  # Error exit code


if __name__ == "__main__":
    sys.exit(main())
