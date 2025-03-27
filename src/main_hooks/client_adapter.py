import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import pandas as pd

# Import hooks from finance client
from src.finance_client.hooks.pretraining import (
    perform_pretraining_analysis,
    generate_trading_scenarios,
    rank_pretraining_symbols
)

from src.finance_client.hooks.data_retrieval import (
    get_historical_data,
    get_market_data,
    get_options_chain,
    get_volatility_data
)

from src.finance_client.hooks.technical_indicators_hook import (
    enrich_with_technical_indicators,
    get_indicator_summary
)

logger = logging.getLogger(__name__)

class FinanceClientAdapter:
    """Adapter for the finance client that connects with the main application"""
    
    def __init__(self, cache_dir: str = "./data-cache"):
        """
        Initialize the adapter
        
        Args:
            cache_dir: Directory for data cache
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized FinanceClientAdapter with cache dir: {self.cache_dir}")
    
    def pretrain(self, symbols: List[str], lookback_days: int = 180) -> Dict[str, Any]:
        """
        Run pretraining analysis for a list of symbols
        
        Args:
            symbols: List of ticker symbols to analyze
            lookback_days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with pretraining results
        """
        logger.info(f"Starting pretraining for {len(symbols)} symbols")
        
        try:
            # Perform pretraining analysis
            results = perform_pretraining_analysis(
                symbols, 
                lookback_days=lookback_days,
                include_options=True,
                cache_dir=self.cache_dir
            )
            
            # Generate trading scenarios
            scenarios = generate_trading_scenarios(results)
            
            # Rank symbols
            rankings = rank_pretraining_symbols(results)
            
            # Add scenarios and rankings to results
            results["trading_scenarios"] = scenarios
            results["symbol_rankings"] = rankings
            
            return results
            
        except Exception as e:
            logger.error(f"Error in pretraining: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def get_market_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the current market conditions
        
        Returns:
            Dictionary with market overview data
        """
        logger.info("Getting market overview")
        
        try:
            # Get market data
            market_data = get_market_data(cache_dir=self.cache_dir)
            
            # Get volatility data
            volatility_data = get_volatility_data(cache_dir=self.cache_dir)
            
            # Format results
            overview = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "volatility_data": volatility_data
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def get_symbol_data(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific symbol
        
        Args:
            symbol: Ticker symbol
            days: Number of days of historical data
            
        Returns:
            Dictionary with symbol data
        """
        logger.info(f"Getting data for {symbol}")
        
        try:
            # Get dates
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Get historical data
            data = get_historical_data(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval="1d", 
                cache_dir=self.cache_dir
            )
            
            if data is None or len(data) == 0:
                return {"status": "error", "error_message": f"No data available for {symbol}"}
            
            # Enrich with technical indicators
            enriched_data = enrich_with_technical_indicators(data, symbol)
            
            # Get technical summary
            technical_summary = get_indicator_summary(enriched_data, symbol)
            
            # Get options data
            options_chain = get_options_chain(symbol, cache_dir=self.cache_dir)
            
            # Format results
            result = {
                "symbol": symbol,
                "current_price": float(data['Close'].iloc[-1]),
                "timestamp": datetime.now().isoformat(),
                "data_points": len(data),
                "technical_summary": technical_summary,
                "has_options_data": options_chain is not None
            }
            
            # Add quality score if available
            if hasattr(data, 'attrs') and 'quality_score' in data.attrs:
                result["data_quality_score"] = float(data.attrs['quality_score'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def save_pretraining_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save pretraining results to a file
        
        Args:
            results: Pretraining results
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pretraining_results_{timestamp}.json"
        
        # Ensure the filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.cache_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Full path to the file
        filepath = os.path.join(results_dir, filename)
        
        try:
            # Convert DataFrame attributes to serializable format
            serializable_results = self._make_serializable(results)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved pretraining results to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving pretraining results: {e}")
            return None
    
    def load_pretraining_results(self, filename: str) -> Dict[str, Any]:
        """
        Load pretraining results from a file
        
        Args:
            filename: Filename to load
            
        Returns:
            Dictionary with pretraining results
        """
        # Ensure the filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Results directory
        results_dir = os.path.join(self.cache_dir, "results")
        
        # Full path to the file
        filepath = os.path.join(results_dir, filename)
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return {"status": "error", "error_message": f"File not found: {filename}"}
            
            # Load from file
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded pretraining results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading pretraining results: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def _make_serializable(self, data: Any) -> Any:
        """
        Convert complex data structures to JSON-serializable format
        
        Args:
            data: Data to convert
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            # Convert dictionary values
            return {k: self._make_serializable(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            # Convert list items
            return [self._make_serializable(item) for item in data]
        
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            # Skip DataFrame/Series with a message
            return f"<{type(data).__name__} object: shape {data.shape if hasattr(data, 'shape') else 'unknown'}>"
        
        elif isinstance(data, (datetime, pd.Timestamp)):
            # Convert datetime to ISO format
            return data.isoformat()
        
        elif isinstance(data, (int, float, str, bool, type(None))):
            # These types are already serializable
            return data
        
        else:
            # Convert other types to string
            return f"<{type(data).__name__} object>" 