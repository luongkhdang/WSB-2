import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Import hooks
from src.finance_client.hooks.data_retrieval import (
    get_historical_data,
    get_market_data,
    get_options_chain,
    get_volatility_data
)

from src.finance_client.hooks.technical_indicators_hook import (
    enrich_with_technical_indicators,
    get_indicator_summary,
    get_trade_recommendations
)

from src.finance_client.hooks.options_analysis import (
    analyze_options_chain,
    analyze_options_sentiment
)

from src.finance_client.hooks.data_validation import (
    validate_historical_data,
    detect_data_anomalies
)

# Import formatters
from src.finance_client.utilities.data_formatter import (
    format_stock_data_for_analysis,
    format_market_data,
    format_spread_data
)

logger = logging.getLogger(__name__)

def perform_pretraining_analysis(symbols: List[str], 
                               lookback_days: int = 180, 
                               include_options: bool = True,
                               cache_dir: str = "./data-cache") -> Dict[str, Any]:
    """
    Perform comprehensive pretraining analysis on a list of symbols
    
    Args:
        symbols: List of ticker symbols to analyze
        lookback_days: Number of days of historical data to analyze
        include_options: Whether to include options analysis
        cache_dir: Directory for data cache
        
    Returns:
        Dictionary with complete analysis results
    """
    logger.info(f"Starting pretraining analysis for {len(symbols)} symbols with {lookback_days} days lookback")
    
    # Get dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Get market data for context
    market_data = get_market_data(cache_dir=cache_dir)
    formatted_market = format_market_data(market_data) if market_data else None
    
    # Get volatility data
    volatility_data = get_volatility_data(cache_dir=cache_dir)
    
    # Process each symbol
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}")
            
            # Get historical data (daily and intraday)
            daily_data = get_historical_data(symbol, start=start_date, end=end_date, interval="1d", cache_dir=cache_dir)
            intraday_data = None
            
            # For stocks with at least 20 days of data, get intraday data for last 7 days
            if daily_data is not None and len(daily_data) >= 20:
                intraday_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                intraday_data = get_historical_data(
                    symbol, 
                    start=intraday_start, 
                    end=end_date, 
                    interval="1h", 
                    cache_dir=cache_dir
                )
            
            # Skip if no data available
            if daily_data is None or len(daily_data) == 0:
                logger.warning(f"No data available for {symbol}")
                results[symbol] = {"status": "no_data"}
                continue
            
            # Enrich with technical indicators
            enriched_data = enrich_with_technical_indicators(daily_data, symbol)
            
            # Get technical summary
            technical_summary = get_indicator_summary(enriched_data, symbol)
            trade_recommendations = get_trade_recommendations(enriched_data, symbol)
            
            # Check for data anomalies
            anomalies = detect_data_anomalies(daily_data)
            
            # Process options data if requested
            options_analysis = None
            options_sentiment = None
            
            if include_options:
                options_chain = get_options_chain(symbol, cache_dir=cache_dir)
                
                if options_chain and 'calls' in options_chain and 'puts' in options_chain:
                    # Analyze options
                    options_analysis = analyze_options_chain(
                        options_chain,
                        price_history=daily_data,
                        volatility_data=volatility_data
                    )
                    
                    # Get options sentiment
                    if options_analysis and 'error' not in options_analysis:
                        options_sentiment = analyze_options_sentiment(options_analysis)
            
            # Format data for analysis
            formatted_data = format_stock_data_for_analysis(enriched_data, symbol, end_date)
            
            # Compile final results
            results[symbol] = {
                "status": "success",
                "daily_data_points": len(daily_data),
                "intraday_data_points": len(intraday_data) if intraday_data is not None else 0,
                "current_price": float(daily_data['Close'].iloc[-1]),
                "formatted_data": formatted_data,
                "technical_summary": technical_summary,
                "trade_recommendations": trade_recommendations,
                "data_anomalies": anomalies,
                "options_analysis": options_analysis,
                "options_sentiment": options_sentiment
            }
            
            # Add quality score if available
            if hasattr(daily_data, 'attrs') and 'quality_score' in daily_data.attrs:
                results[symbol]["data_quality_score"] = float(daily_data.attrs['quality_score'])
            
            logger.info(f"Completed analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results[symbol] = {"status": "error", "error_message": str(e)}
    
    # Add market context to results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "symbols_analyzed": len(symbols),
        "successful_analyses": sum(1 for s in results if results[s].get("status") == "success"),
        "market_data": formatted_market,
        "volatility_data": volatility_data,
        "symbol_results": results
    }
    
    logger.info(f"Completed pretraining analysis for {len(symbols)} symbols")
    return final_results

def generate_trading_scenarios(pretraining_results: Dict[str, Any], 
                              max_scenarios_per_symbol: int = 3) -> List[Dict[str, Any]]:
    """
    Generate trading scenarios based on pretraining analysis
    
    Args:
        pretraining_results: Results from pretraining analysis
        max_scenarios_per_symbol: Maximum number of scenarios per symbol
        
    Returns:
        List of trading scenarios
    """
    if not pretraining_results or "symbol_results" not in pretraining_results:
        logger.error("Invalid pretraining results")
        return []
    
    symbol_results = pretraining_results.get("symbol_results", {})
    market_data = pretraining_results.get("market_data", {})
    volatility_data = pretraining_results.get("volatility_data", {})
    
    # Determine overall market scenario
    market_scenario = "neutral"
    market_strength = 0
    
    if market_data and "market_trend" in market_data:
        market_scenario = market_data["market_trend"]
        market_strength = market_data.get("trend_strength", 50)
    
    # Adjust position sizing based on volatility
    position_sizing = 1.0
    volatility_note = ""
    
    if volatility_data and "risk_adjustment" in volatility_data:
        position_sizing = volatility_data["risk_adjustment"]
        volatility_note = volatility_data.get("adjustment_note", "")
    
    # Generate scenarios
    scenarios = []
    
    for symbol, result in symbol_results.items():
        if result.get("status") != "success":
            continue
        
        technical_summary = result.get("technical_summary", {})
        trade_recs = result.get("trade_recommendations", {})
        options_sentiment = result.get("options_sentiment", {})
        
        # Skip if no technical score
        if "technical_score" not in technical_summary:
            continue
        
        # Get key data points
        tech_score = technical_summary.get("technical_score", 0)
        current_price = result.get("current_price")
        
        # Skip symbols with weak signals
        if abs(tech_score) < 20:
            continue
        
        # Create base scenario
        base_scenario = {
            "symbol": symbol,
            "current_price": current_price,
            "technical_score": tech_score,
            "market_scenario": market_scenario,
            "volatility_adjustment": position_sizing,
            "volatility_note": volatility_note,
            "timestamp": datetime.now().isoformat(),
            "data_points": result.get("daily_data_points", 0)
        }
        
        # Add stock scenarios
        if "position" in trade_recs:
            position = trade_recs["position"]
            
            if position in ["long", "short"]:
                stock_scenario = base_scenario.copy()
                stock_scenario.update({
                    "scenario_type": "stock",
                    "position": position,
                    "entry_price": trade_recs.get("entry_price", current_price),
                    "stop_loss": trade_recs.get("stop_loss"),
                    "target_price": trade_recs.get("target_price"),
                    "key_indicators": trade_recs.get("key_indicators", {})
                })
                
                # Calculate risk/reward ratio
                if "stop_loss" in trade_recs and "target_price" in trade_recs:
                    stop_distance = abs(current_price - trade_recs["stop_loss"])
                    target_distance = abs(trade_recs["target_price"] - current_price)
                    
                    if stop_distance > 0:
                        stock_scenario["risk_reward_ratio"] = target_distance / stop_distance
                
                scenarios.append(stock_scenario)
        
        # Add options scenarios if available
        options_analysis = result.get("options_analysis", {})
        
        if options_analysis and "analysis" in options_analysis:
            analysis = options_analysis["analysis"]
            
            if "top_opportunities" in analysis:
                opportunities = analysis["top_opportunities"]
                
                # Add bullish scenario
                if opportunities.get("bullish") and tech_score > 0:
                    bull_scenario = base_scenario.copy()
                    bull_opp = opportunities["bullish"]
                    
                    bull_scenario.update({
                        "scenario_type": "options",
                        "strategy": bull_opp.get("strategy_type"),
                        "position": "bullish",
                        "options_data": bull_opp,
                        "sentiment_score": options_sentiment.get("sentiment_score", 0) if options_sentiment else 0
                    })
                    
                    scenarios.append(bull_scenario)
                
                # Add bearish scenario
                if opportunities.get("bearish") and tech_score < 0:
                    bear_scenario = base_scenario.copy()
                    bear_opp = opportunities["bearish"]
                    
                    bear_scenario.update({
                        "scenario_type": "options",
                        "strategy": bear_opp.get("strategy_type"),
                        "position": "bearish",
                        "options_data": bear_opp,
                        "sentiment_score": options_sentiment.get("sentiment_score", 0) if options_sentiment else 0
                    })
                    
                    scenarios.append(bear_scenario)
                
                # Add neutral scenario if volatility is high
                if opportunities.get("neutral") and volatility_data.get("price", 0) > 25:
                    neutral_scenario = base_scenario.copy()
                    neutral_opp = opportunities["neutral"]
                    
                    neutral_scenario.update({
                        "scenario_type": "options",
                        "strategy": neutral_opp.get("strategy_type"),
                        "position": "neutral",
                        "options_data": neutral_opp,
                        "sentiment_score": options_sentiment.get("sentiment_score", 0) if options_sentiment else 0
                    })
                    
                    scenarios.append(neutral_scenario)
    
    # Sort scenarios by absolute technical score (strongest signals first)
    scenarios.sort(key=lambda x: abs(x.get("technical_score", 0)), reverse=True)
    
    # Limit scenarios per symbol
    symbol_count = {}
    filtered_scenarios = []
    
    for scenario in scenarios:
        symbol = scenario["symbol"]
        symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
        
        if symbol_count[symbol] <= max_scenarios_per_symbol:
            filtered_scenarios.append(scenario)
    
    return filtered_scenarios

def rank_pretraining_symbols(pretraining_results: Dict[str, Any], 
                            min_technical_score: float = 30.0,
                            min_quality_score: float = 70.0) -> List[Dict[str, Any]]:
    """
    Rank symbols based on pretraining analysis results
    
    Args:
        pretraining_results: Results from pretraining analysis
        min_technical_score: Minimum absolute technical score to consider
        min_quality_score: Minimum data quality score to consider
        
    Returns:
        List of ranked symbols with scores
    """
    if not pretraining_results or "symbol_results" not in pretraining_results:
        logger.error("Invalid pretraining results")
        return []
    
    symbol_results = pretraining_results.get("symbol_results", {})
    
    # Build ranking list
    rankings = []
    
    for symbol, result in symbol_results.items():
        if result.get("status") != "success":
            continue
        
        # Get key scores
        technical_summary = result.get("technical_summary", {})
        data_quality = result.get("data_quality_score", 0)
        
        # Skip symbols with insufficient scores
        if (abs(technical_summary.get("technical_score", 0)) < min_technical_score or
            data_quality < min_quality_score):
            continue
        
        # Create ranking entry
        ranking = {
            "symbol": symbol,
            "technical_score": technical_summary.get("technical_score", 0),
            "data_quality_score": data_quality,
            "current_price": result.get("current_price"),
            "signal": technical_summary.get("indicators", {}).get("overall_signal", "neutral"),
            "data_points": result.get("daily_data_points", 0)
        }
        
        # Add options sentiment if available
        options_sentiment = result.get("options_sentiment", {})
        if options_sentiment and "sentiment_category" in options_sentiment:
            ranking["options_sentiment"] = options_sentiment["sentiment_category"]
            ranking["options_sentiment_score"] = options_sentiment.get("sentiment_score", 0)
        
        # Calculate composite score (technical strength + data quality bonus)
        tech_strength = abs(ranking["technical_score"])
        quality_bonus = (data_quality - min_quality_score) / (100 - min_quality_score) * 10
        
        # Add options alignment bonus if sentiment matches technical direction
        options_bonus = 0
        if "options_sentiment_score" in ranking:
            # Check if options sentiment aligns with technical signal
            if (ranking["technical_score"] > 0 and ranking["options_sentiment_score"] > 0) or \
               (ranking["technical_score"] < 0 and ranking["options_sentiment_score"] < 0):
                # Add bonus proportional to alignment strength
                alignment = min(abs(ranking["options_sentiment_score"]) / 50, 1.0)
                options_bonus = 15 * alignment
        
        # Calculate final score
        ranking["composite_score"] = tech_strength + quality_bonus + options_bonus
        
        rankings.append(ranking)
    
    # Sort by composite score
    rankings.sort(key=lambda x: x["composite_score"], reverse=True)
    
    return rankings