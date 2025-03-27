#!/usr/bin/env python3
"""
Credit Spreads Analysis Module (src/main_hooks/credit_spreads.py)
----------------------------------------------------------------
Identifies and evaluates options credit spread opportunities based on market trend and stock analyses.

Functions:
  - find_credit_spreads - Finds viable credit spread strategies for top-ranked stocks
  - parse_credit_spreads - Parses AI-generated text to extract credit spread details

Dependencies:
  - src.main_utilities.data_processor - For processing options data
  - src.gemini.hooks - For trade plan prompt generation
  - YFinance client - For options chain data
  - Gemini client - For AI analysis

Used by:
  - main.py for identifying credit spread trading opportunities
"""

import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def find_credit_spreads(yfinance_client, gemini_client, market_trend, stock_analyses, trade_plan_prompt_hook):
    """
    Find credit spread opportunities based on market trend and stock analyses

    Parameters:
    - yfinance_client: YFinance client for getting options data
    - gemini_client: Gemini client for AI analysis
    - market_trend: Market trend analysis
    - stock_analyses: Dictionary of stock analyses
    - trade_plan_prompt_hook: Function to generate trade plan prompt

    Returns:
    - List of credit spread opportunities
    """
    logger.info(
        f"Finding credit spread opportunities for {len(stock_analyses)} stocks")

    try:
        # Initialize credit spread opportunities
        credit_spreads = []

        # Skip if market trend suggests avoiding trades
        if market_trend.get("risk_adjustment", "") == "skip":
            logger.warning(
                "Market conditions suggest skipping trades, no credit spreads will be generated")
            return credit_spreads

        # Sort stocks by technical score in descending order
        sorted_stocks = sorted(
            [(sym, analysis) for sym, analysis in stock_analyses.items()
             if 'technical_score' in analysis],
            key=lambda x: x[1].get('technical_score', 0),
            reverse=True
        )

        # Only analyze top 5 stocks for credit spreads
        top_stocks = sorted_stocks[:5]

        # For each top stock, find credit spread opportunities
        for symbol, analysis in top_stocks:
            logger.info(f"Finding credit spread opportunities for {symbol}")

            try:
                # Skip if no options data
                if "options_summary" not in analysis:
                    logger.warning(
                        f"Skipping {symbol} for credit spreads due to missing options data")
                    continue

                # Get up-to-date options data
                options_data = yfinance_client.get_options_chain(symbol)
                if not options_data:
                    logger.warning(
                        f"No options data available for {symbol}, skipping")
                    continue

                # Process options data
                from src.main_utilities.data_processor import process_options_data
                processed_options = process_options_data(options_data)

                # Prepare context for trade plan
                trade_context = {
                    "symbol": symbol,
                    "stock_analysis": analysis,
                    "options_data": processed_options,
                    "market_trend": market_trend,
                    "current_price": analysis.get("raw_data", {}).get("current_price", 0),
                    "atr_percent": analysis.get("raw_data", {}).get("atr_percent", 0),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Generate prompt for credit spread analysis
                if hasattr(trade_plan_prompt_hook, "__name__") and trade_plan_prompt_hook.__name__ == "get_trade_plan_prompt_from_context":
                    # Already has the wrapper function
                    prompt = trade_plan_prompt_hook(trade_context)
                else:
                    # Need to use the imported wrapper function
                    from src.gemini.hooks import get_trade_plan_prompt_from_context
                    prompt = get_trade_plan_prompt_from_context(trade_context)

                # Get analysis from Gemini
                credit_spread_text = gemini_client.generate_text(
                    prompt, temperature=0.3)

                # Parse credit spread opportunities
                parsed_spreads = parse_credit_spreads(
                    credit_spread_text, symbol, analysis)

                # Add to results
                credit_spreads.extend(parsed_spreads)
                logger.info(
                    f"Found {len(parsed_spreads)} credit spread opportunities for {symbol}")

            except Exception as e:
                logger.error(f"Error finding credit spreads for {symbol}: {e}")
                continue

        # Sort by total score in descending order and return
        sorted_spreads = sorted(credit_spreads, key=lambda x: x.get(
            "total_score", 0), reverse=True)
        logger.info(
            f"Found a total of {len(sorted_spreads)} credit spread opportunities")
        return sorted_spreads

    except Exception as e:
        logger.error(f"Error finding credit spreads: {e}")
        return []


def parse_credit_spreads(spread_text, symbol, stock_analysis):
    """
    Parse credit spread opportunities from AI-generated text

    Parameters:
    - spread_text: AI-generated text with credit spread opportunities
    - symbol: Stock symbol
    - stock_analysis: Stock analysis data

    Returns:
    - List of credit spread opportunities
    """
    logger.debug(f"Parsing credit spread opportunities for {symbol}")

    credit_spreads = []

    try:
        # Split into sections by spread type
        sections = re.split(
            r'\n\s*(?:Option|OPTION|Spread|SPREAD|Trade|TRADE)\s*\d+\s*:', spread_text)
        if len(sections) <= 1:
            # Try another pattern
            sections = re.split(
                r'\n\s*(?:Option|OPTION|Spread|SPREAD|Trade|TRADE)\s*\d+\s*\n', spread_text)
            if len(sections) <= 1:
                # Try yet another pattern
                sections = spread_text.split("\n\n")

        # Remove first section if it's just an introduction
        if len(sections) > 1 and "credit spread" not in sections[0].lower() and "setup" not in sections[0].lower():
            sections = sections[1:]

        # Process each section
        for section in sections:
            if not section.strip():
                continue

            # Initialize spread data
            spread = {
                "symbol": symbol,
                "spread_type": "unknown",
                "direction": "neutral",
                "strikes": "",
                "expiration": "",
                "premium": 0,
                "max_loss": 0,
                "probability": 0,
                "quality_score": 0,
                "gamble_score": 0,
                "total_score": 0,
                "success_probability": 0,
                "technical_score": stock_analysis.get("technical_score", 0),
                "sentiment_score": stock_analysis.get("sentiment_score", 0),
                "market_alignment": stock_analysis.get("market_alignment", "neutral"),
                "full_analysis": section.strip()
            }

            # Determine spread type
            if "bull put" in section.lower() or "put credit" in section.lower():
                spread["spread_type"] = "bull put spread"
                spread["direction"] = "bullish"
            elif "bear call" in section.lower() or "call credit" in section.lower():
                spread["spread_type"] = "bear call spread"
                spread["direction"] = "bearish"

            # Extract strikes
            strikes_match = re.search(
                r'[Ss]trikes?:?\s*(\d+\.?\d*)\s*[-/]\s*(\d+\.?\d*)', section)
            if strikes_match:
                short_strike = strikes_match.group(1)
                long_strike = strikes_match.group(2)
                spread["strikes"] = f"{short_strike}/{long_strike}"

            # Extract expiration
            exp_match = re.search(
                r'[Ee]xpir(?:ation|y):?\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})', section)
            if exp_match:
                # Normalize date format to YYYY-MM-DD
                exp_date = exp_match.group(1)
                try:
                    if "-" in exp_date:
                        parts = exp_date.split("-")
                    else:
                        parts = exp_date.split("/")

                    if len(parts) == 3:
                        # Check if year is first or last
                        if len(parts[0]) == 4:  # YYYY-MM-DD
                            spread["expiration"] = exp_date
                        else:  # MM/DD/YYYY
                            spread["expiration"] = f"{parts[2]}-{parts[0]}-{parts[1]}"
                except:
                    spread["expiration"] = exp_date

            # Extract premium
            premium_match = re.search(
                r'[Pp]remium:?\s*\$?(\d+\.?\d*)', section)
            if premium_match:
                spread["premium"] = float(premium_match.group(1))

            # Extract max loss
            max_loss_match = re.search(
                r'[Mm]ax\s*[Ll]oss:?\s*\$?(\d+\.?\d*)', section)
            if max_loss_match:
                spread["max_loss"] = float(max_loss_match.group(1))

            # Extract probability
            prob_match = re.search(
                r'[Pp]robability(?:\s*[Oo]f\s*[Ss]uccess)?:?\s*(\d+\.?\d*)%?', section)
            if prob_match:
                probability = float(prob_match.group(1))
                if probability > 1 and probability <= 100:
                    spread["success_probability"] = probability
                elif probability <= 1:
                    spread["success_probability"] = probability * 100

            # Extract quality score
            quality_match = re.search(
                r'[Qq]uality\s*[Ss]core:?\s*(\d+\.?\d*)', section)
            if quality_match:
                spread["quality_score"] = float(quality_match.group(1))

            # Extract gamble score
            gamble_match = re.search(
                r'[Gg]amble\s*[Ss]core:?\s*(\d+\.?\d*)', section)
            if gamble_match:
                spread["gamble_score"] = float(gamble_match.group(1))

            # Calculate total score if not explicitly provided
            total_score_match = re.search(
                r'[Tt]otal\s*[Ss]core:?\s*(\d+\.?\d*)', section)
            if total_score_match:
                spread["total_score"] = float(total_score_match.group(1))
            else:
                # Calculate total score based on quality and inverse of gamble score
                if spread["quality_score"] > 0:
                    inverse_gamble = 100 - \
                        spread["gamble_score"] if spread["gamble_score"] > 0 else 80
                    spread["total_score"] = (
                        spread["quality_score"] + inverse_gamble) / 2

            # Add to results if it has strikes
            if spread["strikes"]:
                credit_spreads.append(spread)

        return credit_spreads

    except Exception as e:
        logger.error(f"Error parsing credit spreads for {symbol}: {e}")
        return []
