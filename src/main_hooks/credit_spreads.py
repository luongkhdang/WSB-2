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

        # Analyze up to 10 stocks for credit spreads (increased from 5)
        top_stocks = sorted_stocks[:10]

        logger.info(
            f"Analyzing top {len(top_stocks)} stocks for credit spreads: {[sym for sym, _ in top_stocks]}")

        # For each top stock, find credit spread opportunities
        for symbol, analysis in top_stocks:
            logger.info(f"Finding credit spread opportunities for {symbol}")

            try:
                # Skip if no options data but add diagnostic info
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

                # Log options data availability
                logger.info(f"Options data for {symbol}: {len(processed_options.get('top_calls', []))} calls, " +
                            f"{len(processed_options.get('top_puts', []))} puts")

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

                # Log parsing results
                logger.info(
                    f"Parsed {len(parsed_spreads)} credit spread opportunities for {symbol}")

                # If no spreads were found, try to generate a synthetic recommendation
                if len(parsed_spreads) == 0:
                    logger.warning(
                        f"No viable credit spreads found in AI response for {symbol}, attempting to generate fallback spread")
                    # Generate synthetic spread based on market and stock trend
                    fallback_spread = generate_fallback_spread(
                        symbol, analysis, market_trend, processed_options)
                    if fallback_spread:
                        logger.info(
                            f"Generated fallback credit spread for {symbol}")
                        parsed_spreads.append(fallback_spread)

                # Add to results
                credit_spreads.extend(parsed_spreads)
                logger.info(
                    f"Found {len(parsed_spreads)} credit spread opportunities for {symbol}")

            except Exception as e:
                logger.error(f"Error finding credit spreads for {symbol}: {e}")
                logger.exception(e)
                continue

        # Sort by total score in descending order and return
        sorted_spreads = sorted(credit_spreads, key=lambda x: x.get(
            "total_score", 0), reverse=True)
        logger.info(
            f"Found a total of {len(sorted_spreads)} credit spread opportunities")
        return sorted_spreads

    except Exception as e:
        logger.error(f"Error finding credit spreads: {e}")
        logger.exception(e)
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
    logger.debug(f"AI text length: {len(spread_text)} characters")

    credit_spreads = []

    try:
        # Split into sections by spread type - expanded pattern matching
        section_patterns = [
            r'\n\s*(?:Option|OPTION|Spread|SPREAD|Trade|TRADE)\s*\d+\s*:',
            r'\n\s*(?:Option|OPTION|Spread|SPREAD|Trade|TRADE)\s*\d+\s*\n',
            r'\n\s*(?:CREDIT\s+SPREAD|Credit\s+Spread|credit\s+spread)\s*\d*\s*:',
            r'\n\s*(?:BULL\s+PUT|Bear\s+Call|Bull\s+Put|BEAR\s+CALL)\s*\d*\s*:',
            r'\n\s*(?:\d+\.\s+)(?:CREDIT\s+SPREAD|Credit\s+Spread|credit\s+spread)',
            r'\n\s*Recommendation\s*\d*\s*:'
        ]

        # Try each pattern until we find one that works
        sections = []
        for pattern in section_patterns:
            sections = re.split(pattern, spread_text)
            if len(sections) > 1:
                break

        # If no sections found, try a simpler approach
        if len(sections) <= 1:
            sections = spread_text.split("\n\n")

        # Remove first section if it's just an introduction
        intro_keywords = ["credit spread", "setup", "recommendation",
                          "analysis", "quality matrix", "gamble matrix"]
        if len(sections) > 1 and not any(keyword in sections[0].lower() for keyword in intro_keywords):
            sections = sections[1:]

        logger.debug(f"Found {len(sections)} potential credit spread sections")

        # Process each section
        for section in sections:
            if not section.strip():
                continue

            # Initialize spread data with more default values
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

            # Determine spread type - improved matching
            spread_type_patterns = [
                (r'bull\s+put', "bull put spread", "bullish"),
                (r'put\s+credit', "bull put spread", "bullish"),
                (r'bear\s+call', "bear call spread", "bearish"),
                (r'call\s+credit', "bear call spread", "bearish"),
                (r'bullish', "bull put spread", "bullish"),
                (r'bearish', "bear call spread", "bearish")
            ]

            for pattern, spread_type, direction in spread_type_patterns:
                if re.search(pattern, section.lower()):
                    spread["spread_type"] = spread_type
                    spread["direction"] = direction
                    break

            # Extract strikes - improved patterns
            strikes_patterns = [
                r'[Ss]trikes?:?\s*(\d+\.?\d*)\s*[-/]\s*(\d+\.?\d*)',
                r'[Ss]hort\s+[Ss]trike:?\s*(\d+\.?\d*).*?[Ll]ong\s+[Ss]trike:?\s*(\d+\.?\d*)',
                r'[Ss]ell\s+(\d+\.?\d*)\s*[Pp]ut.*?[Bb]uy\s+(\d+\.?\d*)\s*[Pp]ut',
                r'[Ss]ell\s+(\d+\.?\d*)\s*[Cc]all.*?[Bb]uy\s+(\d+\.?\d*)\s*[Cc]all',
                r'(\d+\.?\d*)/(\d+\.?\d*)\s+(?:put|call|spread)',
            ]

            for pattern in strikes_patterns:
                strikes_match = re.search(
                    pattern, section, re.IGNORECASE | re.DOTALL)
                if strikes_match:
                    short_strike = strikes_match.group(1)
                    long_strike = strikes_match.group(2)
                    spread["strikes"] = f"{short_strike}/{long_strike}"
                    break

            # Extract expiration - expanded patterns
            exp_patterns = [
                r'[Ee]xpir(?:ation|y):?\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})',
                r'[Dd][Tt][Ee]:?\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})',
                r'[Dd]ate:?\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})',
                r'(\d{1,2}/\d{1,2}/\d{2,4})\s*expir',
                r'expiring\s+(?:on\s+)?(\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?)',
                r'(\d{1,2}[-/]\d{1,2})\s+[Ee]xpiry'
            ]

            for pattern in exp_patterns:
                exp_match = re.search(pattern, section)
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
                        elif len(parts) == 2:
                            # Only MM/DD, assume current year
                            current_year = datetime.now().year
                            spread["expiration"] = f"{current_year}-{parts[0]}-{parts[1]}"
                    except Exception as e:
                        logger.warning(
                            f"Error parsing expiration date '{exp_date}': {e}")
                        spread["expiration"] = exp_date
                    break

            # Extract premium - expanded patterns
            premium_patterns = [
                r'[Pp]remium:?\s*\$?(\d+\.?\d*)',
                r'[Cc]redit:?\s*\$?(\d+\.?\d*)',
                r'[Rr]eceive:?\s*\$?(\d+\.?\d*)',
                r'[Cc]ollect:?\s*\$?(\d+\.?\d*)',
                r'[Pp]rice:?\s*\$?(\d+\.?\d*)'
            ]

            for pattern in premium_patterns:
                premium_match = re.search(pattern, section)
                if premium_match:
                    spread["premium"] = float(premium_match.group(1))
                    break

            # Extract max loss - expanded patterns
            max_loss_patterns = [
                r'[Mm]ax\s*[Ll]oss:?\s*\$?(\d+\.?\d*)',
                r'[Mm]ax\s*[Rr]isk:?\s*\$?(\d+\.?\d*)',
                r'[Mm]aximum\s*[Ll]oss:?\s*\$?(\d+\.?\d*)',
                r'[Mm]aximum\s*[Rr]isk:?\s*\$?(\d+\.?\d*)',
                r'[Rr]isk:?\s*\$?(\d+\.?\d*)',
            ]

            for pattern in max_loss_patterns:
                max_loss_match = re.search(pattern, section)
                if max_loss_match:
                    spread["max_loss"] = float(max_loss_match.group(1))
                    break

            # Extract probability - expanded patterns
            prob_patterns = [
                r'[Pp]robability(?:\s*[Oo]f\s*[Ss]uccess)?:?\s*(\d+\.?\d*)%?',
                r'[Ss]uccess\s*[Pp]robability:?\s*(\d+\.?\d*)%?',
                r'[Cc]hance\s*[Oo]f\s*[Ss]uccess:?\s*(\d+\.?\d*)%?',
                r'[Pp]robability:?\s*(\d+\.?\d*)%?',
                r'[Pp][Oo][Pp]:?\s*(\d+\.?\d*)%?'
            ]

            for pattern in prob_patterns:
                prob_match = re.search(pattern, section)
                if prob_match:
                    probability = float(prob_match.group(1))
                    if probability > 1 and probability <= 100:
                        spread["success_probability"] = probability
                    elif probability <= 1:
                        spread["success_probability"] = probability * 100
                    break

            # Extract quality score - expanded patterns
            quality_patterns = [
                r'[Qq]uality\s*[Ss]core:?\s*(\d+\.?\d*)',
                r'[Qq]uality:?\s*(\d+\.?\d*)',
                r'[Ss]core:?\s*(\d+\.?\d*)'
            ]

            for pattern in quality_patterns:
                quality_match = re.search(pattern, section)
                if quality_match:
                    spread["quality_score"] = float(quality_match.group(1))
                    break

            # Extract gamble score
            gamble_patterns = [
                r'[Gg]amble\s*[Ss]core:?\s*(\d+\.?\d*)',
                r'[Gg]amble:?\s*(\d+\.?\d*)',
                r'[Rr]isk\s*[Ss]core:?\s*(\d+\.?\d*)'
            ]

            for pattern in gamble_patterns:
                gamble_match = re.search(pattern, section)
                if gamble_match:
                    spread["gamble_score"] = float(gamble_match.group(1))
                    break

            # Extract total score
            total_score_patterns = [
                r'[Tt]otal\s*[Ss]core:?\s*(\d+\.?\d*)',
                r'[Oo]verall\s*[Ss]core:?\s*(\d+\.?\d*)',
                r'[Ff]inal\s*[Ss]core:?\s*(\d+\.?\d*)'
            ]

            total_score_match = None
            for pattern in total_score_patterns:
                total_score_match = re.search(pattern, section)
                if total_score_match:
                    spread["total_score"] = float(total_score_match.group(1))
                    break

            # Calculate total score if not explicitly provided
            if not total_score_match:
                # Calculate total score based on quality and inverse of gamble score
                if spread["quality_score"] > 0:
                    inverse_gamble = 100 - \
                        spread["gamble_score"] if spread["gamble_score"] > 0 else 80
                    spread["total_score"] = (
                        spread["quality_score"] + inverse_gamble) / 2
                # If we have neither quality score nor gamble score, make an estimate
                elif spread["success_probability"] > 0:
                    # Use success probability as a base score
                    spread["total_score"] = spread["success_probability"]

            # RELAXED CRITERIA: Accept spread if it has at least strikes OR (direction AND expiration)
            valid_spread = (
                spread["strikes"] or
                (spread["direction"] != "neutral" and spread["expiration"])
            )

            # If we're missing critical info but have some basic data, try to fill in gaps
            if valid_spread and not spread["strikes"] and spread["direction"] != "neutral":
                # Try to infer some reasonable strike values based on current price
                current_price = stock_analysis.get(
                    "raw_data", {}).get("current_price", 0)
                if current_price > 0:
                    if spread["direction"] == "bullish":
                        # Bull put - strikes below current price
                        short_strike = round(
                            current_price * 0.95, 1)  # 5% below
                        long_strike = round(
                            current_price * 0.90, 1)   # 10% below
                        spread["strikes"] = f"{short_strike}/{long_strike}"
                    else:
                        # Bear call - strikes above current price
                        short_strike = round(
                            current_price * 1.05, 1)  # 5% above
                        long_strike = round(
                            current_price * 1.10, 1)   # 10% above
                        spread["strikes"] = f"{short_strike}/{long_strike}"
                    logger.info(
                        f"Inferred strikes for {symbol}: {spread['strikes']}")

            # If expiration is missing but we have other data, use a default expiration
            if valid_spread and not spread["expiration"]:
                # Use a date approximately 2 weeks in the future
                future_date = datetime.now() + timedelta(days=14)
                spread["expiration"] = future_date.strftime("%Y-%m-%d")
                logger.info(
                    f"Using default expiration for {symbol}: {spread['expiration']}")

            # Ensure reasonable total score
            if valid_spread and spread["total_score"] <= 0:
                # Assign a modest default score
                spread["total_score"] = 65
                logger.info(
                    f"Assigned default total score for {symbol}: {spread['total_score']}")

            # Add to results if valid
            if valid_spread:
                credit_spreads.append(spread)
                logger.debug(
                    f"Valid spread found for {symbol}: {spread['spread_type']} with strikes {spread['strikes']}")
            else:
                logger.debug(
                    f"Rejected invalid spread for {symbol} - missing required data")

        return credit_spreads

    except Exception as e:
        logger.error(f"Error parsing credit spreads for {symbol}: {e}")
        logger.exception(e)
        return []


def generate_fallback_spread(symbol, stock_analysis, market_trend, options_data):
    """
    Generate a synthetic credit spread opportunity when AI doesn't provide valid recommendations

    Parameters:
    - symbol: Stock symbol
    - stock_analysis: Stock analysis data
    - market_trend: Market trend analysis
    - options_data: Options data for the stock

    Returns:
    - Synthetic credit spread opportunity or None if not possible
    """
    logger.info(f"Generating fallback credit spread for {symbol}")

    try:
        # Get current price
        current_price = stock_analysis.get(
            "raw_data", {}).get("current_price", 0)
        if current_price <= 0:
            logger.warning(
                f"Cannot generate fallback spread for {symbol}: Invalid current price")
            return None

        # Determine direction based on stock trend and market trend
        stock_trend = stock_analysis.get("trend", "neutral")
        market_alignment = stock_analysis.get("market_alignment", "neutral")
        market_trend_value = market_trend.get("trend", "neutral")

        # Determine direction - prefer stock trend if it's not neutral
        if stock_trend != "neutral":
            direction = stock_trend
        # Otherwise use market trend
        elif market_trend_value != "neutral":
            direction = market_trend_value
        else:
            # If both are neutral, use technical signals
            if stock_analysis.get("technical_score", 0) > 50:
                direction = "bullish"
            else:
                direction = "bearish"

        # Choose spread type based on direction
        spread_type = "bull put spread" if direction == "bullish" else "bear call spread"

        # Calculate strikes based on ATR and current price
        atr_percent = stock_analysis.get(
            "raw_data", {}).get("atr_percent", 2.0)
        if atr_percent <= 0:
            atr_percent = 2.0  # Default if not available

        # Calculate strikes with a buffer of 1x ATR
        if direction == "bullish":
            # Bull put spread - strikes below current price
            short_strike = round(current_price * (1 - atr_percent/100), 1)
            # 5% below short strike
            long_strike = round(short_strike * 0.95, 1)
        else:
            # Bear call spread - strikes above current price
            short_strike = round(current_price * (1 + atr_percent/100), 1)
            # 5% above short strike
            long_strike = round(short_strike * 1.05, 1)

        # Set expiration to 14 days in the future
        expiration_date = (datetime.now() + timedelta(days=14)
                           ).strftime("%Y-%m-%d")

        # Calculate premium (rough estimate)
        width = abs(short_strike - long_strike)
        premium = round(width * 0.1, 2)  # Roughly 10% of width

        # Create the synthetic spread
        synthetic_spread = {
            "symbol": symbol,
            "spread_type": spread_type,
            "direction": direction,
            "strikes": f"{short_strike}/{long_strike}",
            "expiration": expiration_date,
            "premium": premium,
            "max_loss": round(width - premium, 2),
            "probability": 70,  # Conservative estimate
            "quality_score": 70,  # Moderate quality score
            "gamble_score": 30,  # Low gamble score
            "total_score": 70,  # Reasonable total score
            "success_probability": 70,
            "technical_score": stock_analysis.get("technical_score", 0),
            "sentiment_score": stock_analysis.get("sentiment_score", 0),
            "market_alignment": market_alignment,
            "full_analysis": f"Fallback {spread_type.upper()} RECOMMENDATION\n\nStrike: {short_strike}/{long_strike}\nExpiration: {expiration_date}\nPremium: ${premium}\nMax Loss: ${round(width - premium, 2)}\nProbability: 70%\nQuality Score: 70\nGamble Score: 30\n\nThis is a fallback recommendation based on technical analysis and current market conditions.",
            "is_fallback": True  # Mark as a fallback recommendation
        }

        logger.info(
            f"Generated fallback {spread_type} for {symbol}: {short_strike}/{long_strike} expiring {expiration_date}")
        return synthetic_spread

    except Exception as e:
        logger.error(f"Error generating fallback spread for {symbol}: {e}")
        logger.exception(e)
        return None
