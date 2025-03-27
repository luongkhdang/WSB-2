"""
Enhanced credit spread focused pretraining prompts for Gemini API.

This module provides specialized prompts for the pretraining process that:
1. Focus specifically on credit spread parameters and setups
2. Emphasize volatility, support/resistance levels, and strike selection
3. Follow the strict 6-step process with proper timeframes
4. Include robust reflection and learning mechanisms
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def format_data_for_credit_spreads(data: Dict[str, Any]) -> str:
    """Format stock data with emphasis on credit spread relevant metrics."""
    if not data:
        return "No stock data available."

    ticker = data.get('ticker', 'Unknown')
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Extract critical credit spread metrics
    current_price = data.get('current_price', 0)
    implied_volatility = data.get('implied_volatility', 'Unknown')
    historical_volatility = data.get('historical_volatility', 'Unknown')
    atr = data.get('atr', 'Unknown')
    atr_percent = data.get('atr_percent', 'Unknown')

    # Core metrics section
    core_metrics = f"""
TICKER: {ticker}
DATE: {date}
PRICE: {current_price}
IMPLIED VOLATILITY: {implied_volatility}
HISTORICAL VOLATILITY: {historical_volatility}
ATR: {atr}
ATR%: {atr_percent}
    """

    # Technical indicators section
    tech_indicators = []
    for key in ['rsi', 'macd', 'ema_9', 'ema_21', 'sma_50', 'sma_200', 'bollinger_bands']:
        if key in data:
            tech_indicators.append(f"{key.upper()}: {data[key]}")

    # Support/Resistance section
    support_resistance = []
    if 'support_levels' in data and data['support_levels']:
        support_resistance.append(
            f"SUPPORT LEVELS: {', '.join(map(str, data['support_levels']))}")
    if 'resistance_levels' in data and data['resistance_levels']:
        support_resistance.append(
            f"RESISTANCE LEVELS: {', '.join(map(str, data['resistance_levels']))}")

    # Format multi-timeframe data if available
    timeframe_section = ""
    if 'intraday_data' in data and data['intraday_data']:
        timeframe_section = "\nINTRADAY DATA (15m):\n"
        for key, value in data['intraday_data'].items():
            timeframe_section += f"{key}: {value}\n"

    weekly_section = ""
    if 'weekly_data' in data and data['weekly_data']:
        weekly_section = "\nWEEKLY DATA:\n"
        for key, value in data['weekly_data'].items():
            weekly_section += f"{key}: {value}\n"

    # Add quality warning if present
    quality_warning = f"\nQUALITY WARNING: {data.get('quality_warning')}" if 'quality_warning' in data else ""

    return f"{core_metrics}\nTECHNICAL INDICATORS:\n{chr(10).join(tech_indicators)}\n\nSUPPORT/RESISTANCE:\n{chr(10).join(support_resistance)}{timeframe_section}{weekly_section}{quality_warning}"


def format_credit_spread_market_context(context: Dict[str, Any]) -> str:
    """Format market context with focus on options and volatility metrics."""
    if not context:
        return "No market context available."

    # Extract core market information
    spy_trend = context.get('spy_trend', 'neutral')
    market_trend_score = context.get(
        'market_trend_score', context.get('trend_score', 0))
    vix = context.get('vix', 'Unknown')
    vix_trend = context.get('vix_trend', 'Unknown')

    # Format options-specific data
    options_info = ""
    if 'options_analysis' in context:
        options = context['options_analysis']
        options_info = f"""
OPTIONS CONTEXT:
Direction: {options.get('direction', 'neutral')}
Confidence: {options.get('confidence', 'low')}
IV Rank: {options.get('iv_rank', 'Unknown')}
IV Percentile: {options.get('iv_percentile', 'Unknown')}
Skew: {options.get('skew', 'Unknown')}
Put/Call Ratio: {options.get('put_call_ratio', 'Unknown')}
"""

    # Format sector information
    sector_info = ""
    if 'sector_performance' in context:
        sectors = []
        for sector, perf in context['sector_performance'].items():
            if isinstance(perf, (int, float)):
                sectors.append(f"{sector}: {perf:.2f}%")
            else:
                sectors.append(f"{sector}: {perf}")
        sector_info = "\nSECTOR PERFORMANCE:\n" + "\n".join(sectors)

    return f"""
MARKET CONTEXT:
SPY TREND: {spy_trend}
MARKET TREND SCORE: {market_trend_score}
VIX: {vix}
VIX TREND: {vix_trend}
{options_info}{sector_info}
"""


def format_credit_spread_memory(memory: Dict[str, Any]) -> str:
    """Format memory context with emphasis on spread-relevant patterns and levels."""
    if not memory:
        return "No memory context available."

    # Format key levels (critical for credit spreads)
    key_levels = []
    if 'key_levels' in memory and memory['key_levels']:
        for level_type, levels in memory['key_levels'].items():
            if levels:
                key_levels.append(
                    f"{level_type}: {', '.join(map(str, levels))}")

    # Format volatility history
    volatility_history = []
    if 'volatility_history' in memory and memory['volatility_history']:
        for entry in memory['volatility_history']:
            volatility_history.append(
                f"- {entry.get('date', 'Unknown')}: IV {entry.get('iv', 'Unknown')}, HV {entry.get('hv', 'Unknown')}")

    # Format pattern reliability
    reliability = []
    if 'pattern_reliability' in memory and memory['pattern_reliability']:
        for pattern_type, stats in memory['pattern_reliability'].items():
            if stats.get('total', 0) > 0:
                accuracy = stats.get('accuracy', 0)
                reliability.append(
                    f"{pattern_type}: {accuracy:.1f}% accuracy ({stats.get('correct', 0)}/{stats.get('total', 0)})")

    # Format spread performance
    spread_performance = []
    if 'spread_performance' in memory and memory['spread_performance']:
        for spread_type, stats in memory['spread_performance'].items():
            if stats.get('total', 0) > 0:
                win_rate = stats.get('win_rate', 0)
                spread_performance.append(
                    f"{spread_type}: {win_rate:.1f}% win rate ({stats.get('wins', 0)}/{stats.get('total', 0)})")

    return f"""
KEY LEVELS (Critical for Strike Selection):
{chr(10).join(key_levels) if key_levels else "No key levels identified yet."}

VOLATILITY HISTORY:
{chr(10).join(volatility_history) if volatility_history else "No volatility history recorded yet."}

PATTERN RELIABILITY:
{chr(10).join(reliability) if reliability else "No pattern reliability data yet."}

SPREAD PERFORMANCE:
{chr(10).join(spread_performance) if spread_performance else "No spread performance data yet."}
"""


def get_step1_prompt(
    ticker: str,
    stock_data: Dict[str, Any],
    market_context: Dict[str, Any],
    memory_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate Step 1 prompt that analyzes 1mo/1d data for long-term context.

    This is the first step in the 6-step process that establishes baseline
    trend, volatility, and key levels from 1-month daily data.
    """
    date_str = stock_data.get('date', datetime.now().strftime('%Y-%m-%d'))

    prompt = f"""
STEP 1 - PRETRAINING MODE: You are analyzing 1-month daily historical data for {ticker} as of {date_str}.
You are establishing the baseline context that will be used for intraday analysis in later steps.

FOCUS: As an expert in credit spread trading, your analysis must focus on parameters critical for
spread setup and strike selection (volatility, trend strength, key levels, etc.)

MARKET DATA:
{format_data_for_credit_spreads(stock_data)}

MARKET CONTEXT:
{format_credit_spread_market_context(market_context)}

MEMORY CONTEXT:
{format_credit_spread_memory(memory_context) if memory_context else "No previous patterns identified."}

CRITICAL TREND INSTRUCTION:
You MUST avoid excessive "neutral" assessments. Market trends are directional approximately 95% of the time.
Only classify as "neutral" when there is TRULY no directional bias (approximately 5% of cases).
- Bullish: When MACD > 0.5, RSI > 60, price above key EMAs, or clear positive momentum.
- Bearish: When MACD < -0.5, RSI < 40, price below key EMAs, or clear negative momentum.
- Neutral: ONLY when MACD is tightly range-bound (-0.1 to 0.1), RSI is 45-55, and price is tightly consolidating.

Provide a decisive analysis with the following EXACT sections:

1. LONG-TERM TREND ANALYSIS:
   - Daily trend direction and strength
   - Key moving averages and their alignment
   - Major price structure (higher highs/lows, consolidation, etc.)

2. VOLATILITY ASSESSMENT:
   - Historical and implied volatility analysis
   - ATR and percentage of price
   - Volatility trend (expanding/contracting)
   - Implication for option premium pricing

3. KEY LEVELS IDENTIFICATION:
   - Major support/resistance levels
   - Volume profile nodes
   - Recent range boundaries
   - Gap areas
   - These levels will be critical for strike selection

4. CREDIT SPREAD CONTEXT:
   - Ideal spread types given current conditions (bull put, bear call, iron condor)
   - Optimal DTE range based on volatility environment
   - Strike selection guidelines based on key levels
   - Target premium and risk parameters

5. NEXT DAY PREDICTION:
   Use this exact format for parsing:
   ```
   PREDICTION OUTPUT:
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   timeframe: [next_day]
   confidence: [60-95]
   key_levels: [list of critical price levels]
   invalidation: [condition that would invalidate this prediction]
   ```

You MUST be decisive with a confidence level between 60-95%. This forecast will be verified against 
actual outcomes, so you will be held accountable for accuracy.

Format your response with clear section headers and detailed analysis under each one.
"""
    return prompt


def get_step2_to_6_prompt(
    step_number: int,
    ticker: str,
    date_str: str,
    stock_data: Dict[str, Any],
    market_context: Dict[str, Any],
    memory_context: Dict[str, Any],
    previous_prediction: Optional[Dict[str, Any]] = None,
    actual_outcome: Optional[Dict[str, Any]] = None,
    weighted_strategies: Optional[List[str]] = None,
    strategy_reasoning: Optional[str] = None
) -> str:
    """
    Generate prompts for Steps 2-6 that analyze 15-minute intraday data.

    These steps follow the first step and focus on intraday patterns while
    building upon the context from previous steps.

    Parameters:
    - step_number: The step number (2-6)
    - ticker: Stock symbol
    - date_str: Current date string
    - stock_data: Dict containing stock data with intraday (15m) data
    - market_context: Dict with market context
    - memory_context: Dict with learning memory from previous steps
    - previous_prediction: The prediction from the previous step
    - actual_outcome: The actual outcome for the previous prediction
    - weighted_strategies: List of strategies with their weights based on historical performance
    - strategy_reasoning: Explanation of why these strategies were recommended
    """
    # Determine if this is a reflection prompt (has previous prediction and actual outcome)
    is_reflection = previous_prediction is not None and actual_outcome is not None
    reflection_section = ""

    if is_reflection:
        # Calculate accuracy metrics
        predicted_direction = previous_prediction.get('direction', 'neutral')
        actual_direction = "bullish" if actual_outcome.get(
            'price_change_percent', 0) > 0 else "bearish" if actual_outcome.get('price_change_percent', 0) < 0 else "neutral"
        direction_correct = predicted_direction == actual_direction

        predicted_magnitude = previous_prediction.get('magnitude', 0)
        actual_magnitude = abs(actual_outcome.get('price_change_percent', 0))
        magnitude_error = abs(predicted_magnitude - actual_magnitude)

        reflection_section = f"""
REFLECTION SECTION:
Previous Prediction (Step {step_number-1}):
- Direction: {predicted_direction}
- Magnitude: {predicted_magnitude}%
- Confidence: {previous_prediction.get('confidence', 0)}%

Actual Outcome:
- Direction: {actual_direction}
- Magnitude: {actual_outcome.get('price_change_percent', 0)}%
- Price Change: ${actual_outcome.get('price_change', 0):.2f}
- Open: ${actual_outcome.get('open', 0):.2f}
- Close: ${actual_outcome.get('close', 0):.2f}
- High: ${actual_outcome.get('high', 0):.2f}
- Low: ${actual_outcome.get('low', 0):.2f}

Accuracy Assessment:
- Direction Correct: {"Yes" if direction_correct else "No"}
- Magnitude Error: {magnitude_error:.2f}%
- Within Expected Range: {"Yes" if magnitude_error <= predicted_magnitude * 0.5 else "No"}

REFLECTION INSTRUCTIONS:
1. Evaluate why your previous prediction was {"correct" if direction_correct else "incorrect"}
2. Identify which indicators were most reliable
3. Note which patterns or signals you should have weighted more/less
4. Suggest specific adjustments to your analysis approach
"""

    # Extract the secret for the fixed reference section
    secret_section = ""
    if "secret" in memory_context:
        secret = memory_context["secret"]
        secret_section = f"""
SECRET (Fixed Reference - Do Not Alter): 
- Baseline Trend: {secret['baseline_trend']} ({secret['trend_confidence']}% confidence)
- Volatility Anchor: {secret['volatility_anchor']}% (30-day ATR)
- Core Levels: Support={secret['core_levels']['support']}, Resistance={secret['core_levels']['resistance']}

Use this secret as your foundation - adjustments must align with this stable reference point, not contradict it.
Deviate from the secret ONLY with 70%+ confidence based on significant new information.
"""

    # Add strategy weighting section if provided
    strategy_section = ""
    if weighted_strategies and strategy_reasoning:
        strategy_section = f"""
STRATEGY PERFORMANCE AND WEIGHTING:
{strategy_reasoning}

RECOMMENDED STRATEGY PRIORITIES:
{', '.join(weighted_strategies)}

Focus on the highest-weighted strategies based on historical performance, but consider current market conditions.
"""

    prompt = f"""
STEP {step_number} - 15-MINUTE INTRADAY PRETRAINING: You are analyzing {ticker} for {date_str} using 15-minute intraday data.
{"You must first reflect on your previous prediction accuracy, then analyze the current data." if is_reflection else "This is part of the 6-step pretraining process for credit spread trading."}

{secret_section}

{reflection_section}

{strategy_section}

MARKET DATA:
{format_data_for_credit_spreads(stock_data)}

MARKET CONTEXT:
{format_credit_spread_market_context(market_context)}

MEMORY CONTEXT:
{format_credit_spread_memory(memory_context)}

CRITICAL TREND INSTRUCTION:
You MUST avoid excessive "neutral" assessments. Market trends are directional approximately 95% of the time.
Only classify as "neutral" when there is TRULY no directional bias (approximately 5% of cases).
- Bullish: When MACD > 0.5, RSI > 60, price above key EMAs, or clear positive momentum.
- Bearish: When MACD < -0.5, RSI < 40, price below key EMAs, or clear negative momentum.
- Neutral: ONLY when MACD is tightly range-bound (-0.1 to 0.1), RSI is 45-55, and price is tightly consolidating.

Provide a comprehensive intraday analysis with these EXACT sections:

1. {"REFLECTION AND LEARNING ADJUSTMENT:" if is_reflection else "PRIOR CONTEXT INTEGRATION:"}
   {("- Evaluate the accuracy of your previous prediction" + chr(10) +
     "- Identify which technical signals were reliable/unreliable" + chr(10) +
     "- Specify weight adjustments for future analysis" + chr(10) +
     "- Update pattern reliability scores") if is_reflection else
        ("- How today's data connects to the longer-term view" + chr(10) +
         "- Confirmation or contradiction of prior analysis" + chr(10) +
         "- Changes in key level relevance")}

2. INTRADAY PATTERN ANALYSIS:
   - 15-minute chart patterns and formations
   - Volume profile and significant transactions
   - Intraday pivots and reversals
   - Momentum and trend characteristics

3. VOLATILITY AND OPTIONS FOCUS:
   - Intraday volatility assessment
   - Implications for credit spread pricing
   - Ideal entry timing based on volatility patterns
   - Potential for premium expansion/contraction

4. KEY LEVEL INTERACTION:
   - How price is interacting with previously identified levels
   - New intraday support/resistance levels
   - Critical zones for strike selection

5. CREDIT SPREAD OPPORTUNITY:
   - Specific credit spread setup identification
   - CONSIDER DIRECTIONAL STRATEGIES based on trend, not just credit spreads
   - Ideal strike selection with rationale
   - Entry timing recommendation based on intraday patterns
   - Probability of success estimate
   - Risk/reward profile with specific targets and stops

6. PREDICTION:
   Use this exact format for parsing:
   ```
   PREDICTION OUTPUT:
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   timeframe: [next_day]
   confidence: [60-95]
   specific_levels: [list of critical price levels for tomorrow]
   spread_recommendation: [bull put spread/bear call spread/iron condor/long call/long put/call debit spread/put debit spread]
   strike_recommendation: [specific strike prices with rationale]
   risk_reward_ratio: [1:X - expressed as a ratio where X is the reward multiple of risk]
   stop_loss: [percentage or price level]
   profit_target: [percentage or price level]
   invalidation: [condition that would invalidate this prediction]
   ```

7. MEMORY UPDATE:
   Use this exact format for parsing:
   ```
   MEMORY UPDATE:
   reliability_update: {{
        "pattern_type": {{
            "accuracy": [percentage], 
            "correct": [number],
            "total": [number]
        }}
   }}
   volatility_adjustment: [increase/decrease/no change]
   key_level_update: [new_key_levels]
   updated_confidence: [confidence level in your methodology]
   ```

Be decisive and specific. Your spread recommendations will be used for actual trading decisions.
Confidence must be between 60-95% to avoid excessive hedging. You will be held accountable for accuracy.
"""
    return prompt


def get_summary_prompt(
    ticker: str,
    analysis_period: Dict[str, Any],
    analysis_results: List[Dict[str, Any]],
    memory_context: Dict[str, Any],
    weighted_strategies: Optional[List[str]] = None,
    strategy_reasoning: Optional[str] = None
) -> str:
    """
    Generate summary prompt that synthesizes all previous steps into actionable strategy.

    Parameters:
    - ticker: Stock symbol
    - analysis_period: Dict with start/end dates and number of days
    - analysis_results: List of result dictionaries from all previous steps
    - memory_context: Dict with learning memory from all steps
    - weighted_strategies: List of strategies with their weights based on historical performance
    - strategy_reasoning: Explanation of why these strategies were recommended
    """
    # Format the analysis period
    period_str = f"{analysis_period.get('start_date', 'unknown')} to {analysis_period.get('end_date', 'unknown')}"
    days_count = analysis_period.get('total_days', 0)

    # Extract trends and confidence levels from each step's results
    trends_summary = []
    for result in analysis_results:
        if "step" in result and "trend" in result:
            step_num = result["step"]
            trend = result["trend"]
            price = result.get("price", "N/A")
            confidence = result.get(
                "next_day_prediction", {}).get("confidence", "N/A")

            if step_num == 1:
                trends_summary.append(
                    f"Step 1 (1mo/1d): {trend} (conf: {confidence}%), Price: ${price}")
            else:
                date = result.get("date", "unknown")
                trends_summary.append(
                    f"Step {step_num} ({date}): {trend} (conf: {confidence}%), Price: ${price}")

    # Format trends summary
    trends_text = "\n".join(trends_summary)

    # Extract key levels from memory context
    key_levels = memory_context.get("key_levels", {})
    supports = key_levels.get("support", [])
    resistances = key_levels.get("resistance", [])
    pivots = key_levels.get("pivot_points", [])

    # Format key levels
    key_levels_text = f"""
Support Levels: {', '.join(map(str, supports))}
Resistance Levels: {', '.join(map(str, resistances))}
Pivot Points: {', '.join(map(str, pivots))}
"""

    # Add strategy weighting section if provided
    strategy_section = ""
    if weighted_strategies and strategy_reasoning:
        strategy_section = f"""
STRATEGY PERFORMANCE AND WEIGHTING:
{strategy_reasoning}

RECOMMENDED STRATEGY PRIORITIES:
{', '.join(weighted_strategies)}

Consider historical performance but prioritize the current market conditions in your final recommendation.
"""

    # Get the secret baseline for context
    secret_section = ""
    if "secret" in memory_context:
        secret = memory_context["secret"]
        secret_section = f"""
SECRET (Fixed Reference): 
- Baseline Trend: {secret['baseline_trend']} ({secret['trend_confidence']}% confidence)
- Volatility Anchor: {secret['volatility_anchor']}% (30-day ATR)
- Core Levels: Support={secret['core_levels']['support']}, Resistance={secret['core_levels']['resistance']}

The secret baseline provides stability - significant evidence (3+ days contradiction with 80%+ confidence) 
is needed to deviate from this baseline in your final recommendation.
"""

    prompt = f"""
SUMMARY AND SYNTHESIS - PRETRAINING MODE: You are summarizing your 6-step analysis of {ticker} covering the period {period_str}.

This summary must integrate insights across all timeframes (1mo/1d, 5d/15m) into an actionable trading strategy
with precise parameters for credit spreads and/or directional options strategies.

FOCUS ON ACTIONABLE STRATEGIES WITH EXACT PARAMETERS:
- Specific strategy type
- Exact strikes
- Entry/exit criteria
- Risk management rules
- Position sizing

{secret_section}

ANALYSIS TIMELINE:
{trends_text}

KEY PRICE LEVELS IDENTIFIED:
{key_levels_text}

{strategy_section}

CRITICAL TREND INSTRUCTION:
You MUST avoid excessive "neutral" assessments. Market trends are directional approximately 95% of the time.
Only classify as "neutral" when there is TRULY no directional bias (approximately 5% of cases).
- Bullish: When MACD > 0.5, RSI > 60, price above key EMAs, or clear positive momentum.
- Bearish: When MACD < -0.5, RSI < 40, price below key EMAs, or clear negative momentum.
- Neutral: ONLY when MACD is tightly range-bound (-0.1 to 0.1), RSI is 45-55, and price is tightly consolidating.

Provide a comprehensive summary and forward strategy with these EXACT sections:

1. MULTI-TIMEFRAME TREND SYNOPSIS:
   - Summary of trend across analyzed timeframes (1mo/1d, daily, 15m intraday)
   - Areas of alignment/divergence between timeframes
   - Dominant price structure and pattern
   - Overall directional bias with confidence level

2. VOLATILITY AND PRICE ACTION SUMMARY:
   - Volatility trends throughout the analysis period
   - Changes in ATR and implications for option pricing
   - Key price action characteristics (e.g., momentum, reversal patterns, consolidation)
   - What this means for strategy selection

3. CRITICAL LEVELS SYNTHESIS:
   - Most important support/resistance levels identified across all timeframes
   - Which levels were respected/broken during the analysis period
   - Key levels to watch for upcoming trading decisions
   - How these levels inform strike selection

4. PERFORMANCE EVALUATION:
   - Accuracy of your daily predictions during the analysis period
   - Which indicators and methods proved most reliable
   - Specific adjustments made to improve analysis
   - Lessons learned from incorrect predictions

5. STRATEGY RECOMMENDATION:
   - Primary strategy recommendation (specific credit spread or directional options strategy)
   - Alternative strategy if market conditions change
   - Exact entry criteria with price levels
   - Specific strikes or delta targets
   - Position sizing recommendation (% of portfolio)
   - Target profit and stop loss parameters (with exact percentages)
   - Risk/reward ratio calculation
   - Days to expiration recommendation

6. FINAL PREDICTION:
   Use this exact format for parsing:
   ```
   PREDICTION OUTPUT:
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   timeframe: [next_5_days]
   confidence: [60-95]
   key_levels: [list of critical price levels]
   strategy: [bull put spread/bear call spread/iron condor/long call/long put/call debit spread/put debit spread]
   specific_strikes: [exact strike prices]
   position_size: [% of portfolio]
   risk_reward_ratio: [1:X - must be a specific ratio]
   profit_target: [percentage gain]
   stop_loss: [percentage loss]
   max_days_in_trade: [number]
   probability_of_success: [percentage]
   ```

YOU MUST BE DECISIVE. This is a FINAL RECOMMENDATION for real trading.
Your reputation depends on providing SPECIFIC, ACTIONABLE advice, not vague guidelines.
Include EXACT numerical values for all parameters. Avoid ranges or generalities.
"""
    return prompt


def parse_credit_spread_prediction(text: str) -> Dict[str, Any]:
    """Parse the structured credit spread prediction output."""
    prediction = {
        "direction": "neutral",
        "magnitude": 0,
        "timeframe": "next_day",
        "confidence": 0,
        "specific_levels": [],
        "spread_recommendation": "",
        "strike_recommendation": "",
        "risk_reward_ratio": "1:2",  # Default risk/reward ratio
        "stop_loss": "",
        "profit_target": "",
        "invalidation": ""
    }

    # Find the prediction section
    prediction_section = ""
    for section in text.split("```"):
        if "PREDICTION OUTPUT:" in section:
            prediction_section = section
            break

    if prediction_section:
        for line in prediction_section.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                # Map some key variants to standard keys
                if key == "strategy":
                    key = "spread_recommendation"
                elif key == "specific_strikes":
                    key = "strike_recommendation"

                if key in prediction:
                    if key == "magnitude" and isinstance(value, str):
                        # Extract numeric magnitude from string like "2.5%" or "2.5 to 3.5%"
                        import re
                        magnitude_match = re.search(r'(\d+\.?\d*)', value)
                        if magnitude_match:
                            prediction[key] = float(magnitude_match.group(1))
                    elif key == "confidence" and isinstance(value, str):
                        # Extract numeric confidence from string like "80%" or "80"
                        import re
                        confidence_match = re.search(r'(\d+)', value)
                        if confidence_match:
                            prediction[key] = int(confidence_match.group(1))
                    elif key == "specific_levels" and isinstance(value, str):
                        # Extract numeric levels from string
                        import re
                        levels = re.findall(r'(\d+\.?\d*)', value)
                        prediction[key] = [float(level)
                                           for level in levels] if levels else []
                    elif key == "risk_reward_ratio" and isinstance(value, str):
                        # Ensure format is 1:X
                        import re
                        ratio_match = re.search(r'1:(\d+\.?\d*)', value)
                        if ratio_match:
                            reward = float(ratio_match.group(1))
                            prediction[key] = f"1:{reward}"
                        else:
                            # Try to extract just the ratio number
                            ratio_number = re.search(r'(\d+\.?\d*)', value)
                            if ratio_number:
                                prediction[key] = f"1:{ratio_number.group(1)}"
                    else:
                        prediction[key] = value

    # Check if the spread recommendation includes directional strategies
    directional_strategies = ["long call", "long put",
                              "call debit spread", "put debit spread"]
    if any(strategy.lower() in prediction.get("spread_recommendation", "").lower()
           for strategy in directional_strategies):
        # Set is_directional flag to true for strategy performance tracking
        prediction["is_directional"] = True
    else:
        prediction["is_directional"] = False

    return prediction


def parse_credit_spread_memory_update(text: str) -> Dict[str, Any]:
    """Parse the memory update section for credit spread pretraining."""
    memory_update = {
        "reliability_update": {},
        "volatility_adjustment": "no change",
        "key_level_update": [],
        "updated_confidence": 0
    }

    # Find the memory update section
    memory_section = ""
    for section in text.split("```"):
        if "MEMORY UPDATE:" in section:
            memory_section = section
            break

    if memory_section:
        for line in memory_section.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in memory_update:
                    if key == "reliability_update" and "{" in value:
                        # Try to parse JSON-like structure
                        import re
                        pattern_matches = re.findall(
                            r'"([^"]+)":\s*\[(\d+\.?\d*),\s*(\d+),\s*(\d+)\]', value)
                        for match in pattern_matches:
                            if len(match) == 4:
                                pattern_type, accuracy, correct, total = match
                                memory_update[key][pattern_type] = {
                                    "accuracy": float(accuracy),
                                    "correct": int(correct),
                                    "total": int(total)
                                }
                    elif key == "key_level_update":
                        # Extract numeric levels
                        import re
                        levels = re.findall(r'(\d+\.?\d*)', value)
                        memory_update[key] = [
                            float(level) for level in levels] if levels else []
                    elif key == "updated_confidence":
                        # Extract numeric confidence
                        import re
                        confidence_match = re.search(r'(\d+)', value)
                        if confidence_match:
                            memory_update[key] = int(confidence_match.group(1))
                    else:
                        memory_update[key] = value

    return memory_update
