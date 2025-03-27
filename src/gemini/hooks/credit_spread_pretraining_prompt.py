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
Deviate from the secret ONLY with 90%+ confidence based on significant new information.
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
   - Ideal strike selection with rationale
   - Entry timing recommendation based on intraday patterns
   - Probability of success estimate
   - Risk/reward profile

6. PREDICTION:
   Use this exact format for parsing:
   ```
   PREDICTION OUTPUT:
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   timeframe: [next_day]
   confidence: [60-95]
   specific_levels: [list of critical price levels for tomorrow]
   spread_recommendation: [bull put spread/bear call spread/iron condor]
   strike_recommendation: [specific strike prices with rationale]
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
    Generate prompt for the summary step that concludes the 6-step analysis.

    This summary synthesizes all the previous analyses and provides actionable
    trading strategy for credit spreads.

    Parameters:
    - ticker: Stock symbol
    - analysis_period: Dict with start and end dates
    - analysis_results: List of results from previous steps
    - memory_context: Dict with learning memory from all steps
    - weighted_strategies: List of strategies with their weights based on historical performance
    - strategy_reasoning: Explanation of why these strategies were recommended
    """
    # Extract period information
    start_date = analysis_period.get('start_date', 'unknown')
    end_date = analysis_period.get('end_date', 'unknown')

    # Calculate accuracy metrics
    total_predictions = sum(1 for r in analysis_results if r.get(
        'prediction_accuracy') is not None)
    correct_predictions = sum(1 for r in analysis_results
                              if r.get('prediction_accuracy') is not None
                              and r.get('prediction_accuracy', {}).get('direction_correct', False))

    accuracy_rate = correct_predictions / \
        total_predictions if total_predictions > 0 else 0

    # Get performance matrix for different strategies
    bull_put_data = memory_context.get('spread_performance', {}).get(
        'bull_put', {'wins': 0, 'total': 0, 'win_rate': 0})
    bear_call_data = memory_context.get('spread_performance', {}).get(
        'bear_call', {'wins': 0, 'total': 0, 'win_rate': 0})
    iron_condor_data = memory_context.get('spread_performance', {}).get(
        'iron_condor', {'wins': 0, 'total': 0, 'win_rate': 0})

    # Format strategy statistics
    strategy_stats = f"""
Strategy Performance:
- Bull Put Spreads: {bull_put_data.get('wins', 0)}/{bull_put_data.get('total', 0)} successful ({bull_put_data.get('win_rate', 0)*100:.1f}%)
- Bear Call Spreads: {bear_call_data.get('wins', 0)}/{bear_call_data.get('total', 0)} successful ({bear_call_data.get('win_rate', 0)*100:.1f}%)
- Iron Condors: {iron_condor_data.get('wins', 0)}/{iron_condor_data.get('total', 0)} successful ({iron_condor_data.get('win_rate', 0)*100:.1f}%)
"""

    # Add strategy weighting section if provided
    strategy_section = ""
    if weighted_strategies and strategy_reasoning:
        strategy_section = f"""
OPTIMIZED STRATEGY ALLOCATION:
{strategy_reasoning}

RECOMMENDED STRATEGY PRIORITIES:
{', '.join(weighted_strategies)}

Focus particularly on the highest-weighted strategies, but consider current market conditions and adjust accordingly.
"""

    # Generate the baseline secret key
    secret_anchor = ""
    if "secret" in memory_context:
        secret = memory_context["secret"]
        secret_anchor = f"""
SECRET BASELINE (Do not contradict):
- Baseline Trend: {secret.get('baseline_trend', 'neutral')} 
- Trend Confidence: {secret.get('trend_confidence', 60)}%
- Volatility Anchor: {secret.get('volatility_anchor', 1.0)}%
- Core Support Levels: {secret.get('core_levels', {}).get('support', [])}
- Core Resistance Levels: {secret.get('core_levels', {}).get('resistance', [])}
"""

    # Extract daily and intraday predictions
    daily_predictions = [r for r in analysis_results if r.get('step') == 1]
    intraday_predictions = [
        r for r in analysis_results if r.get('step') in range(2, 7)]

    # Format predictions for the summary
    daily_summary = ""
    if daily_predictions:
        daily_pred = daily_predictions[0].get('next_day_prediction', {})
        daily_summary = f"""
Long-term Daily Analysis:
- Trend: {daily_pred.get('direction', 'neutral')}
- Confidence: {daily_pred.get('confidence', 0)}%
- Key Levels: {daily_pred.get('specific_levels', 'None identified')}
"""

    intraday_summary = ""
    if intraday_predictions:
        intraday_summary = "Intraday Analysis Progression:\n"
        for i, pred in enumerate(intraday_predictions):
            prediction = pred.get('next_day_prediction', {})
            intraday_summary += f"- Day {i+1}: {prediction.get('direction', 'neutral')} with {prediction.get('confidence', 0)}% confidence\n"

    # Create the prompt
    prompt = f"""
SUMMARY AND STRATEGY DEVELOPMENT FOR {ticker}

You are synthesizing a comprehensive 6-step credit spread trading analysis for {ticker} covering the period {start_date} to {end_date}.

ANALYSIS METRICS:
- Total Trading Days Analyzed: {len(intraday_predictions)}
- Overall Prediction Accuracy: {accuracy_rate*100:.1f}% ({correct_predictions}/{total_predictions})
{strategy_stats}

{secret_anchor}

{strategy_section}

ANALYSIS SUMMARY:
{daily_summary}
{intraday_summary}

Your task is to synthesize all analysis steps into a coherent trading strategy with specific recommendations:

1. TREND SYNTHESIS AND PATTERN IDENTIFICATION:
   - Consensus direction across all timeframes
   - Most reliable technical patterns observed
   - Divergences between intraday and daily analysis
   - Volume profile assessment

2. KEY LEVEL CONSOLIDATION:
   - Synthesis of critical support and resistance levels
   - Price action behavior around these levels
   - Most important price zones for spread strike selection
   - Highest probability boundaries for price movement

3. VOLATILITY PROFILE:
   - Summary of volatility behavior and expectations
   - IV vs HV relationship and implications
   - Ideal premium collection zones
   - Risk management parameters based on volatility profile

4. OPTIMIZED SPREAD STRATEGY:
   - Most appropriate credit spread type for current conditions
   - Specific strike selection with detailed rationale 
   - Optimal entry timing recommendations
   - Position sizing and risk parameters
   - Clear invalidation criteria

5. MULTI-TIMEFRAME FORECAST:
   Use this EXACT format for each timeframe:
   ```
   FORECAST: NEXT DAY (1-2 Day Outlook)
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   confidence: [60-95]
   spread_recommendation: [specific spread type]
   strike_selection: [specific strikes with detailed rationale]
   key_levels: [critical price levels]
   invalidation: [specific invalidation criteria]
   risk_reward: [calculated risk/reward ratio]
   ```

   ```
   FORECAST: NEXT WEEK (3-5 Day Outlook)
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   confidence: [60-95]
   spread_recommendation: [specific spread type]
   strike_selection: [specific strikes with detailed rationale]
   key_levels: [critical price levels]
   invalidation: [specific invalidation criteria]
   risk_reward: [calculated risk/reward ratio]
   ```

   ```
   FORECAST: NEXT MONTH (20-30 Day Outlook)
   direction: [bullish/bearish/neutral]
   magnitude: [expected percentage change]
   confidence: [60-95]
   spread_recommendation: [specific spread type]
   strike_selection: [specific strikes with detailed rationale]
   key_levels: [critical price levels]
   invalidation: [specific invalidation criteria]
   risk_reward: [calculated risk/reward ratio]
   ```

CRITICAL GUIDELINES:
1. Be extremely specific with strike selections and exact price levels
2. Include numerical risk/reward calculations 
3. Give precise entry and exit criteria, not general guidelines
4. Demonstrate critical thinking in synthesizing contradictory signals
5. Specify which technical indicators proved most reliable for this ticker
6. Focus on actionable credit spread strategies, not general market commentary

Your analysis will be used for actual trading decisions with significant capital at risk.
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
                    else:
                        prediction[key] = value

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
