"""
Enhanced pretraining prompts for Gemini API.

This module provides optimized prompts for the pretraining process that:
1. Maximize the value of each API call
2. Combine multiple analysis stages into single requests
3. Enable structured output for better pattern recognition
4. Support multi-timeframe analysis
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def format_stock_data(data: Dict[str, Any]) -> str:
    """Format stock data for inclusion in prompts, handling both single and multi-timeframe data."""
    if not data:
        return "No stock data available."

    ticker = data.get('ticker', 'Unknown')
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Format daily data
    daily_data = []
    for key, value in data.items():
        if key not in ['ticker', 'date', 'weekly_data', 'quality_warning']:
            if isinstance(value, float):
                daily_data.append(f"{key}: {value:.2f}")
            else:
                daily_data.append(f"{key}: {value}")

    # Format weekly data if available
    weekly_section = ""
    if 'weekly_data' in data and data['weekly_data']:
        weekly_data = []
        for key, value in data['weekly_data'].items():
            if isinstance(value, float):
                weekly_data.append(f"{key}: {value:.2f}")
            else:
                weekly_data.append(f"{key}: {value}")
        weekly_section = f"\nWEEKLY DATA:\n" + "\n".join(weekly_data)

    # Add quality warning if present
    quality_warning = f"\nQUALITY WARNING: {data.get('quality_warning')}" if 'quality_warning' in data else ""

    return f"TICKER: {ticker}\nDATE: {date}\nDAILY DATA:\n" + "\n".join(daily_data) + weekly_section + quality_warning


def format_market_context(context: Dict[str, Any]) -> str:
    """Format market context for inclusion in prompts."""
    if not context:
        return "No market context available."

    # Extract core market information
    spy_trend = context.get('spy_trend', 'neutral')
    market_trend_score = context.get(
        'market_trend_score', context.get('trend_score', 0))
    vix = context.get('vix', 'Unknown')

    # Format sector information if available
    sector_info = ""
    if 'sector_performance' in context:
        sectors = []
        for sector, perf in context['sector_performance'].items():
            if isinstance(perf, (int, float)):
                sectors.append(f"{sector}: {perf:.2f}%")
            else:
                sectors.append(f"{sector}: {perf}")
        sector_info = "\nSECTOR PERFORMANCE:\n" + "\n".join(sectors)

    # Format options data if available
    options_info = ""
    if 'options_analysis' in context:
        options = context['options_analysis']
        options_info = f"\nOPTIONS CONTEXT:\nDirection: {options.get('direction', 'neutral')}\nConfidence: {options.get('confidence', 'low')}"

    return f"SPY TREND: {spy_trend}\nMARKET TREND SCORE: {market_trend_score}\nVIX: {vix}{sector_info}{options_info}"


def format_memory_context(memory: Dict[str, Any]) -> str:
    """Format memory context for inclusion in prompts."""
    if not memory:
        return "No memory context available."

    # Format pattern library
    pattern_library = []
    if 'pattern_library' in memory and memory['pattern_library']:
        for pattern_type, patterns in memory['pattern_library'].items():
            if patterns:
                pattern_library.append(
                    f"{pattern_type}: {len(patterns)} pattern(s) identified")

    # Format key levels
    key_levels = []
    if 'key_levels' in memory and memory['key_levels']:
        for level_type, levels in memory['key_levels'].items():
            if levels:
                key_levels.append(
                    f"{level_type}: {', '.join(map(str, levels))}")

    # Format pattern reliability
    reliability = []
    if 'pattern_reliability' in memory and memory['pattern_reliability']:
        for pattern_type, stats in memory['pattern_reliability'].items():
            if stats.get('total', 0) > 0:
                accuracy = stats.get('accuracy', 0)
                reliability.append(
                    f"{pattern_type}: {accuracy:.1f}% accuracy ({stats.get('correct', 0)}/{stats.get('total', 0)})")

    # Format successful patterns
    successful_patterns = []
    if 'successful_patterns' in memory and memory['successful_patterns']:
        # Only include the most recent 3
        for pattern in memory['successful_patterns'][:3]:
            successful_patterns.append(
                f"- {pattern.get('type', 'Unknown')} at {pattern.get('price', 0)}")

    return f"""PATTERN LIBRARY:
{chr(10).join(pattern_library) if pattern_library else "No patterns identified yet."}

KEY LEVELS:
{chr(10).join(key_levels) if key_levels else "No key levels identified yet."}

PATTERN RELIABILITY:
{chr(10).join(reliability) if reliability else "No pattern reliability data yet."}

SUCCESSFUL PATTERNS:
{chr(10).join(successful_patterns) if successful_patterns else "No successful patterns recorded yet."}

MULTI-TIMEFRAME:
Weekly Trend: {memory.get('multi_timeframe', {}).get('weekly_trend', 'neutral')}
"""


def get_enhanced_pretraining_prompt(
    stock_data: Dict[str, Any],
    market_context: Dict[str, Any],
    memory_context: Optional[Dict[str, Any]] = None,
    current_date: Optional[str] = None,
    next_day_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an enhanced prompt that combines analysis and reflection in a single call.

    Parameters:
    - stock_data: Dict containing stock data for the current date
    - market_context: Dict with market context for the current date
    - memory_context: Optional dict with pattern memory and learning
    - current_date: Optional string with the current date in analysis
    - next_day_data: Optional dict with the next day's data (for reflection)

    Returns:
    - A comprehensive prompt string for the AI Analyzer
    """
    date_str = current_date if current_date else stock_data.get(
        'date', datetime.now().strftime('%Y-%m-%d'))
    ticker = stock_data.get('ticker', 'Unknown')

    # Determine if this is a reflection prompt (next day data available)
    is_reflection = next_day_data is not None
    reflection_section = ""

    if is_reflection:
        # Format next day data for reflection
        next_date = next_day_data.get('date', 'next day')
        next_price = next_day_data.get('current_price', 0)
        prev_price = stock_data.get('current_price', 0)

        if prev_price > 0:
            price_change = (next_price - prev_price) / prev_price * 100
            price_change_str = f"{price_change:.2f}%"
        else:
            price_change_str = "unknown"

        direction = "up" if price_change > 0 else "down" if price_change < 0 else "flat"

        reflection_section = f"""
        REFLECTION MODE: This is a reflection prompt with next day data.
        
        ACTUAL OUTCOME:
        Next Day: {next_date}
        Price Change: {price_change_str} ({direction})
        Opening Price: {next_day_data.get('open', 'unknown')}
        Closing Price: {next_day_data.get('close', 'unknown')}
        High: {next_day_data.get('high', 'unknown')}
        Low: {next_day_data.get('low', 'unknown')}
        Volume: {next_day_data.get('volume', 'unknown')}
        
        REFLECTION INSTRUCTIONS:
        1. Evaluate the accuracy of your previous prediction
        2. Identify which patterns performed as expected
        3. Note which technical indicators were most reliable
        4. Update pattern reliability scores based on outcome
        5. Suggest adjustments to your analysis approach
        """

    # Check for data quality issues
    limited_data = False
    extremely_limited_data = False
    data_warning = ""

    if "quality_warning" in stock_data:
        data_warning = stock_data["quality_warning"]
        if "SEVERELY limited" in data_warning:
            extremely_limited_data = True
        elif "limited" in data_warning.lower():
            limited_data = True

        # Add data warning section
        data_warning = f"""
        DATA QUALITY WARNING: {data_warning}
        
        Given the limited historical data available, please adjust your analysis as follows:
        1. Focus on shorter-term indicators and price action
        2. Avoid complex patterns that require extensive historical data
        3. Lower confidence scores for any predictions
        4. Prioritize recent price movements over historical patterns
        5. Be explicit about the limitations in your analysis
        """

    # Adjust pattern recognition rules based on data availability
    pattern_recognition_section = """
    RULES FOR PATTERN RECOGNITION:
    1. Identify chart patterns (Head & Shoulders, Double Tops/Bottoms, Flags, Triangles, etc.)
    2. Calculate ZigZag pivot points to identify swing highs/lows
    3. Detect support/resistance levels and trendlines
    4. Score patterns by clarity, completion, and confirmation
    5. Assess pattern reliability based on historical occurrence
    """

    if extremely_limited_data:
        pattern_recognition_section = """
        RULES FOR PATTERN RECOGNITION (EXTREMELY LIMITED DATA MODE):
        1. Focus ONLY on basic price action (up/down/sideways)
        2. Identify only the most obvious support/resistance levels
        3. Avoid complex pattern identification due to insufficient data
        4. Use only the most recent price movements for analysis
        5. Maintain low confidence in all pattern identifications
        """
    elif limited_data:
        pattern_recognition_section = """
        RULES FOR PATTERN RECOGNITION (LIMITED DATA MODE):
        1. Focus on simple patterns (basic trends, support/resistance)
        2. Use a simplified ZigZag approach for pivot points
        3. Detect only clear and obvious support/resistance levels
        4. Score patterns conservatively due to limited data
        5. Prioritize recent price action over historical patterns
        """

    # Build the main prompt
    prompt = f"""
    {"REFLECTION AND " if is_reflection else ""}PRETRAINING MODE: You are analyzing historical data for {ticker} on {date_str}.
    {"You must first analyze the current day, then reflect on your previous prediction's accuracy." if is_reflection else "You only have data up to this date. DO NOT use knowledge of future market movements."}
    
    {data_warning}
    
    {pattern_recognition_section}
    
    TECHNICAL ANALYSIS RULES:
    1. Use EMAs (9,21) and SMAs (50,200) for trend identification
    2. Consider RSI, MACD, and Bollinger Bands for confirmation
    3. Evaluate volume patterns and trend strength (ADX)
    4. Identify market phase (trending, ranging, or transitioning)
    
    MULTI-TIMEFRAME RULES:
    1. Compare daily and weekly trends for alignment
    2. Identify confluence across timeframes
    3. Give higher weight to patterns confirmed across timeframes
    
    MARKET DATA:
    {format_stock_data(stock_data)}
    
    MARKET CONTEXT:
    {format_market_context(market_context)}
    
    MEMORY CONTEXT:
    {format_memory_context(memory_context) if memory_context else "No previous patterns identified."}
    
    {reflection_section}
    
    Provide a comprehensive analysis with these EXACT sections:
    1. PATTERN RECOGNITION: (identify all chart patterns, support/resistance, pivot points)
    2. TECHNICAL ANALYSIS: (analyze all indicators, trend strength, price structure)
    3. MULTI-TIMEFRAME ANALYSIS: (compare daily/weekly alignment)
    4. PREDICTION: (specific direction, magnitude, timeframe, confidence)
    5. {"REFLECTION: (evaluate previous prediction accuracy)" if is_reflection else ""}
    6. MEMORY UPDATE: (patterns to add/update, reliability changes)
    
    For PREDICTION, use this exact format so it can be parsed programmatically:
    ```
    PREDICTION OUTPUT:
    direction: [bullish/bearish/neutral]
    magnitude: [expected percentage change]
    timeframe: [next_day/next_week]
    confidence: [0-100]
    invalidation: [condition that would invalidate this prediction]
    ```
    
    For MEMORY UPDATE, use this exact format so it can be parsed programmatically:
    ```
    PATTERN STORAGE:
    pattern_type: [type]
    pattern_points: [list of price points]
    confidence: [0-100]
    prediction: [bullish/bearish/neutral]
    key_levels: [support/resistance levels]
    ```
    
    {"Include a 'reliability_update' section with the updated accuracy scores for each pattern type." if is_reflection else ""}
    """

    return prompt


def get_enhanced_summary_prompt(
    ticker: str,
    analysis_period: Dict[str, Any],
    analysis_results: List[Dict[str, Any]],
    memory_context: Dict[str, Any]
) -> str:
    """
    Generate an enhanced summary prompt that creates a comprehensive review
    of the entire pretraining sequence.

    Parameters:
    - ticker: Stock symbol
    - analysis_period: Dict with start_date and end_date
    - analysis_results: List of analysis results from the pretraining sequence
    - memory_context: Dict with pattern memory and learning from pretraining

    Returns:
    - A summary prompt string for the AI Analyzer
    """
    start_date = analysis_period.get('start_date', 'unknown')
    end_date = analysis_period.get('end_date', 'unknown')

    # Extract recent analysis results (last 3)
    recent_analyses = analysis_results[-3:] if len(
        analysis_results) > 3 else analysis_results
    recent_summary = "\n\n".join([
        f"Date: {analysis.get('date', 'unknown')}\n"
        f"Price: {analysis.get('price', 0)}\n"
        f"Trend: {analysis.get('trend', 'neutral')}\n"
        f"Technical Score: {analysis.get('technical_score', 0)}\n"
        f"Prediction: {json.dumps(analysis.get('next_day_prediction', {}), indent=2)}"
        for analysis in recent_analyses
    ])

    # Format pattern reliability from memory context
    reliability_summary = []
    if 'pattern_reliability' in memory_context:
        for pattern_type, stats in memory_context['pattern_reliability'].items():
            if stats.get('total', 0) > 0:
                accuracy = stats.get('accuracy', 0)
                reliability_summary.append(
                    f"{pattern_type}: {accuracy:.1f}% accuracy ({stats.get('correct', 0)}/{stats.get('total', 0)})"
                )

    # Build prediction horizons
    prediction_horizons = [
        "Next Day (1-2 days)", "Next Week (3-5 days)", "Next Month (20-30 days)"]

    return f"""
    PRETRAINING SUMMARY MODE: You are now generating a final summary and multi-timeframe forecast
    based on the entire pretraining sequence for {ticker}.
    
    Pretraining Period: {start_date} to {end_date}
    Total Analyses: {len(analysis_results)}
    
    Recent Analysis Summary:
    {recent_summary}
    
    Pattern Reliability Summary:
    {chr(10).join(reliability_summary) if reliability_summary else "No pattern reliability data available."}
    
    MEMORY CONTEXT:
    {format_memory_context(memory_context)}
    
    Based on all the analyses and reflections from the pretraining period, please provide:
    
    1. KEY PATTERNS OVERVIEW:
       - Summarize the most consistent technical patterns observed
       - Identify which indicators provided the most reliable signals
       - Highlight any correlations between market conditions and stock movement
    
    2. VOLATILITY PROFILE:
       - Characterize the stock's typical volatility range
       - Identify conditions that preceded volatility changes
       - Suggest optimal position sizing based on historical volatility
    
    3. MARKET RELATIONSHIP:
       - How strongly does this stock correlate with broad market moves?
       - Does it tend to lead or lag market trends?
       - Any sectors or other stocks that show strong correlation?
    
    4. MULTI-TIMEFRAME FORECAST:
       For each of the following timeframes, provide a detailed prediction with this exact format:
       
       ```
       FORECAST: [timeframe]
       direction: [bullish/bearish/neutral]
       magnitude: [expected percentage change]
       confidence: [0-100]
       key_triggers: [events or conditions that would trigger the move]
       risk_factors: [conditions that could invalidate the forecast]
       ```
    
       Your forecast MUST include separate sections for each of these horizons:
       {', '.join(prediction_horizons)}
    
    5. TRADING STRATEGY RECOMMENDATIONS:
       - Optimal types of setups to watch for
       - Specific entry criteria based on historical patterns
       - Suggested risk management parameters
       - For credit spreads specifically:
         * Ideal DTE (days to expiration)
         * Strike price selection criteria
         * Maximum position size recommendations
         * Adjustment/management strategy
    
    REMINDER: This summary will be used to inform actual trading decisions, so be specific, 
    data-driven, and focused on actionable insights rather than generalities.
    
    Format your response with clear section headers and structured output for the forecasts.
    """


def get_enhanced_batch_prompt(
    ticker: str,
    date_windows: List[Dict[str, Any]],
    market_contexts: Dict[str, Dict[str, Any]],
    memory_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a prompt that processes multiple days in a single batch to maximize API efficiency.

    Parameters:
    - ticker: Stock symbol
    - date_windows: List of dicts with date and stock data for multiple consecutive days
    - market_contexts: Dict mapping dates to market context data
    - memory_context: Optional dict with pattern memory and learning

    Returns:
    - A batch processing prompt string for the AI Analyzer
    """
    if not date_windows:
        return "Error: No date windows provided for batch processing."

    # Format dates for display
    date_range = f"{date_windows[0]['date']} to {date_windows[-1]['date']}"

    # Create sections for each date
    date_sections = []
    for window in date_windows:
        date = window.get('date', 'unknown')
        market_context = market_contexts.get(date, {"spy_trend": "neutral"})

        data_section = format_stock_data(window)
        market_section = format_market_context(market_context)

        date_sections.append(f"""
        DATE: {date}
        
        {data_section}
        
        MARKET CONTEXT:
        {market_section}
        """)

    return f"""
    BATCH PRETRAINING MODE: You are analyzing multiple days of historical data for {ticker} from {date_range}.
    
    RULES FOR PATTERN RECOGNITION:
    1. Identify chart patterns (Head & Shoulders, Double Tops/Bottoms, Flags, Triangles, etc.)
    2. Calculate ZigZag pivot points to identify swing highs/lows
    3. Detect support/resistance levels and trendlines
    4. Score patterns by clarity, completion, and confirmation
    5. Assess pattern reliability based on historical occurrence
    
    TECHNICAL ANALYSIS RULES:
    1. Use EMAs (9,21) and SMAs (50,200) for trend identification
    2. Consider RSI, MACD, and Bollinger Bands for confirmation
    3. Evaluate volume patterns and trend strength (ADX)
    4. Identify market phase (trending, ranging, or transitioning)
    
    MEMORY CONTEXT:
    {format_memory_context(memory_context) if memory_context else "No previous patterns identified."}
    
    BATCH DATA:
    {chr(10).join(date_sections)}
    
    For each date, provide a separate analysis with these sections:
    1. DATE: [the date being analyzed]
    2. PATTERN RECOGNITION: (identify all chart patterns, support/resistance, pivot points)
    3. TECHNICAL ANALYSIS: (analyze all indicators, trend strength, price structure)
    4. PREDICTION: (specific direction, magnitude, timeframe, confidence)
    
    After analyzing all dates, provide:
    
    SEQUENCE ANALYSIS:
    1. Pattern Evolution: How patterns developed across the sequence
    2. Key Support/Resistance: Levels that remained consistent
    3. Trend Changes: Where and why the trend changed
    4. Prediction Accuracy: How accurate previous day predictions would have been
    
    MEMORY UPDATE:
    ```
    PATTERN STORAGE:
    pattern_type: [type]
    pattern_points: [list of price points]
    confidence: [0-100]
    prediction: [bullish/bearish/neutral]
    key_levels: [support/resistance levels]
    ```
    
    Format your response with clear separation between each date's analysis.
    """

# Additional utility functions to parse structured output from enhanced prompts


def parse_enhanced_prediction(text: str) -> Dict[str, Any]:
    """Parse the structured prediction output from the enhanced prompt response."""
    prediction = {
        "direction": "neutral",
        "magnitude": 0,
        "timeframe": "next_day",
        "confidence": 0,
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
                    else:
                        prediction[key] = value

    return prediction


def parse_pattern_storage(text: str) -> List[Dict[str, Any]]:
    """Parse the structured pattern storage output from the enhanced prompt response."""
    patterns = []

    # Find all pattern storage sections
    pattern_sections = []
    for section in text.split("```"):
        if "PATTERN STORAGE:" in section:
            pattern_sections.append(section)

    for section in pattern_sections:
        pattern = {
            "pattern_type": "",
            "pattern_points": [],
            "confidence": 0,
            "prediction": "neutral",
            "key_levels": []
        }

        for line in section.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in pattern:
                    if key == "pattern_points" or key == "key_levels":
                        # Convert string representation of list to actual list
                        import re
                        points = re.findall(r'[\d\.]+', value)
                        pattern[key] = [float(p) for p in points if p]
                    elif key == "confidence":
                        # Extract numeric confidence
                        import re
                        confidence_match = re.search(r'(\d+)', value)
                        if confidence_match:
                            pattern[key] = int(confidence_match.group(1))
                    else:
                        pattern[key] = value

        if pattern["pattern_type"]:  # Only add if we have a pattern type
            patterns.append(pattern)

    return patterns


def parse_reliability_update(text: str) -> Dict[str, Dict[str, Any]]:
    """Parse the reliability update section from a reflection response."""
    reliability = {}

    # Find the reliability update section
    reliability_section = ""
    for section in text.split("\n\n"):
        if "reliability_update" in section.lower():
            reliability_section = section
            break

    if reliability_section:
        for line in reliability_section.split("\n"):
            line = line.strip()
            if ":" in line and "accuracy" in line.lower():
                # Parse lines like "head_and_shoulders: 75% accuracy (3/4)"
                import re
                pattern_match = re.match(
                    r'([a-z_&]+):\s*(\d+)%\s*accuracy\s*\((\d+)/(\d+)\)', line, re.IGNORECASE)
                if pattern_match:
                    pattern_type = pattern_match.group(1).lower()
                    accuracy = int(pattern_match.group(2))
                    correct = int(pattern_match.group(3))
                    total = int(pattern_match.group(4))

                    reliability[pattern_type] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total
                    }

    return reliability
