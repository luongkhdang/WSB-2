"""
Pretraining prompt hooks for Gemini API.

This module provides functions for building prompts that support the staged,
reflective pretraining process for the AI Analyzer, allowing it to build a
deep understanding of market trends and stock behavior over time.
"""

from datetime import datetime, timedelta
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def get_pretraining_prompt(stock_data, market_context, previous_analysis=None, current_date=None, is_reflection=False, is_summary=False, outcome_context=None):
    """
    Generate a prompt for pretraining the AI Analyzer on historical stock data.

    Parameters:
    - stock_data: Dict containing the stock price and technical indicators for the date being analyzed
    - market_context: Dict with market trend analysis for the date being analyzed
    - previous_analysis: Optional dict containing the previous day's analysis and reflection (if any)
    - current_date: The date being analyzed in the pretraining sequence (string format)
    - is_reflection: Boolean indicating if this is a reflection prompt (after seeing the next day's data)
    - is_summary: Boolean indicating if this is a summary prompt for the entire pretraining sequence
    - outcome_context: Optional dict containing actual outcome data for reflection (price changes, accuracy metrics)

    Returns:
    - A prompt string for the AI Analyzer to analyze the stock data
    """
    # Validate input data to prevent silent failures
    if not stock_data or not isinstance(stock_data, dict):
        logger.warning(f"Invalid stock_data provided: {type(stock_data)}")
        stock_data = {}

    if not market_context or not isinstance(market_context, dict):
        logger.warning(
            f"Invalid market_context provided: {type(market_context)}")
        market_context = {"spy_trend": "neutral"}

    # Format the date for display
    date_str = current_date if current_date else datetime.now().strftime("%Y-%m-%d")

    if is_summary:
        return _build_summary_prompt(stock_data, market_context, date_str)
    elif is_reflection:
        return _build_reflection_prompt(stock_data, market_context, previous_analysis, date_str, outcome_context)
    else:
        return _build_analysis_prompt(stock_data, market_context, previous_analysis, date_str)


def safe_extract(value, default):
    """
    Safely extract a value, returning a default if the value is None.
    This helps prevent errors when building prompts.

    Parameters:
    - value: The value to extract
    - default: Default value to return if value is None

    Returns:
    - The value if not None, otherwise the default
    """
    if value is None:
        return default

    # Handle pandas Series
    if isinstance(value, pd.Series):
        if len(value) == 0:  # Check length instead of using .empty as a truth value
            return default
        return value.iloc[0]

    return value


def _build_analysis_prompt(stock_data, market_context, previous_analysis, date_str):
    """Build an analysis prompt for the pretraining phase with enhanced prediction capabilities."""

    # Extract ticker symbol
    ticker = safe_extract(stock_data.get('ticker'), 'the underlying')

    # Extract key technical data for summary
    current_price = safe_extract(stock_data.get('current_price'), 'unknown')
    open_price = safe_extract(stock_data.get('open'), 'unknown')
    high_price = safe_extract(stock_data.get('high'), 'unknown')
    low_price = safe_extract(stock_data.get('low'), 'unknown')
    volume = safe_extract(stock_data.get('volume'), 'unknown')
    avg_volume = safe_extract(stock_data.get('avg_volume'), 'unknown')

    # Technical indicators
    ema_9 = safe_extract(stock_data.get('ema_9'), 'unknown')
    ema_21 = safe_extract(stock_data.get('ema_21'), 'unknown')
    ema_50 = safe_extract(stock_data.get('ema_50'), 'unknown')
    ema_200 = safe_extract(stock_data.get('ema_200'), 'unknown')
    rsi = safe_extract(stock_data.get('rsi'), 'unknown')
    macd = safe_extract(stock_data.get('macd'), 'unknown')
    macd_signal = safe_extract(stock_data.get('macd_signal'), 'unknown')
    atr = safe_extract(stock_data.get('atr'), 'unknown')
    atr_percent = safe_extract(stock_data.get('atr_percent'), 'unknown')

    # Build technical summary
    technical_summary = f"""
    Technical Data Summary:
    - Current Price: {current_price}
    - OHLC: Open {open_price}, High {high_price}, Low {low_price}
    - Volume: {volume} (Avg: {avg_volume})
    - EMA 9/21: {ema_9}/{ema_21}
    - EMA 50/200: {ema_50}/{ema_200}
    - RSI: {rsi}
    - MACD: {macd} (Signal: {macd_signal})
    - ATR: {atr} ({atr_percent}% of price)
    """

    # Enhanced: Process previous analysis and build context
    previous_context = ""
    continuous_learning_context = ""
    memory_insights = ""

    if previous_analysis:
        # Handle case where previous_analysis might be a more complex continuous context object
        if isinstance(previous_analysis, dict) and 'previous_analysis' in previous_analysis:
            # Extract insights from the continuous context
            prev = previous_analysis.get('previous_analysis', {})
            reflection = previous_analysis.get('reflection', {})
            actual_outcome = previous_analysis.get('actual_outcome', {})
            memory = previous_analysis.get('memory', {})

            # Build enhanced context from previous analysis
            prev_trend = safe_extract(prev.get('trend'), 'neutral')
            prev_tech_score = safe_extract(prev.get('technical_score'), 0)
            prev_sent_score = safe_extract(prev.get('sentiment_score'), 0)
            prev_date = safe_extract(prev.get('date'), 'previous date')
            prev_movement = safe_extract(
                prev.get('movement_prediction'), 'unknown')

            # Extract reflection insights
            accuracy = "unknown"
            key_learning = "No specific learning available"
            weight_adjustment = "No adjustment specified"

            if reflection:
                accuracy = safe_extract(reflection.get(
                    'prediction_accuracy'), 'unknown')

                if isinstance(reflection.get('lessons_learned'), list) and reflection.get('lessons_learned'):
                    key_learning = reflection.get('lessons_learned')[0]

                weight_adjustment = safe_extract(reflection.get(
                    'weight_adjustment'), 'No adjustment specified')

            # Extract actual outcome
            actual_direction = "unknown"
            actual_change = 0

            if actual_outcome:
                actual_direction = safe_extract(
                    actual_outcome.get('actual_direction'), 'unknown')
                actual_change = safe_extract(
                    actual_outcome.get('actual_change_pct'), 0)

            # Build continuous learning context
            continuous_learning_context = f"""
            CONTINUOUS LEARNING CONTEXT:
            Previous Analysis ({prev_date}):
            - Trend: {prev_trend}
            - Technical Score: {prev_tech_score}
            - Predictability Score: {prev_sent_score} (formerly Sentiment Score)
            - Movement Prediction: {prev_movement}
            
            Actual Outcome:
            - Direction: {actual_direction}
            - Price Change: {actual_change:.2f}%
            
            Reflection Insights:
            - Prediction Accuracy: {accuracy}
            - Key Learning: {key_learning}
            - Weight Adjustment: {weight_adjustment}
            """

            # Add memory insights if available
            if memory and isinstance(memory, dict):
                # Extract successful patterns
                successful_patterns = memory.get('successful_patterns', [])
                if successful_patterns and len(successful_patterns) > 0:
                    most_recent = successful_patterns[-1]
                    pattern_tech = most_recent.get('technical_indicators', {})

                    memory_insights = f"""
                    PATTERN MEMORY:
                    Most recent successful pattern:
                    - Trend: {pattern_tech.get('trend', 'unknown')}
                    - Technical Score: {pattern_tech.get('technical_score', 0)}
                    - Predictability Score: {pattern_tech.get('sentiment_score', 0)}
                    - Market Alignment: {pattern_tech.get('market_alignment', 'unknown')}
                    
                    Similar current conditions should be given more weight in the analysis.
                    """

                # Extract prediction accuracy stats
                prediction_stats = memory.get(
                    'prediction_accuracy', {}).get('next_day', {})
                if prediction_stats:
                    total = prediction_stats.get('total', 0)
                    if total > 0:
                        correct = prediction_stats.get('correct', 0)
                        accuracy_pct = (correct / total) * 100
                        memory_insights += f"""
                        PREDICTION STATS:
                        - Next-day prediction accuracy: {accuracy_pct:.1f}% ({correct}/{total})
                        """
        else:
            # Handle simple previous analysis
            prev_date = safe_extract(
                previous_analysis.get('date'), 'previous date')
            prev_analysis_type = safe_extract(
                previous_analysis.get('analysis_type'), 'unknown')
            prev_trend = safe_extract(
                previous_analysis.get('trend'), 'neutral')
            prev_tech_score = safe_extract(
                previous_analysis.get('technical_score'), 0)
            prev_sent_score = safe_extract(
                previous_analysis.get('sentiment_score'), 0)

            previous_context = f"""
            Previous Analysis ({prev_date}, {prev_analysis_type}):
            - Trend: {prev_trend}
            - Technical Score: {prev_tech_score}
            - Predictability Score: {prev_sent_score} (formerly Sentiment Score)
            """

    # Use the most informative context available
    context_section = continuous_learning_context if continuous_learning_context else previous_context
    context_section += memory_insights

    return f'''
    PRETRAINING MODE: You are now analyzing historical stock data for {date_str}. 
    You only have data up to this date. DO NOT use knowledge of future market movements.
    
    Analyze the underlying stock with this data:
    
    Stock Data: {stock_data}
    Market Context: {market_context}
    Current Date for Analysis: {date_str}
    {technical_summary}
    {context_section}
    
    Follow these rules for your analysis:
    1. Price Trend Analysis - Look at multiple indicators:
       - Price vs EMAs: Price > 9/21 EMA is bullish (+10 to Technicals); Price < 9/21 EMA is bearish (+10 to Technicals if bearish setup)
       - RSI: 0-30 (oversold), 70-100 (overbought), 30-70 (neutral)
       - MACD: Cross above signal (bullish), below signal (bearish)
       - Volume: Above average (strong signal), below average (weak signal)
       
    2. Calculate Technical Score (0-30):
       - EMAs aligned and price above all: +10
       - EMAs aligned and price below all: +10 
       - RSI confirms trend: +5 
       - MACD confirms trend: +5
       - Key levels hold/break: +5
       - Volume confirms: +5
       
    3. Calculate Predictability Score (0-30) - THIS REPLACES SENTIMENT SCORE:
       - Clear chart pattern (triangle, flag, etc.): +10
       - Breakout potential from consolidation: +10
       - ATR % in normal range (average volatility): +5
       - Low ATR % (tight consolidation): +7
       - Strong support/resistance nearby: +8
       - Clear trend channel: +7
       - High-volume consolidation: +5
       - Low noise in price action: +5
       - Historical trend consistency: +5
       An ideal predictable setup would show:
       - Clear pattern formation
       - Recent consolidation with decreasing volatility 
       - Strong historical support/resistance levels
       - Price approaching key decision point
       - Consistent behavior at similar levels in the past
       
    4. Market Alignment:
       - Trend aligned with SPY (same direction): aligned
       - Trend opposite SPY: divergent
       - Leading SPY (moves before SPY): leading
       
    5. Trade Planning - For ALL recommendations:
       - Identify specific support/resistance levels for stop loss/target
       - Identify entry price and catalyst
       - Estimate move magnitude (based on ATR and prior moves)
       - Set confidence level based on pattern clarity and confirmations
       
    6. Prediction Requirements - High confidence = 70%+, provide:
       - Direction: Bullish, Bearish, or Neutral (with clear reasoning)
       - Magnitude: Expected % move in specific range (e.g., "1.2-1.5%" not "1-3%")
       - Timeframe: Next day and 7-14 day horizons with different confidence
       - Catalysts: Technical triggers to watch for
       - Invalidation: When to abandon the prediction
       
    7. Format your prediction precisely:
       Next day prediction: [Direction] move of [X.X-X.X]% with [XX]% confidence
       Next week prediction: [Direction] move of [X.X-X.X]% with [XX]% confidence
       
    Provide the following sections in your analysis:
    
    TECHNICAL ANALYSIS:
    [Your detailed technical analysis]
    
    PATTERN RECOGNITION:
    [Identify chart patterns and key levels]
    
    PREDICTION:
    [Clear, specific predictions as described above]
    
    SUMMARY:
    Trend: [bullish/bearish/neutral]
    Technical Score: [0-30]
    Predictability Score: [0-30] (based on pattern clarity, volatility conditions, and historical consistency)
    Total Score: [0-60]
    Market Alignment: [aligned/divergent/leading]
    
    Return your full analysis including all required sections.
    '''


def _build_reflection_prompt(stock_data, market_context, previous_analysis, date_str, outcome_context=None):
    """Build a reflection prompt for analyzing prediction accuracy."""

    # Extract previous analysis information
    if not previous_analysis or not isinstance(previous_analysis, dict):
        return f"Error: No previous analysis available for reflection on {date_str}."

    # Extract key information from previous analysis
    ticker = safe_extract(
        previous_analysis.get('ticker'), 'the stock')
    trend = safe_extract(
        previous_analysis.get('trend'), 'neutral')
    technical_score = safe_extract(
        previous_analysis.get('technical_score'), 0)
    # Use predictability score instead of sentiment score
    predictability_score = safe_extract(
        previous_analysis.get('sentiment_score'), 0)
    risk_assessment = safe_extract(
        previous_analysis.get('risk_assessment'), 'normal')
    market_alignment = safe_extract(
        previous_analysis.get('market_alignment'), 'neutral')
    
    # Get previous date from context
    previous_date = safe_extract(previous_analysis.get('date'), 'previous date')

    # Extract support/resistance levels
    support_levels = []
    resistance_levels = []
    if 'support_levels' in previous_analysis and isinstance(previous_analysis['support_levels'], list):
        support_levels = previous_analysis['support_levels']
    if 'resistance_levels' in previous_analysis and isinstance(previous_analysis['resistance_levels'], list):
        resistance_levels = previous_analysis['resistance_levels']

    # Format support/resistance levels for display
    support_text = "None identified"
    if support_levels:
        support_text = ", ".join(f"${level:.2f}" for level in support_levels)
    
    resistance_text = "None identified"
    if resistance_levels:
        resistance_text = ", ".join(f"${level:.2f}" for level in resistance_levels)
    
    # Get information about price changes
    price_change = safe_extract(stock_data.get('percent_change'), 0)
    current_price = safe_extract(stock_data.get('current_price'), 0)
    
    # Get previous and current ATR 
    previous_atr = safe_extract(previous_analysis.get('atr_percent'), 0)
    current_atr = safe_extract(stock_data.get('atr_percent'), 0)
    
    # Determine actual trend based on price change
    actual_trend = "neutral"
    if price_change > 1.0:
        actual_trend = "strongly bullish"
    elif price_change > 0.5:
        actual_trend = "bullish"
    elif price_change < -1.0:
        actual_trend = "strongly bearish"
    elif price_change < -0.5:
        actual_trend = "bearish"
    
    # Extract prediction text from previous analysis
    prediction_text = "Not available"
    if 'movement_prediction' in previous_analysis:
        prediction_text = previous_analysis['movement_prediction']
    elif 'next_day_prediction' in previous_analysis and isinstance(previous_analysis['next_day_prediction'], dict):
        next_day_pred = previous_analysis['next_day_prediction']
        direction = next_day_pred.get('direction', 'unknown')
        magnitude = next_day_pred.get('magnitude', 0)
        confidence = next_day_pred.get('confidence', 0)
        prediction_text = f"{direction} move of {magnitude}% with {confidence}% confidence"
    
    # Attempt to extract pattern information from previous analysis
    pattern_text = "No pattern analysis available"
    pattern_match = False
    pattern_recognition = ""
    
    # Search for pattern information in full analysis text
    if 'full_analysis' in previous_analysis:
        full_text = previous_analysis['full_analysis']
        pattern_section = ""
        
        # Look for pattern recognition section
        import re
        pattern_match = re.search(r'PATTERN RECOGNITION:\s*(.*?)(?:\n\n|\nPREDICTION:)', full_text, re.IGNORECASE | re.DOTALL)
        if pattern_match:
            pattern_recognition = pattern_match.group(1).strip()
            pattern_text = pattern_recognition
    
    # Get accuracy information from outcome context if available
    accuracy_info = ""
    if outcome_context and isinstance(outcome_context, dict):
        actual_direction = safe_extract(
            outcome_context.get('actual_direction'), 'unknown')
        predicted_direction = safe_extract(
            outcome_context.get('predicted_direction'), 'unknown')
        
        direction_correct = actual_direction == predicted_direction
        
        actual_change = safe_extract(
            outcome_context.get('actual_change_pct'), 0)
        predicted_change = safe_extract(
            outcome_context.get('predicted_change'), 0)
        
        if predicted_change > 0:
            magnitude_error = abs(actual_change - predicted_change)
            accuracy_info = f"""
            Prediction Analysis:
            - Predicted Direction: {predicted_direction.title()}
            - Actual Direction: {actual_direction.title()}
            - Direction Correct: {'Yes' if direction_correct else 'No'}
            - Predicted Change: {predicted_change:.2f}%
            - Actual Change: {actual_change:.2f}%
            - Magnitude Error: {magnitude_error:.2f}%
            """
        accuracy_info = ""

    # Calculate ATR change if available
    atr_change_text = ""
    if previous_atr > 0 and current_atr > 0:
        atr_percent_change = (
            (current_atr - previous_atr) / previous_atr) * 100
        atr_change_text = f"ATR Change: {atr_percent_change:.2f}%"

    # Format the reflection prompt
    return f'''
    ## Pretraining Reflection - {previous_date}/{date_str}
    
    This reflection analyzes the accuracy of my previous prediction for {previous_date}, given the actual outcome on {date_str}. 
    
    Previous Analysis:
    - Stock Trend: {trend}
    - Technical Score: {technical_score}
    - Predictability Score: {predictability_score} (formerly Sentiment Score)
    - Risk Assessment: {risk_assessment}
    - Market Alignment: {market_alignment}
    - Movement Prediction: {prediction_text}
    - Support Levels: {support_text}
    - Resistance Levels: {resistance_text}
    - Pattern Recognition: {pattern_text}
    
    Actual Outcome:
    - Price Change: {price_percent_change:.2f}%
    - Actual Trend: {actual_trend}
    - {atr_change_text}
    {accuracy_info}
    
    Based on this information, please provide a detailed reflection on the accuracy of the previous prediction, organized into these sections:
    
    **1. Trend Prediction Accuracy Assessment:**
    - How accurate was the directional prediction? Be specific with error measurements.
    - What was the error margin in percentage terms?
    - Would a stronger (less neutral) directional bias have been more accurate?
    - Given the market context, what would have been the most accurate prediction?
    - Which indicators were most reliable in this prediction?
    
    **2. Pattern & Volatility Assessment:**
    - Did you correctly identify chart patterns? If not, what patterns were actually forming?
    - Were support/resistance levels correctly identified and respected?
    - Did you correctly predict volatility changes?
    - What factors contributed to the volatility shift?
    - What specific indicators would have better predicted this volatility change?
    - How did the actual price action compare to the predicted pattern behavior?
    
    **3. Narrowing Prediction Ranges:**
    - Was the prediction range too wide (e.g., 1-3% instead of 1.2-1.5%)?
    - What techniques could narrow future prediction ranges while maintaining accuracy?
    - Should confidence levels be adjusted higher or lower based on this outcome?
    - What minimum confidence threshold should be applied for actionable predictions?
    
    **4. Technical Analysis Review:**
    - Which technical indicators were most helpful?
    - Which were misleading?
    - How did price action around support/resistance levels affect the outcome?
    - How did price action around EMAs affect the outcome?
    - Which indicators should receive more weight in future analyses?
    
    **5. Market Alignment Impact:**
    - How did the broader market trend affect this stock?
    - Was alignment or divergence from SPY more impactful?
    - Would overall market analysis have improved this prediction?
    - Did the stock lead or lag market movements?
    
    **6. Options Implications:**
    - Would the analysis have correctly positioned credit spreads?
    - What would have been the optimal strike selection?
    - How would a 7-15 DTE credit spread strategy have performed?
    - How would this movement have affected premiums and probability of profit?
    
    After your detailed reflection, provide a structured summary with EXACTLY these numbered points:
    
    1. Prediction Accuracy: [high|moderate|low] - with percentage accuracy
    2. Quantified Error: [specific numeric error values]
    3. Key Learning: [most important insight - be specific and actionable]
    4. Weight Adjustment: [how to adjust technical vs. predictability vs. market alignment weights]
    5. Bias Adjustment: [less neutral, more directional when indicators support it]
    6. Pattern Recognition Improvement: [specific pattern identification technique]
    7. Credit Spread Strategy Refinement: [specific recommendation for credit spreads]
    '''


def generate_pretraining_context(pretraining_results):
    """
    Generate a comprehensive context string based on all pretraining steps with enhanced statistical insights.

    Parameters:
    - pretraining_results: List of dicts containing results from each pretraining step

    Returns:
    - A formatted string summarizing the pretraining insights with focus on practical trading application
    """
    if not pretraining_results:
        return "No pretraining data available."

    # Helper function to safely extract values
    def safe_extract(value, default):
        """Safely extract values from potentially pandas Series objects"""
        if value is None:
            return default
        # Handle pandas Series - fixed to avoid truth value comparison
        if isinstance(value, pd.Series):
            if len(value) == 0:  # Check length instead of using .empty as a truth value
                return default
            return value.iloc[0]
        return value

    # Build a summary of all pretraining steps
    context = "PRETRAINING INSIGHTS:\n\n"

    # Tracking for statistical analysis
    accuracy_scores = []
    error_magnitudes = []
    key_learnings = []
    trend_predictions = {"correct": 0, "close": 0, "incorrect": 0}

    for i, result in enumerate(pretraining_results):
        # Process each result, safely extracting values
        date = safe_extract(result.get('date', f"Step {i+1}"), f"Step {i+1}")
        analysis_type = safe_extract(result.get(
            'analysis_type', 'Unknown'), 'Unknown')

        if analysis_type.startswith('reflection'):
            accuracy = safe_extract(result.get(
                'accuracy', 'Unknown'), 'Unknown')
            accuracy_score = safe_extract(result.get('accuracy_score', 0), 0)
            accuracy_scores.append(accuracy_score)

            key_learning = safe_extract(result.get(
                'key_learning', 'No learning recorded'), 'No learning recorded')
            # Validate learning content
            if key_learning and len(str(key_learning)) > 5:
                key_learnings.append(key_learning)

            # Extract quantified error if available
            error_data = safe_extract(result.get('quantified_error', ''), '')
            if error_data:
                context += f"{i+1}. {date} ({analysis_type}) - Accuracy: {accuracy}\n"
                context += f"   Error Assessment: {error_data}\n"
                context += f"   Learning: {key_learning}\n\n"

                # Try to extract numeric error for statistical analysis
                try:
                    error_parts = str(error_data).lower().split('by ')
                    if len(error_parts) > 1:
                        error_part = error_parts[1]
                        numeric_error = ''.join(
                            c for c in error_part if c.isdigit() or c == '.')
                        if numeric_error:
                            error_magnitudes.append(float(numeric_error))
                except Exception as e:
                    logger.debug(f"Could not extract numeric error: {e}")
            else:
                context += f"{i+1}. {date} ({analysis_type}) - Accuracy: {accuracy}\n"
                context += f"   Learning: {key_learning}\n\n"

    # Add statistical analysis and conclusions
    context += "\nCONCLUSIONS FROM PRETRAINING:\n"

    # Calculate overall accuracy with error handling
    if accuracy_scores:
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        context += f"- Overall prediction accuracy: {avg_accuracy:.1f}/10\n"

    # Calculate mean error magnitude if available
    if error_magnitudes:
        mean_error = sum(error_magnitudes) / len(error_magnitudes)
        context += f"- Mean prediction error magnitude: {mean_error:.2f}%\n"

    # Trend prediction accuracy if available
    total_predictions = sum(trend_predictions.values())
    if total_predictions > 0:
        correct_pct = (trend_predictions["correct"] / total_predictions) * 100
        context += f"- Directional accuracy: {correct_pct:.1f}%\n"

    # Extract key learnings with deduplication
    unique_learnings = []
    for learning in key_learnings:
        is_duplicate = False
        learning_str = str(learning)  # Ensure string format
        for existing in unique_learnings:
            existing_str = str(existing)  # Ensure string format
            if learning_str.lower() in existing_str.lower() or existing_str.lower() in learning_str.lower():
                is_duplicate = True
                break
        if not is_duplicate:
            unique_learnings.append(learning_str)

    if unique_learnings:
        context += "- Key insights:\n"
        for learning in unique_learnings[-5:]:  # Show last 5 unique learnings
            context += f"  * {learning}\n"

    # Add credit spread strategy implications
    context += "\nCREDIT SPREAD STRATEGY IMPLICATIONS:\n"

    # Extract credit spread insights from reflections
    spread_insights = []
    for r in pretraining_results:
        insight = safe_extract(
            r.get('credit_spread_strategy_refinement', ''), '')
        if insight and len(str(insight)) > 0:
            spread_insights.append(str(insight))

    if spread_insights:
        for i, insight in enumerate(spread_insights[-3:]):  # Last 3 insights
            context += f"{i+1}. {insight}\n"
    else:
        context += "Not enough data to provide specific credit spread strategy insights.\n"

    # Add weight adjustments from the most recent reflection
    if pretraining_results:
        latest_result = pretraining_results[-1]
        weight_adjustment = safe_extract(
            latest_result.get('weight_adjustment', ''), '')
        if weight_adjustment:
            context += f"\nRECOMMENDED WEIGHT ADJUSTMENTS:\n{weight_adjustment}\n"

    return context


def _build_summary_prompt(stock_data, market_context, date_str):
    """Build a summary prompt for the entire pretraining sequence."""

    # Extract summary information
    ticker = "Unknown"
    analyses_count = 0
    memory_context = None
    prediction_horizons = ["next_day", "next_week", "next_month"]

    # For summary mode, stock_data is actually summary_context
    if isinstance(stock_data, dict):
        ticker = safe_extract(stock_data.get('ticker'), 'the stock')
        start_date = safe_extract(stock_data.get('start_date'), 'start date')
        end_date = safe_extract(stock_data.get('end_date'), date_str)
        analyses_count = safe_extract(stock_data.get('analyses_count'), 0)
        recent_analyses = safe_extract(stock_data.get('recent_analyses'), [])
        prediction_horizons = safe_extract(stock_data.get('prediction_horizons'), [
                                           "next_day", "next_week", "next_month"])
        memory_context = safe_extract(stock_data.get('memory'), None)

    # Build a summary of recent analyses
    recent_summary = ""
    if isinstance(recent_analyses, list) and recent_analyses:
        for i, analysis in enumerate(recent_analyses):
            if isinstance(analysis, dict):
                analysis_type = safe_extract(
                    analysis.get('analysis_type', ''), 'unknown')
                analysis_date = safe_extract(
                    analysis.get('date', ''), 'unknown date')
                trend = safe_extract(analysis.get(
                    'trend', ''), 'unknown trend')

                recent_summary += f"Analysis {i+1} ({analysis_type} on {analysis_date}): {trend}\n"

    # Add memory context for successful patterns if available
    successful_patterns_section = ""
    lessons_learned_section = ""
    weight_adjustments_section = ""
    prediction_accuracy_section = ""

    if memory_context and isinstance(memory_context, dict):
        # Add successful patterns
        successful_patterns = memory_context.get('successful_patterns', [])
        if successful_patterns:
            successful_patterns_section = "SUCCESSFUL PREDICTION PATTERNS:\n"
            # Show last 3 patterns
            for i, pattern in enumerate(successful_patterns[-3:]):
                pattern_date = pattern.get('date', 'unknown')
                tech_indicators = pattern.get('technical_indicators', {})
                trend = tech_indicators.get('trend', 'unknown')
                tech_score = tech_indicators.get('technical_score', 0)
                sentiment_score = tech_indicators.get('sentiment_score', 0)
                market_alignment = tech_indicators.get(
                    'market_alignment', 'unknown')

                prediction = pattern.get('prediction', {})
                pred_direction = prediction.get('direction', 'unknown')
                pred_magnitude = prediction.get('magnitude', 0)

                outcome = pattern.get('outcome', {})
                actual_change = outcome.get('actual_change_pct', 0)

                successful_patterns_section += f"{i+1}. Date: {pattern_date}, Setup: {trend} trend, Tech:{tech_score}, Sentiment:{sentiment_score}, Market:{market_alignment}\n"
                successful_patterns_section += f"   Predicted: {pred_direction} direction, {pred_magnitude}% magnitude, Actual: {actual_change:.2f}%\n"

        # Add lessons learned
        lessons_learned = memory_context.get('lessons_learned', [])
        if lessons_learned:
            lessons_learned_section = "KEY LESSONS LEARNED:\n"
            unique_lessons = set()
            for lesson in lessons_learned:
                if isinstance(lesson, str) and lesson not in unique_lessons and len(unique_lessons) < 5:
                    unique_lessons.add(lesson)
                    lessons_learned_section += f"• {lesson}\n"

        # Add weight adjustments
        weight_adjustments = memory_context.get('weight_adjustments', [])
        if weight_adjustments:
            weight_adjustments_section = "WEIGHT ADJUSTMENT HISTORY:\n"
            # Show last 3 adjustments
            for i, adjustment in enumerate(weight_adjustments[-3:]):
                if isinstance(adjustment, str):
                    weight_adjustments_section += f"{i+1}. {adjustment}\n"

        # Add prediction accuracy statistics
        prediction_accuracy = memory_context.get('prediction_accuracy', {})
        if prediction_accuracy:
            prediction_accuracy_section = "PREDICTION ACCURACY STATISTICS:\n"
            for horizon, stats in prediction_accuracy.items():
                total = stats.get('total', 0)
                if total > 0:
                    correct = stats.get('correct', 0)
                    accuracy = (correct / total) * 100
                    errors = stats.get('errors', [])
                    avg_error = sum(errors) / len(errors) if errors else 0

                    prediction_accuracy_section += f"{horizon.replace('_', ' ').title()}: {accuracy:.1f}% accuracy ({correct}/{total}), Avg Error: {avg_error:.2f}%\n"

    # Build the complete memory-enhanced summary prompt
    memory_section = ""
    if successful_patterns_section or lessons_learned_section or weight_adjustments_section or prediction_accuracy_section:
        memory_section = f"""
        LEARNING MEMORY CONTEXT:
        {successful_patterns_section}
        {lessons_learned_section}
        {weight_adjustments_section}
        {prediction_accuracy_section}
        
        Use this learning memory to inform your final analysis and improve future predictions.
        """

    return f'''
    PRETRAINING SUMMARY MODE: You are now generating a final summary and multi-timeframe forecast
    based on the entire pretraining sequence for {ticker}.
    
    Pretraining Period: {start_date} to {end_date}
    Total Analyses: {analyses_count}
    
    Recent Analysis Summary:
    {recent_summary}
    {memory_section}
    
    Based on all the analyses and reflections from the pretraining period, please provide:
    
    1. Key Patterns Overview:
       - Summarize the most consistent technical patterns observed
       - Identify which indicators provided the most reliable signals
       - Highlight any correlations between market conditions and stock movement
    
    2. Volatility Profile:
       - Characterize the stock's typical volatility range
       - Identify conditions that preceded volatility changes
       - Suggest optimal position sizing based on historical volatility
    
    3. Market Relationship:
       - How strongly does this stock correlate with broad market moves?
       - Does it tend to lead or lag market trends?
       - Any sectors or other stocks that show strong correlation?
    
    4. Multi-Timeframe Forecast:
       For each of the following timeframes, provide a detailed prediction with:
       - Direction (bullish/bearish/neutral)
       - Magnitude (expected % range)
       - Confidence level (0-100%)
       - Key catalysts or triggers to watch
       - Risk factors that could invalidate the forecast
    
    Your forecast MUST include separate sections for each of these horizons:
    {', '.join(prediction_horizons)}
    
    5. Trading Strategy Recommendations:
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
    '''
