#!/usr/bin/env python3
"""
Discord Integration Hooks (src/main_hooks/discord_hooks.py)
----------------------------------------------------------
Creates hook functions for sending pretraining results and analysis to Discord.

Functions:
  - create_pretraining_message_hook - Creates a function that sends pretraining results to Discord

Dependencies:
  - Discord client from src.discord.discord_client
  
Used by:
  - main.py for sending pretraining data to Discord
"""

import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


def create_pretraining_message_hook(discord_client, ticker):
    """
    Create a hook function that sends pretraining results to Discord

    Parameters:
    - discord_client: Discord client for sending messages
    - ticker: Stock symbol being analyzed (default, can be overridden by analysis)

    Returns:
    - Hook function that takes pretraining analysis and sends it to Discord
    """
    def discord_hook(analysis):
        """
        Hook function for pretraining to send messages to Discord

        Parameters:
        - analysis: Pretraining analysis result to send
        """
        # Ensure we have the Discord client
        if not discord_client:
            logger.warning(
                f"Discord client not available, skipping notification for {ticker}")
            return

        # Make sure we have an analysis object
        if not analysis or not isinstance(analysis, dict):
            logger.warning(
                f"Invalid analysis object for {ticker}, skipping notification")
            return

        try:
            # Get the ticker from the analysis or use the default
            analysis_ticker = analysis.get("ticker", ticker)

            # Log what we're processing
            logger.info(
                f"Processing pretraining hook for {analysis_ticker} (default ticker: {ticker})")

            # Extract key information from the analysis
            analysis_type = analysis.get(
                "analysis_type", analysis.get("type", "unknown"))
            date = analysis.get("date", datetime.now().strftime("%Y-%m-%d"))

            logger.info(f"Analysis type: {analysis_type}, date: {date}")

            # Handle different types of analyses
            if analysis_type == "summary" or "summary" in analysis_type:
                # Summary message
                logger.info(f"Sending summary for {analysis_ticker}")
                summary_text = analysis.get(
                    "full_summary", "No summary available")

                # Extract prediction horizons
                predictions = []
                for horizon in ["next_day", "next_week", "next_month"]:
                    prediction = analysis.get(f"{horizon}_prediction", {})
                    if prediction:
                        direction = prediction.get("direction", "neutral")
                        magnitude = prediction.get("magnitude", 0)
                        confidence = prediction.get("confidence", 0)

                        predictions.append(
                            f"{horizon.replace('_', ' ').title()}: {direction}, {magnitude}% change, {confidence}% confidence")

                prediction_text = "\n".join(
                    predictions) if predictions else "No specific predictions"

                # Send pretraining summary to Discord
                discord_client.send_pretraining_summary(
                    ticker=analysis_ticker,
                    date=date,
                    analysis_type="Final Summary",
                    summary=summary_text,
                    predictions=prediction_text
                )
                logger.info(
                    f"Sent pretraining summary for {analysis_ticker} to Discord")

            elif "reflection" in analysis_type:
                # Reflection message
                logger.info(f"Sending reflection for {analysis_ticker}")
                reflection_text = analysis.get(
                    "full_reflection", "No reflection available")

                # Extract key learnings for a more concise message
                key_learnings = []

                # Look for the numbered list at the end of reflection
                if "lessons_learned" in analysis and isinstance(analysis["lessons_learned"], list):
                    key_learnings = analysis["lessons_learned"]
                else:
                    # Try to extract from text
                    matches = re.findall(
                        r'\d+\.\s+(Key Learning|Prediction Accuracy|Quantified Error|Weight Adjustment|Credit Spread Strategy):\s*([^\n]+)', reflection_text)
                    for key, value in matches:
                        key_learnings.append(f"{key}: {value}")

                learning_text = "\n".join(
                    key_learnings) if key_learnings else "No key learnings extracted"

                # Create summary of prediction accuracy
                accuracy_info = ""
                if "prediction_accuracy" in analysis:
                    accuracy = 1 if analysis.get(
                        "correct_direction", False) else 0
                    error = analysis.get("magnitude_error", 0)
                    accuracy_info = f"Prediction was {'correct' if accuracy else 'incorrect'} with magnitude error of {error}%"

                # Send reflection to Discord
                discord_client.send_pretraining_reflection(
                    ticker=analysis_ticker,
                    date=date,
                    analysis_type="Reflection",
                    reflection=reflection_text,
                    key_learnings=learning_text,
                    accuracy_info=accuracy_info
                )
                logger.info(
                    f"Sent pretraining reflection for {analysis_ticker} to Discord")

            else:
                # Regular analysis message
                logger.info(f"Sending regular analysis for {analysis_ticker}")
                full_analysis = analysis.get(
                    "full_analysis", "No analysis available")

                # Extract key metrics
                trend = analysis.get("trend", "neutral")
                technical_score = analysis.get("technical_score", 0)
                sentiment_score = analysis.get("sentiment_score", 0)

                logger.debug(
                    f"Analysis metrics - trend: {trend}, technical: {technical_score}, sentiment: {sentiment_score}")

                # Extract movement prediction if available
                movement_prediction = analysis.get("movement_prediction", "")
                next_day_prediction = analysis.get("next_day_prediction", {})

                prediction_text = ""
                if next_day_prediction:
                    direction = next_day_prediction.get("direction", "neutral")
                    magnitude = next_day_prediction.get("magnitude", 0)
                    confidence = next_day_prediction.get("confidence", 0)
                    prediction_text = f"Prediction: {direction}, {magnitude}% change, {confidence}% confidence"
                elif movement_prediction:
                    prediction_text = f"Movement Prediction: {movement_prediction}"

                # Check if continuous learning context exists
                memory_context = None
                if "memory_context" in analysis:
                    memory_context = analysis["memory_context"]

                    # If it exists, create a summary of patterns found
                    if memory_context and isinstance(memory_context, dict):
                        patterns = memory_context.get(
                            "successful_patterns", [])
                        accuracy_stats = memory_context.get(
                            "prediction_accuracy", {}).get("next_day", {})

                        if patterns or accuracy_stats:
                            pattern_count = len(patterns)
                            accuracy = 0
                            if accuracy_stats:
                                total = accuracy_stats.get("total", 0)
                                correct = accuracy_stats.get("correct", 0)
                                if total > 0:
                                    accuracy = (correct / total) * 100

                            memory_text = f"Pattern Memory: {pattern_count} successful patterns recorded. Prediction accuracy: {accuracy:.1f}%"
                            prediction_text += f"\n{memory_text}"

                # Send analysis to Discord
                logger.info(
                    f"Sending pretraining analysis to Discord webhook for {analysis_ticker}")
                result = discord_client.send_pretraining_analysis(
                    ticker=analysis_ticker,
                    date=date,
                    analysis_type=analysis_type,
                    trend=trend,
                    technical_score=technical_score,
                    sentiment_score=sentiment_score,
                    prediction=prediction_text,
                    analysis=full_analysis
                )
                logger.info(f"Discord send result: {result}")

        except Exception as e:
            logger.error(f"Error sending pretraining message to Discord: {e}")
            # Log the full stack trace for better debugging
            logger.exception(e)

    return discord_hook
