#!/usr/bin/env python3
"""
Discord Integration Client (src/discord/discord_client.py)
---------------------------------------------------------
Handles sending notifications and reports to Discord webhooks.

Class:
  - DiscordClient - Client for sending formatted messages to Discord

Methods:
  - send_message - Base method for sending Discord messages
  - send_trade_alert - Sends trading alerts and signals
  - send_market_analysis - Sends market condition reports
  - send_analysis - Sends stock analysis results
  - send_pretraining_* - Methods for sending pretraining data
  - plus various other specialized message formats

Dependencies:
  - requests - For HTTP communication with Discord API
  - python-dotenv - For loading webhook URLs from .env file
  - Environment variables (DISCORD_WEBHOOK_URL_*) for webhook endpoints

Used by:
  - main.py for sending notifications about analysis results
  - discord_hooks.py for sending pretraining data
"""

import os
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# !IMPORTANT: THE RATE LIMIT IS 30 REQUESTS PER MINUTE. RETRY EVERY 10 SECONDS ONLY IF YOU GET A 429 ERROR (RATE LIMIT ERROR).


class DiscordClient:
    """
    Client for sending messages to Discord webhooks.
    """

    def __init__(self):
        # Load webhook URLs from environment variables
        self.webhook_urls = {
            'default': os.getenv('DISCORD_WEBHOOK_URL'),
            'trade_alerts': os.getenv('DISCORD_WEBHOOK_URL_TRADE_ALERTS'),
            'market_analysis': os.getenv('DISCORD_WEBHOOK_URL_MARKET_ANALYSIS'),
            'journal': os.getenv('DISCORD_WEBHOOK_URL_JOURNAL'),
            'backtest': os.getenv('DISCORD_WEBHOOK_URL_BACKTEST'),
            'trade-stats': os.getenv('DISCORD_WEBHOOK_URL_TRADE_STATS'),
            'trade-opinion': os.getenv('DISCORD_WEBHOOK_URL_TRADE_OPINION'),
            'full-analysis': os.getenv('DISCORD_WEBHOOK_URL_FULL_ANALYSIS'),
            'pretraining': os.getenv('DISCORD_WEBHOOK_URL_PRETRAINING'),
            'error-alerts': os.getenv('DISCORD_WEBHOOK_URL_ERROR_ALERTS')
        }

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def send_message(self,
                     content: str,
                     webhook_type: str = 'default',
                     username: Optional[str] = None,
                     embed_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to a Discord webhook.

        Args:
            content: The message content
            webhook_type: Type of webhook to use ('default', 'trade_alerts', 'market_analysis', 'journal', 'backtest')
            username: Optional custom username for the webhook
            embed_data: Optional embed data for rich content

        Returns:
            bool: True if successful, False otherwise
        """
        webhook_url = self.webhook_urls.get(webhook_type)

        if not webhook_url:
            self.logger.error(f"No webhook URL found for type: {webhook_type}")
            return False

        payload = {'content': content}

        if username:
            payload['username'] = username

        if embed_data:
            payload['embeds'] = [embed_data]

        try:
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 204:
                self.logger.info(
                    f"Message sent successfully to {webhook_type} webhook")
                return True
            else:
                self.logger.error(
                    f"Failed to send message: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending Discord message: {str(e)}")
            return False

    def send_trade_alert(self,
                         ticker: str,
                         action: str,
                         price: float,
                         notes: Optional[str] = None) -> bool:
        """
        Send a formatted trade alert message

        Args:
            ticker: Stock/crypto ticker symbol
            action: Action taken (BUY/SELL)
            price: Current price at alert
            notes: Optional additional notes

        Returns:
            bool: True if successful, False otherwise
        """
        action = action.upper()
        color = 0x00FF00 if action == "BUY" else 0xFF0000  # Green for buy, red for sell

        embed = {
            "title": f"{action} ALERT: {ticker}",
            "description": f"Action: {action}\nPrice: ${price:,.2f}",
            "color": color,
            "footer": {"text": "WSB-2 Trading Bot"}
        }

        if notes:
            embed["fields"] = [{"name": "Notes", "value": notes}]

        return self.send_message(
            content=f"New {action} alert for {ticker}",
            webhook_type="trade_alerts",
            embed_data=embed
        )

    def send_market_analysis(self,
                             title: str,
                             content: str,
                             metrics: Optional[Dict[str, Union[str, float, int]]] = None) -> bool:
        """
        Send market analysis information

        Args:
            title: Analysis title
            content: Analysis content (can be long, will be split if needed)
            metrics: Optional dictionary of metrics to include

        Returns:
            bool: True if successful, False otherwise
        """
        # Discord has a 2000 character limit per message
        # Discord embed description limit is 4096 characters
        # We'll use 2000 as a safe limit for embed descriptions
        MAX_EMBED_DESCRIPTION_LENGTH = 2000

        # Check if we need to split the content
        if content and len(content) > MAX_EMBED_DESCRIPTION_LENGTH:
            self.logger.info(
                f"Content length exceeds Discord limit ({len(content)} chars). Splitting into chunks.")

            # First message with metrics and title
            first_chunk = content[:MAX_EMBED_DESCRIPTION_LENGTH]
            remaining_content = content[MAX_EMBED_DESCRIPTION_LENGTH:]

            embed = {
                "title": title,
                "description": first_chunk,
                "color": 0x0099FF,
                "footer": {"text": "WSB-2 Market Analysis (Part 1)"}
            }

            if metrics:
                field_list = []
                for key, value in metrics.items():
                    field_list.append({
                        "name": key,
                        "value": str(value),
                        "inline": True
                    })
                embed["fields"] = field_list

            # Send first part
            success = self.send_message(
                content=(
                    "--------------------------------------------------------------------\n"
                    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n"
                    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n"
                    f"📊 **Market Analysis: {title} (Part 1)**\n"
                    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n"
                    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n"
                    "--------------------------------------------------------------------\n"
                ),
                webhook_type="market_analysis",
                embed_data=embed
            )

            if not success:
                self.logger.error(
                    "Failed to send first part of market analysis")
                return False

            # Split remaining content into chunks and send as follow-up messages
            part_number = 2
            while remaining_content:
                chunk = remaining_content[:MAX_EMBED_DESCRIPTION_LENGTH]
                remaining_content = remaining_content[MAX_EMBED_DESCRIPTION_LENGTH:]

                embed = {
                    "description": chunk,
                    "color": 0x0099FF,
                    "footer": {"text": f"WSB-2 Market Analysis (Part {part_number})"}
                }

                success = self.send_message(
                    content=f"📊 **Market Analysis: {title} (Part {part_number})**\n",
                    webhook_type="market_analysis",
                    embed_data=embed
                )

                if not success:
                    self.logger.error(
                        f"Failed to send part {part_number} of market analysis")
                    return False

                part_number += 1

            return True

        else:
            # Original behavior for content within limits
            embed = {
                "title": title,
                "description": content,
                "color": 0x0099FF,
                "footer": {"text": "WSB-2 Market Analysis"}
            }

            if metrics:
                field_list = []
                for key, value in metrics.items():
                    field_list.append({
                        "name": key,
                        "value": str(value),
                        "inline": True
                    })
                embed["fields"] = field_list

            return self.send_message(
                content=(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 **Market Analysis: {title}**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                webhook_type="market_analysis",
                embed_data=embed
            )

    def send_journal_entry(self,
                           title: str,
                           content: str) -> bool:
        """
        Send a journal entry to Discord

        Args:
            title: Entry title
            content: Entry content

        Returns:
            bool: True if successful, False otherwise
        """
        return self.send_message(
            content=f"# {title}\n\n{content}",
            webhook_type="journal"
        )

    def send_backtest_results(self,
                              strategy_name: str,
                              performance_metrics: Dict[str, Any],
                              timeframe: str) -> bool:
        """
        Send backtest results to Discord

        Args:
            strategy_name: Name of the strategy
            performance_metrics: Dictionary of performance metrics
            timeframe: Backtest timeframe

        Returns:
            bool: True if successful, False otherwise
        """
        # Format metrics into fields
        fields = []
        for key, value in performance_metrics.items():
            fields.append({
                "name": key,
                "value": f"{value}",
                "inline": True
            })

        embed = {
            "title": f"Backtest Results: {strategy_name}",
            "description": f"Timeframe: {timeframe}",
            "fields": fields,
            "color": 0x9B59B6,
            "footer": {"text": "WSB-2 Backtest Engine"}
        }

        return self.send_message(
            content=f"New backtest results for {strategy_name}",
            webhook_type="backtest",
            embed_data=embed
        )

    def send_analysis(self, analysis: Dict[str, Any], ticker: str = 'UNKNOWN', title: str = None) -> bool:
        """
        Send stock analysis to Discord channel

        Args:
            analysis: Dictionary containing analysis data
            ticker: Stock ticker symbol
            title: Optional custom title for the analysis embed

        Returns:
            bool: True if successful, False otherwise
        """
        # Prepare the embed content with analysis details
        trend = analysis.get('trend', 'neutral').upper()
        technical_score = analysis.get('technical_score', 0)
        sentiment_score = analysis.get('sentiment_score', 0)
        risk_assessment = analysis.get('risk_assessment', 'normal').upper()
        market_alignment = analysis.get('market_alignment', 'neutral').upper()

        # Set color based on trend
        color = 0x00FF00 if trend == "BULLISH" else 0xFF0000 if trend == "BEARISH" else 0xFFAA00

        # Create description from full analysis if available
        description = analysis.get('full_analysis', '')

        # Discord embed description limit (max 4096, but we'll use 2000 to be safe)
        MAX_EMBED_DESCRIPTION_LENGTH = 2000

        # Add price information for the title if no custom title is provided
        if title is None:
            title = f"{ticker} Stock Analysis"
            if 'raw_data' in analysis and 'current_price' in analysis['raw_data']:
                price = analysis['raw_data']['current_price']
                if price:
                    title = f"{ticker} Stock Analysis - ${price:.2f}"

        # Create fields list for the trade stats message
        fields = [
            {"name": "Trend", "value": trend, "inline": True},
            {"name": "Technical Score", "value": str(
                technical_score), "inline": True},
            {"name": "Sentiment Score", "value": str(
                sentiment_score), "inline": True},
            {"name": "Risk", "value": risk_assessment, "inline": True},
            {"name": "Market Alignment", "value": market_alignment, "inline": True}
        ]

        # Add options analysis if available
        options_analysis = analysis.get('options_analysis', '')
        if options_analysis:
            if len(options_analysis) > 500:
                options_analysis = options_analysis[:500] + "..."
            fields.append({"name": "Options Analysis",
                          "value": options_analysis, "inline": False})

        # 1. Extract ANALYZER FULL OPINION section if present
        analyzer_opinion = ""
        # First, try to find if the section exists
        if "ANALYZER FULL OPINION:" in description:
            # Get everything after "ANALYZER FULL OPINION:"
            opinion_start = description.find("ANALYZER FULL OPINION:")
            if opinion_start != -1:
                # Extract from the section header to the end or until another major section
                opinion_text = description[opinion_start +
                                           len("ANALYZER FULL OPINION:"):]
                # Trim leading/trailing whitespace
                analyzer_opinion = opinion_text.strip()

                # Log what we found for debugging
                self.logger.info(
                    f"Found ANALYZER FULL OPINION section for {ticker}, length: {len(analyzer_opinion)} chars")
                self.logger.info(f"First 100 chars: {analyzer_opinion[:100]}")

        # If opinion found, send it to trade-opinion webhook
        if analyzer_opinion:
            self.logger.info(
                f"Sending ANALYZER FULL OPINION to trade-opinion webhook for {ticker}")
            opinion_embed = {
                "title": f"{ticker} Trading Opinion",
                "description": analyzer_opinion[:MAX_EMBED_DESCRIPTION_LENGTH],
                "color": color,
                "footer": {"text": f"WSB-2 Trading Opinion • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            }

            self.send_message(
                content=(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💡 **Trading Opinion for {ticker}**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                webhook_type="trade-opinion",
                embed_data=opinion_embed
            )

        # 2. Send stats to trade-stats webhook
        self.logger.info(f"Sending stats to trade-stats webhook for {ticker}")
        stats_embed = {
            "title": f"{ticker} Stats Summary",
            "color": color,
            "fields": fields,
            "footer": {"text": f"WSB-2 Trading Stats • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        self.send_message(
            content=(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 **Trading Stats for {ticker}**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            webhook_type="trade-stats",
            embed_data=stats_embed
        )

        # 3. Send full analysis to full-analysis webhook
        success = True

        # Check if we need to split the content
        if description and len(description) > MAX_EMBED_DESCRIPTION_LENGTH:
            self.logger.info(
                f"Analysis description exceeds Discord limit ({len(description)} chars). Splitting into parts.")

            # First message with summary
            first_chunk = description[:MAX_EMBED_DESCRIPTION_LENGTH]
            remaining_content = description[MAX_EMBED_DESCRIPTION_LENGTH:]

            # Create the main embed with the first chunk
            embed = {
                "title": title,
                "description": first_chunk,
                "color": color,
                "footer": {"text": f"WSB-2 Full Analysis (Part 1) • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            }

            # Send first part with all the fields and metadata
            success = self.send_message(
                content=(
                    "--------------------------------------------------------------------\n"
                    f"📈 **Full Analysis for {ticker} (Part 1)**\n"
                    "--------------------------------------------------------------------\n"
                ),
                webhook_type="full-analysis",
                embed_data=embed
            )

            if not success:
                self.logger.error(
                    "Failed to send first part of stock analysis")
                return False

            # Split remaining content into chunks and send as follow-up messages
            part_number = 2
            while remaining_content:
                chunk = remaining_content[:MAX_EMBED_DESCRIPTION_LENGTH]
                remaining_content = remaining_content[MAX_EMBED_DESCRIPTION_LENGTH:]

                embed = {
                    "title": f"{ticker} Analysis Continued",
                    "description": chunk,
                    "color": color,
                    "footer": {"text": f"WSB-2 Full Analysis (Part {part_number}) • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
                }

                success = self.send_message(
                    content=f"📈 **Full Analysis for {ticker} (Part {part_number})**\n",
                    webhook_type="full-analysis",
                    embed_data=embed
                )
                if not success:
                    self.logger.error(
                        f"Failed to send part {part_number} of stock analysis")
                    return False

                part_number += 1
        else:
            # Original behavior for content within limits
            embed = {
                "title": title,
                "description": description if len(description) <= MAX_EMBED_DESCRIPTION_LENGTH else description[:MAX_EMBED_DESCRIPTION_LENGTH] + "...",
                "color": color,
                "footer": {"text": f"WSB-2 Full Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            }

            success = self.send_message(
                content=(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📈 **Full Analysis for {ticker}**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                webhook_type="full-analysis",
                embed_data=embed
            )

        return success

    def send_pretraining(self, ticker: str, start_date: str, end_date: str, data_points: int, pretraining_time: float = None, summary: str = None) -> bool:
        """
        Send pretraining results to Discord pretraining webhook

        Args:
            ticker: Stock ticker symbol
            start_date: Start date of pretraining data
            end_date: End date of pretraining data
            data_points: Number of historical data points analyzed
            pretraining_time: Optional time taken for pretraining in seconds
            summary: Optional summary of pretraining insights

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Preparing pretraining notification for {ticker}")

        # Verify inputs
        if not ticker:
            self.logger.error("Missing ticker in send_pretraining call")
            return False

        # Prepare fields for the embed
        fields = [
            {"name": "Ticker", "value": ticker, "inline": True},
            {"name": "Date Range", "value": f"{start_date} to {end_date}", "inline": True},
            {"name": "Data Points", "value": str(data_points), "inline": True}
        ]

        if pretraining_time is not None:
            fields.append({
                "name": "Processing Time",
                "value": f"{pretraining_time:.2f} seconds",
                "inline": True
            })

        # Create description
        description = "Historical data analysis complete."
        if summary:
            description = summary[:2000]  # Limit to 2000 chars for safety

        # Create embed
        embed = {
            "title": f"Pretraining Completed: {ticker}",
            "description": description,
            "color": 0x9370DB,  # Medium purple color
            "fields": fields,
            "footer": {"text": f"WSB-2 Pretraining • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        self.logger.info(
            f"Sending pretraining notification for {ticker} to Discord webhook")
        result = self.send_message(
            content=(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 **Pretraining Completed for {ticker}**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            webhook_type="pretraining",
            embed_data=embed
        )

        if result:
            self.logger.info(
                f"Successfully sent pretraining notification for {ticker}")
        else:
            self.logger.error(
                f"Failed to send pretraining notification for {ticker}")

        return result

    def send_pretraining_analysis_notification(self, ticker: str, insights: dict, accuracy: float, stock_analysis: dict = None) -> bool:
        """
        Send notification that pretraining data is being used for analysis

        Args:
            ticker: Stock ticker symbol
            insights: Dictionary of pretraining insights
            accuracy: Pretraining prediction accuracy score
            stock_analysis: Optional stock analysis results

        Returns:
            bool: True if successful, False otherwise
        """
        # Prepare fields for the embed
        fields = [
            {"name": "Ticker", "value": ticker, "inline": True},
            {"name": "Prediction Accuracy",
                "value": f"{accuracy:.1f}/10", "inline": True}
        ]

        # Add key learnings
        key_learnings = insights.get("key_learnings", [])
        if key_learnings:
            learning_text = "\n".join(
                [f"• {learning}" for learning in key_learnings[:3]])
            fields.append({
                "name": "Key Learnings Applied",
                "value": learning_text[:1024] if learning_text else "None identified",
                "inline": False
            })

        # Add specific adjustments
        adjustments = []
        if insights.get("use_momentum"):
            adjustments.append("Added momentum analysis")
        if insights.get("bias_bullish"):
            adjustments.append("Increased weight on positive signals")
        if insights.get("more_technicals"):
            adjustments.append("Using expanded technical indicators")

        if adjustments:
            fields.append({
                "name": "Pretraining Adjustments",
                "value": "\n".join([f"• {adj}" for adj in adjustments]),
                "inline": False
            })

        # Add stock analysis summary if available
        if stock_analysis:
            stock_summary = f"**Trend**: {stock_analysis.get('trend', 'Unknown')}\n"
            stock_summary += f"**Technical Score**: {stock_analysis.get('technical_score', 0)}\n"
            stock_summary += f"**Market Alignment**: {stock_analysis.get('market_alignment', 'Unknown')}"

            fields.append({
                "name": "Analysis Result",
                "value": stock_summary,
                "inline": False
            })

        # Create description
        description = f"Using pretrained AI insights for {ticker} analysis. The pretraining model has been tuned based on historical performance and learned patterns."

        # Create embed
        embed = {
            "title": f"🧠 Pretraining Applied: {ticker}",
            "description": description,
            "color": 0x9370DB,  # Medium purple color
            "fields": fields,
            "footer": {"text": f"WSB-2 AI Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        return self.send_message(
            content=(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 **Using Pretrained AI Insights for {ticker}**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            webhook_type="pretraining",
            embed_data=embed
        )

    def send_pretraining_analysis(self, ticker: str, date: str, analysis_type: str, trend: str,
                                  technical_score: int, sentiment_score: int, prediction: str, analysis: str) -> bool:
        """
        Send pretraining analysis to Discord pretraining webhook

        Args:
            ticker: Stock ticker symbol
            date: Date of analysis
            analysis_type: Type of analysis (e.g., intraday, daily)
            trend: Stock trend (bullish/bearish/neutral)
            technical_score: Technical score from analysis
            sentiment_score: Sentiment score from analysis
            prediction: Prediction text
            analysis: Full analysis text

        Returns:
            bool: True if successful, False otherwise
        """
        # Prepare fields for the embed
        fields = [
            {"name": "Ticker", "value": ticker, "inline": True},
            {"name": "Date", "value": date, "inline": True},
            {"name": "Analysis Type", "value": analysis_type, "inline": True},
            {"name": "Trend", "value": trend, "inline": True},
            {"name": "Technical Score", "value": str(
                technical_score), "inline": True},
            {"name": "Sentiment Score", "value": str(
                sentiment_score), "inline": True}
        ]

        # Add prediction if available
        if prediction:
            fields.append({
                "name": "Prediction",
                "value": prediction[:1024],
                "inline": False
            })

        # Truncate analysis to fit in Discord's limits
        analysis_text = analysis[:2000] if len(analysis) > 2000 else analysis

        # Create embed
        embed = {
            "title": f"Pretraining Analysis: {ticker} on {date}",
            "description": analysis_text,
            "color": 0x9370DB,  # Medium purple color
            "fields": fields,
            "footer": {"text": f"WSB-2 Pretraining • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        return self.send_message(
            content=f"🔍 **Pretraining Analysis for {ticker}**",
            webhook_type="pretraining",
            embed_data=embed
        )

    def send_pretraining_reflection(self, ticker: str, date: str, analysis_type: str,
                                    reflection: str, key_learnings: str, accuracy_info: str = "") -> bool:
        """
        Send pretraining reflection to Discord pretraining webhook

        Args:
            ticker: Stock ticker symbol
            date: Date of reflection
            analysis_type: Type of analysis (e.g., reflection)
            reflection: Full reflection text
            key_learnings: Key learnings from reflection
            accuracy_info: Optional accuracy information

        Returns:
            bool: True if successful, False otherwise
        """
        # Prepare fields for the embed
        fields = [
            {"name": "Ticker", "value": ticker, "inline": True},
            {"name": "Date", "value": date, "inline": True},
            {"name": "Analysis Type", "value": analysis_type, "inline": True}
        ]

        # Add key learnings if available
        if key_learnings:
            fields.append({
                "name": "Key Learnings",
                "value": key_learnings[:1024],
                "inline": False
            })

        # Add accuracy info if available
        if accuracy_info:
            fields.append({
                "name": "Prediction Accuracy",
                "value": accuracy_info,
                "inline": False
            })

        # Truncate reflection to fit in Discord's limits
        reflection_text = reflection[:2000] if len(
            reflection) > 2000 else reflection

        # Create embed
        embed = {
            "title": f"Pretraining Reflection: {ticker} on {date}",
            "description": reflection_text,
            "color": 0x4B0082,  # Indigo color for reflections
            "fields": fields,
            "footer": {"text": f"WSB-2 Pretraining • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        return self.send_message(
            content=f"🧠 **Pretraining Reflection for {ticker}**",
            webhook_type="pretraining",
            embed_data=embed
        )

    def send_pretraining_summary(self, ticker: str, date: str, analysis_type: str,
                                 summary: str, predictions: str) -> bool:
        """
        Send pretraining summary to Discord pretraining webhook

        Args:
            ticker: Stock ticker symbol
            date: Date of summary
            analysis_type: Type of analysis (e.g., summary)
            summary: Full summary text
            predictions: Multi-timeframe predictions

        Returns:
            bool: True if successful, False otherwise
        """
        # Prepare fields for the embed
        fields = [
            {"name": "Ticker", "value": ticker, "inline": True},
            {"name": "Date", "value": date, "inline": True},
            {"name": "Analysis Type", "value": analysis_type, "inline": True}
        ]

        # Add predictions if available
        if predictions:
            fields.append({
                "name": "Predictions",
                "value": predictions[:1024],
                "inline": False
            })

        # Truncate summary to fit in Discord's limits
        summary_text = summary[:2000] if len(summary) > 2000 else summary

        # Create embed
        embed = {
            "title": f"Pretraining Summary: {ticker}",
            "description": summary_text,
            "color": 0x800080,  # Purple color for summaries
            "fields": fields,
            "footer": {"text": f"WSB-2 Pretraining • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }

        return self.send_message(
            content=f"📊 **Pretraining Final Summary for {ticker}**",
            webhook_type="pretraining",
            embed_data=embed
        )

    def send_error_alert(self, title: str, message: str, suggestions: str = None) -> bool:
        """
        Send critical error alerts to Discord error-alerts webhook

        Args:
            title: Error title
            message: Detailed error message
            suggestions: Optional suggestions for resolving the issue

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Sending critical error alert: {title}")

        # Create description with detailed error message
        description = message

        # Add suggestions if available
        fields = []
        if suggestions:
            fields.append({
                "name": "Suggested Actions",
                "value": suggestions,
                "inline": False
            })

        # Create timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add timestamp field
        fields.append({
            "name": "Timestamp",
            "value": timestamp,
            "inline": True
        })

        # Create embed with red color for errors
        embed = {
            "title": title,
            # Limit to 2000 chars for Discord
            "description": description[:2000],
            "color": 0xFF0000,  # Red color for errors
            "fields": fields,
            "footer": {"text": "WSB-2 Critical Alert - TRADING SYSTEM HALTED"}
        }

        # Send to error-alerts webhook for maximum visibility
        result = self.send_message(
            content="⚠️ **CRITICAL ERROR - TRADING SYSTEM HALTED** ⚠️",
            webhook_type="error-alerts",
            username="WSB-2 Error Monitor",
            embed_data=embed
        )

        if result:
            self.logger.info(f"Successfully sent error alert: {title}")
        else:
            self.logger.error(f"Failed to send error alert: {title}")

        return result
