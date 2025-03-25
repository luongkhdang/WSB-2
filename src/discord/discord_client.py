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
            'full-analysis': os.getenv('DISCORD_WEBHOOK_URL_FULL_ANALYSIS')
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
                self.logger.info(f"Message sent successfully to {webhook_type} webhook")
                return True
            else:
                self.logger.error(f"Failed to send message: {response.status_code} - {response.text}")
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
            self.logger.info(f"Content length exceeds Discord limit ({len(content)} chars). Splitting into chunks.")
            
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
                self.logger.error("Failed to send first part of market analysis")
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
                    self.logger.error(f"Failed to send part {part_number} of market analysis")
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
    
    def send_analysis(self, analysis: Dict[str, Any], ticker: str = 'UNKNOWN') -> bool:
        """
        Send stock analysis to Discord channel
        
        Args:
            analysis: Dictionary containing analysis data
            ticker: Stock ticker symbol
            
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
        
        # Add price information for the title
        title = f"{ticker} Stock Analysis"
        if 'raw_data' in analysis and 'current_price' in analysis['raw_data']:
            price = analysis['raw_data']['current_price']
            if price:
                title = f"{ticker} Stock Analysis - ${price:.2f}"
        
        # Create fields list for the trade stats message
        fields = [
            {"name": "Trend", "value": trend, "inline": True},
            {"name": "Technical Score", "value": str(technical_score), "inline": True},
            {"name": "Sentiment Score", "value": str(sentiment_score), "inline": True},
            {"name": "Risk", "value": risk_assessment, "inline": True},
            {"name": "Market Alignment", "value": market_alignment, "inline": True}
        ]
        
        # Add options analysis if available
        options_analysis = analysis.get('options_analysis', '')
        if options_analysis:
            if len(options_analysis) > 500:
                options_analysis = options_analysis[:500] + "..."
            fields.append({"name": "Options Analysis", "value": options_analysis, "inline": False})
            
        # 1. Extract ANALYZER FULL OPINION section if present
        analyzer_opinion = ""
        # First, try to find if the section exists
        if "ANALYZER FULL OPINION:" in description:
            # Get everything after "ANALYZER FULL OPINION:" 
            opinion_start = description.find("ANALYZER FULL OPINION:")
            if opinion_start != -1:
                # Extract from the section header to the end or until another major section
                opinion_text = description[opinion_start + len("ANALYZER FULL OPINION:"):]
                # Trim leading/trailing whitespace
                analyzer_opinion = opinion_text.strip()
                
                # Log what we found for debugging
                self.logger.info(f"Found ANALYZER FULL OPINION section for {ticker}, length: {len(analyzer_opinion)} chars")
                self.logger.info(f"First 100 chars: {analyzer_opinion[:100]}")
                
        # If opinion found, send it to trade-opinion webhook
        if analyzer_opinion:
            self.logger.info(f"Sending ANALYZER FULL OPINION to trade-opinion webhook for {ticker}")
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
            self.logger.info(f"Analysis description exceeds Discord limit ({len(description)} chars). Splitting into parts.")
            
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
                self.logger.error("Failed to send first part of stock analysis")
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
                    self.logger.error(f"Failed to send part {part_number} of stock analysis")
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
