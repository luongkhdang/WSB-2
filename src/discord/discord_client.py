import os
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union
from datetime import datetime

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
            'backtest': os.getenv('DISCORD_WEBHOOK_URL_BACKTEST')
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
            content: Analysis content
            metrics: Optional dictionary of metrics to include
            
        Returns:
            bool: True if successful, False otherwise
        """
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
            content=f"Market Analysis: {title}",
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
        if len(description) > 1500:  # Discord embed description limit is 2048, leave room for fields
            description = description[:1500] + "..."
            
        # Create fields list
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
            
        # Create the embed
        embed = {
            "title": f"{ticker} Stock Analysis",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": f"WSB-2 Stock Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        
        # Add price information if available
        if 'raw_data' in analysis and 'current_price' in analysis['raw_data']:
            price = analysis['raw_data']['current_price']
            if price:
                embed["title"] = f"{ticker} Stock Analysis - ${price:.2f}"
        
        return self.send_message(
            content=f"Stock Analysis for {ticker}",
            webhook_type="market_analysis",
            embed_data=embed
        )
