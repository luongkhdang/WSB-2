import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List

# Set up logger
logger = logging.getLogger(__name__)

class DiscordClient:
    """Client for sending messages to Discord webhooks"""
    
    def __init__(self, webhook_url=None, enabled=True):
        """Initialize Discord client with webhook URL"""
        self.webhook_url = webhook_url
        self.enabled = enabled
        
    def send_trade_alert(self, ticker='UNKNOWN', action='ALERT', price=0.0, notes=''):
        """Send a trade alert to Discord channel"""
        if not self.enabled or not self.webhook_url:
            logger.info("Discord notifications not enabled, skipping")
            return
            
        try:
            # Create message content
            content = f"**{action} ALERT: {ticker} @ ${price:.2f}**\n\n"
            
            if notes:
                content += f"{notes}\n"
                
            # Create webhook payload
            payload = {
                "content": content
            }
            
            # Send the request
            response = requests.post(
                self.webhook_url,
                json=payload
            )
            
            if response.status_code >= 400:
                logger.error(f"Error sending Discord alert: {response.status_code}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            return False
            
    def send_analysis(self, analysis, ticker='UNKNOWN'):
        """Send stock analysis to Discord channel"""
        if not self.enabled or not self.webhook_url:
            logger.info("Discord notifications not enabled, skipping")
            return
            
        try:
            content = f"**{ticker} Analysis** ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
            
            # Add trend, price, and scores
            trend = analysis.get('trend', 'UNKNOWN').upper()
            trend_emoji = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"
            
            content += f"{trend_emoji} **Trend:** {trend}\n"
            
            if 'raw_data' in analysis and analysis['raw_data']:
                price = analysis['raw_data'].get('current_price', 0)
                content += f"**Price:** ${price:.2f}\n"
                
            if 'technical_score' in analysis:
                tech_score = analysis['technical_score']
                content += f"**Technical Score:** {tech_score}/10\n"
                
            if 'fundamental_score' in analysis:
                fund_score = analysis['fundamental_score']
                content += f"**Fundamental Score:** {fund_score}/10\n"
                
            if 'sentiment_score' in analysis:
                sent_score = analysis['sentiment_score']
                content += f"**Sentiment Score:** {sent_score}/10\n"
                
            content += "\n"  # Add spacing
            
            # Add analysis summary
            if 'summary' in analysis and analysis['summary']:
                content += f"**Summary:**\n{analysis['summary']}\n\n"
                
            # Add options analysis if available
            if 'options_analysis' in analysis and analysis['options_analysis']:
                content += f"**Options Analysis:**\n{analysis['options_analysis']}\n\n"
                
            # Add options summary data if available
            if 'options_summary' in analysis and analysis['options_summary']:
                content += "**Options Data:**\n"
                # Get the first expiration date for simplicity
                for exp_date, exp_data in list(analysis['options_summary'].items())[:1]:  # Just first expiration
                    days_to_exp = exp_data.get('days_to_expiration', '')
                    content += f"• Expiration: {exp_date} ({days_to_exp} days out)\n"
                    
                    # IV metrics
                    call_iv = exp_data.get('avg_call_iv', 0) * 100
                    put_iv = exp_data.get('avg_put_iv', 0) * 100
                    iv_skew = exp_data.get('iv_skew', 0)
                    content += f"• Implied Volatility: Calls {call_iv:.1f}% / Puts {put_iv:.1f}%\n"
                    content += f"• IV Skew: {iv_skew:.2f}\n"
                    
                    # Volume metrics
                    call_put_ratio = exp_data.get('call_put_volume_ratio', '')
                    content += f"• Call/Put Volume Ratio: {call_put_ratio:.2f}\n"
                    
                    # ATM options
                    if 'atm_call' in exp_data and exp_data['atm_call']:
                        atm_call = exp_data['atm_call']
                        content += f"• ATM Call: Strike ${atm_call.get('strike', 0)}, Delta {atm_call.get('delta', 0):.2f}, IV {atm_call.get('iv', 0)*100:.1f}%\n"
                        
                    if 'atm_put' in exp_data and exp_data['atm_put']:
                        atm_put = exp_data['atm_put']
                        content += f"• ATM Put: Strike ${atm_put.get('strike', 0)}, Delta {atm_put.get('delta', 0):.2f}, IV {atm_put.get('iv', 0)*100:.1f}%\n"
                
                content += "\n"  # Add spacing
            
            # Add key technical indicators
            if 'technical_indicators' in analysis and analysis['technical_indicators']:
                content += "**Technical Indicators:**\n"
                indicators = analysis['technical_indicators']
                
                for indicator, value in indicators.items():
                    if indicator in ['rsi', 'macd', 'stochastic']:
                        content += f"• {indicator.upper()}: {value}\n"
                        
                content += "\n"  # Add spacing
                
            # Add trading signals
            if 'buy_signals' in analysis and analysis['buy_signals']:
                content += "**Buy Signals:**\n"
                for signal in analysis['buy_signals']:
                    content += f"• {signal}\n"
                content += "\n"
                
            if 'sell_signals' in analysis and analysis['sell_signals']:
                content += "**Sell Signals:**\n"
                for signal in analysis['sell_signals']:
                    content += f"• {signal}\n"
                content += "\n"
                
            # Add key insights
            if 'key_insights' in analysis and analysis['key_insights']:
                content += "**Key Insights:**\n"
                for insight in analysis['key_insights'][:3]:  # Limit to top 3 insights
                    content += f"• {insight}\n"
                    
            # Create webhook payload
            payload = {
                "content": content[:2000]  # Discord has a 2000 character limit
            }
            
            # Send the request
            response = requests.post(
                self.webhook_url,
                json=payload
            )
            
            # Check if the message was too long and send the rest as a follow-up
            if len(content) > 2000:
                remaining_content = content[2000:]
                # Split into chunks of 2000 characters
                chunks = [remaining_content[i:i+2000] for i in range(0, len(remaining_content), 2000)]
                
                for chunk in chunks:
                    follow_up_payload = {"content": chunk}
                    requests.post(self.webhook_url, json=follow_up_payload)
            
            if response.status_code >= 400:
                logger.error(f"Error sending Discord notification: {response.status_code}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord analysis: {e}")
            return False 