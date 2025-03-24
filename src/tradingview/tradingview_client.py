import requests
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingViewClient:
    """
    Client for interacting with TradingView webhook alerts.
    This class handles setting up and managing webhooks for TradingView alerts.
    """
    
    def __init__(self):
        """Initialize the TradingView client."""
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
    def get_webhook_url(self, endpoint: str = "webhook") -> str:
        """
        Get the webhook URL to use in TradingView alerts.
        
        Args:
            endpoint: The specific endpoint for the webhook
            
        Returns:
            The full webhook URL
        """
        return f"http://{self.api_host}:{self.api_port}/{endpoint}"
    
    def validate_webhook_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Validate that the webhook payload from TradingView has all required fields.
        
        Args:
            payload: The webhook payload from TradingView
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["ticker", "strategy", "action"]
        return all(field in payload for field in required_fields)
    
    @staticmethod
    def format_discord_message(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a TradingView alert payload into a Discord message.
        
        Args:
            payload: The webhook payload from TradingView
            
        Returns:
            A formatted Discord message payload
        """
        ticker = payload.get("ticker", "Unknown")
        strategy = payload.get("strategy", "Unknown")
        action = payload.get("action", "Unknown")
        price = payload.get("price", "Unknown")
        
        message = {
            "embeds": [
                {
                    "title": f"TradingView Alert: {action.upper()} {ticker}",
                    "color": 65280 if action.lower() == "buy" else 16711680,  # Green for buy, red for sell
                    "fields": [
                        {"name": "Strategy", "value": strategy, "inline": True},
                        {"name": "Action", "value": action.upper(), "inline": True},
                        {"name": "Price", "value": str(price), "inline": True}
                    ],
                    "footer": {"text": "Generated from TradingView Alert"}
                }
            ]
        }
        
        # Add additional fields if they exist
        for key, value in payload.items():
            if key not in ["ticker", "strategy", "action", "price"]:
                message["embeds"][0]["fields"].append(
                    {"name": key.capitalize(), "value": str(value), "inline": True}
                )
                
        return message
    
    @staticmethod
    def send_to_discord(webhook_url: str, payload: Dict[str, Any]) -> bool:
        """
        Send a formatted message to Discord via webhook.
        
        Args:
            webhook_url: The Discord webhook URL
            payload: The message payload to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            formatted_message = TradingViewClient.format_discord_message(payload)
            response = requests.post(
                webhook_url,
                json=formatted_message,
                headers={"Content-Type": "application/json"}
            )
            return response.status_code == 204
        except Exception as e:
            print(f"Error sending to Discord: {e}")
            return False

    def test_webhook(self, webhook_url: Optional[str] = None) -> bool:
        """
        Test the webhook functionality with a sample alert.
        
        Args:
            webhook_url: Optional override for the Discord webhook URL
            
        Returns:
            True if successful, False otherwise
        """
        if webhook_url is None:
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL_TRADE_ALERTS")
            
        if not webhook_url:
            print("No webhook URL provided and none found in environment variables")
            return False
            
        test_payload = {
            "ticker": "AAPL",
            "strategy": "Test Strategy",
            "action": "buy",
            "price": 190.75,
            "notes": "This is a test alert"
        }
        
        return self.send_to_discord(webhook_url, test_payload)
