
# Add this modified method to discord/discord_client.py

def send_market_analysis(self, title, content, metrics=None):
    """
    Send market analysis information with improved error handling
    
    Args:
        title: Analysis title
        content: Analysis content (will be truncated if too long)
        metrics: Optional dictionary of metrics to include
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Truncate content if it's too long (Discord limit is ~2000 characters)
        if len(content) > 1900:
            self.logger.warning(f"Content exceeds 1900 characters, truncating to 1900 characters")
            content = content[:1900] + "...(truncated)"
            
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
    except Exception as e:
        self.logger.error(f"Error sending market analysis: {e}")
        return False
