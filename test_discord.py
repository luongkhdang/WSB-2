from src.discord.discord_client import DiscordClient
import logging

# Set up logging to see the chunking process
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_discord_client():
    client = DiscordClient()
    
    # Create a very long message (about 6000 characters)
    long_content = "This is a test of the market analysis Discord formatting with a very long message that exceeds the Discord limit. " * 50
    print(f"Message length: {len(long_content)} characters")
    
    # Test sending a market analysis with a long message
    print("Sending market analysis...")
    result = client.send_market_analysis(
        title="Test Long Message", 
        content=long_content,
        metrics={
            "Test Metric": "Value",
            "Another Metric": 123,
            "Third Metric": "Test"
        }
    )
    
    print(f"Result: {result}")
    
if __name__ == "__main__":
    test_discord_client() 