#!/usr/bin/env python3
"""
Test script for the enhanced options sentiment analysis in YFinanceClient
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finance_client.client.yfinance_client import YFinanceClient

def main():
    """Test the enhanced options sentiment analysis"""
    print("Testing enhanced options sentiment analysis...")
    
    # Initialize the client
    client = YFinanceClient()
    
    # Get options sentiment data
    sentiment_data = client.get_options_sentiment()
    
    # Display the results
    print("\nOptions Sentiment Analysis Results:")
    print(f"Expiration Date: {sentiment_data.get('expiration_date')}")
    print(f"Call/Put Volume Ratio: {sentiment_data.get('call_put_volume_ratio')}")
    print(f"Call/Put OI Ratio: {sentiment_data.get('call_put_oi_ratio')}")
    print(f"Call IV Avg: {sentiment_data.get('call_iv_avg')}")
    print(f"Put IV Avg: {sentiment_data.get('put_iv_avg')}")
    print(f"IV Skew: {sentiment_data.get('iv_skew')}")
    print(f"Sentiment: {sentiment_data.get('sentiment')}")
    print(f"Total Call Volume: {sentiment_data.get('total_call_volume'):,}")
    print(f"Total Put Volume: {sentiment_data.get('total_put_volume'):,}")
    
    # Check if this is fallback data
    if sentiment_data.get('note'):
        print(f"\nNote: {sentiment_data.get('note')}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 