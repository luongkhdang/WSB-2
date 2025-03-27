import pandas as pd
from datetime import date
from src.finance_client.client.yfinance_client import YFinanceClient
from src.main_utilities.data_processor import format_stock_data_for_analysis
from src.main_utilities.file_operations import get_historical_market_context

# Initialize client
client = YFinanceClient()
print("\n[1] Testing YFinanceClient data fetching...\n")

# Test data fetching
data = client.get_historical_data(
    "SPY", date(2023, 3, 18), date(2023, 3, 24), "1d")
print(f"Retrieved {len(data)} rows of data")
print("Sample data:")
print(data.head())
print("\n-----------------------------\n")

# Test data processing
print("\n[2] Testing format_stock_data_for_analysis function...\n")
result = format_stock_data_for_analysis(data, "SPY", debug=True)
print(f"MACD available: {result.get('macd_available', False)}")
print(f"ATR%: {result.get('atr_percentage', 'N/A')}")
print(f"Current price: {result.get('current_price', 'N/A')}")
print("\n-----------------------------\n")

# Test market context
print("\n[3] Testing get_historical_market_context function...\n")
market_context = get_historical_market_context(date(2023, 3, 24), client)
print(f"Market context: {market_context}")
print("\n-----------------------------\n")
