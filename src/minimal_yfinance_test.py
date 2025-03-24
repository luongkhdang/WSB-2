import yfinance as yf
print("Testing yfinance...")

# Basic usage example directly from the yfinance docs
try:
    # This should work based on the official docs
    data = yf.download("SPY", period="1d")
    print("Download successful!")
    print(data)
except Exception as e:
    print(f"Download failed: {e}")

# Try the Ticker functionality
try:
    # This should also work based on the official docs
    msft = yf.Ticker("MSFT")
    print("\nTicker creation successful!")
    print(f"Historical data: {msft.history(period='1d')}")
except Exception as e:
    print(f"Ticker creation failed: {e}") 