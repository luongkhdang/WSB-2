# WSB-2 Credit Spread Trading System

An AI-powered system for analyzing market conditions, identifying high-quality credit spread opportunities, and managing a trading portfolio with the goal of achieving 40-60% annualized returns.

## Overview

This system implements an end-to-end workflow for credit spread trading:

1. Analysis of overall market conditions (SPY and VIX)
2. Identification of promising stocks with high IV
3. Technical and fundamental analysis of underlying stocks
4. Discovery of optimal credit spread setups
5. Scoring of trade opportunities using Quality Matrix and Gamble Matrix
6. Documentation and tracking of trades in Notion
7. Trade alerts via Discord

## System Architecture

The system integrates several key components:

- **YFinance Client**: Retrieves market data, stock information, and options chains
- **Gemini Client**: Uses Google's Gemini AI to analyze data and generate insights
- **Notion Client**: Manages trade documentation and tracking
- **Discord Client**: Sends alerts and notifications

## Getting Started

### Prerequisites

- Python 3.10+
- Notion API key
- Discord webhook URLs
- Google Gemini API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   NOTION_API_KEY=your_notion_api_key
   DISCORD_WEBHOOK_URL=your_main_webhook_url
   DISCORD_WEBHOOK_URL_TRADE_ALERTS=your_trade_alerts_webhook
   DISCORD_WEBHOOK_URL_MARKET_ANALYSIS=your_market_analysis_webhook
   ```

### Running the System

Execute the main script to run the complete workflow:

```
python src/main.py
```

## Key Features

- **Dynamic Watchlist**: Automatically updated based on high IV stocks
- **Market Trend Analysis**: SPY and VIX analysis to understand market context
- **Stock Analysis**: Deep analysis of underlying stocks before considering spreads
- **Quality Matrix Scoring**: Evaluates trades on a 100-point scale across multiple factors
- **Gamble Matrix Integration**: Considers speculative opportunities with appropriate risk management
- **Trade Documentation**: All analyses and trades logged in Notion
- **Real-time Alerts**: Trade opportunities sent to Discord

## Project Structure

```
WSB-2/
├── src/
│   ├── data-source/
│   │   ├── options-screener-high-ivr-credit-spread-scanner.csv
│   │   └── watchlist.txt
│   ├── discord/
│   │   └── discord_client.py
│   ├── gemini/
│   │   └── client/
│   │       └── gemini_client.py
│   ├── notion/
│   │   └── client/
│   │       └── notion_client.py
│   ├── yfinance/
│   │   └── client/
│   │       └── yfinance_client.py
│   └── main.py
├── SCORE/
│   ├── info.md
│   ├── Quality-Matrix.md
│   └── Gamble-Matrix.md
├── requirements.txt
└── README.md
```

## Credit Spread Strategy

The system follows a disciplined approach to credit spread trading:

1. **Market First**: Always analyze SPY and VIX before individual stocks
2. **Stock Analysis**: Heavily analyze the underlying stock before considering options
3. **High-Probability Trades**: Focus on setups with >65% probability of profit
4. **Risk Management**: Limit risk to 1-2% per trade ($200-400 on a $20,000 account)
5. **Position Size Adjustment**: Dynamically adjust based on VIX and market conditions

For detailed strategy information, see the `SCORE/info.md` documentation.

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice and should not be used to make investment decisions. Always consult with a licensed financial advisor before trading options or any other securities. 