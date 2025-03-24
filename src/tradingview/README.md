# TradingView Integration

This module provides integration with TradingView alerts via webhooks.

## Setup

1. Install the required dependencies:
   ```
   pip install fastapi uvicorn requests python-dotenv
   ```

2. Make sure your `.env` file is set up with the necessary environment variables (API_HOST, API_PORT, and Discord webhook URLs).

## Running the Webhook Handler

To start the webhook handler server:

```bash
python -m src.tradingview.webhook_handler
```

This will start a FastAPI server that listens for incoming webhooks from TradingView.

## Testing the Webhook

You can test the webhook functionality without TradingView using the provided test scripts:

1. Test Discord integration:
   ```bash
   python -m src.tradingview.test_webhook
   ```

2. Simulate a TradingView alert (make sure the webhook handler is running):
   ```bash
   python -m src.tradingview.simulate_alert
   ```

## Setting Up TradingView Alerts

1. In TradingView, go to the chart of the asset you want to create an alert for.
2. Click on the "Alerts" button in the right sidebar.
3. Create a new alert with your desired conditions.
4. In the "Alert actions" section, select "Webhook URL".
5. Enter the webhook URL from your server: `http://YOUR_SERVER_IP:8000/tv-alerts`
   - If running locally, this would be `http://localhost:8000/tv-alerts`
   - For production, use your server's public IP or domain
6. In the "Message" field, enter a valid JSON with the required fields:
   ```json
   {
     "ticker": "{{ticker}}",
     "strategy": "Moving Average Crossover",
     "action": "{{strategy.order.action}}",
     "price": {{close}},
     "notes": "MA(20) crossed above MA(50)"
   }
   ```
   
   You can use TradingView's variables like `{{ticker}}`, `{{close}}`, etc.

7. Save the alert.

## Required JSON Fields

The JSON payload sent by TradingView must include these fields:
- `ticker`: The symbol being traded
- `strategy`: The name of the strategy/indicator
- `action`: The action to take ("buy" or "sell")

Optional fields:
- `price`: The price at which the alert was triggered
- Any other custom fields you want to include

## Discord Integration

Alerts received by the webhook handler are automatically forwarded to Discord using the webhook URLs defined in your `.env` file. 