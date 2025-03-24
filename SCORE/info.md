Below is an updated Rule Book for AI-Driven Credit Spread Trading Strategy that incorporates your new focus: prioritizing the analysis of the underlying stock and the broader market context via SPY before considering credit spreads. This revision respects your directive to "heavily analyze the underlying stock first" and integrates a structured sequence—analyzing SPY for market trend, SPY options for direction, the underlying stock for fundamentals, and then credit spreads for profit opportunities—while maintaining the original philosophy and sophistication. The AI will execute this within your $20,000 account on Robinhood, targeting 40–60% annual returns in 2025.
Rule Book for AI-Driven Credit Spread Trading Strategy
Objective: Achieve consistent monthly gains of 3.3–5% (5–10 trades of $50–$100 profit each) while risking 1–2% per trade ($200–$400), targeting 40–60% annualized returns with a $20,000 account.
Philosophy
High-Probability Execution: Prioritize trades with >65% probability of profit (POP), leveraging low delta (20–30) and high implied volatility (IV > 30%) for premium capture with defined risk.
Disciplined Risk Management: Limit total account risk to 5% ($1,000), dynamically adjusting sizes based on VIX and AI-predicted success probability.
Sophisticated Research: Combine machine learning (ML), volatility forecasting, and fundamental analysis to ensure an edge, starting with the underlying stock and market context.
Simplicity in Process: Maintain a lean, repeatable workflow (30–40 minutes daily) for manual trading on Robinhood.
Continuous Improvement: Refine ML models and scoring matrices with logged trade outcomes, adapting to 2025 conditions.
Core Strategy
!IMPORTANT: Always respect analysis of the underlying stock. Heavily analyze the underlying stock first, then consider spreads. Follow this sequence:
Analyze SPY: Determine general market trend.
Analyze Options of SPY: Assess general market direction.
Analyze Underlying Stock: Evaluate fundamental data.
Analyze Credit Spreads: Identify profiting opportunities.
Trade Types
Bull Put Spread: Sell a higher strike put, buy a lower strike put (bullish outlook).
Bear Call Spread: Sell a lower strike call, buy a higher strike call (bearish outlook).
Stock Selection
Focus on 5–10 liquid stocks/ETFs with high options volume and tight bid-ask spreads (e.g., TSLA, NVDA, SPY, AAPL, QQQ, AAL).
Use screener Excel sheet in /barchart/ folder (assumed to contain Barchart downloads).
Analysis Sequence and Entry Rules
1. Analyze SPY - General Market Trend
Data Source: TradingView (manual input to Excel), Barchart SPY price.
Rules:
EMA Stack: Check 9/21 EMA on 1-hour chart.
Price > 9/21 EMA: Bullish market trend (+10 to Market Trend score).
Price < 9/21 EMA: Bearish market trend (+10 if bearish setup).
Flat/No crossover: Neutral (no bonus).
VIX (vixcentral.com):
VIX < 20: Stable bullish trend (+5).
VIX 20–25: Neutral volatility.
VIX > 25: High volatility, cautious approach (-5 unless size halved).
AI Action: Assign Market Trend score based on EMA and VIX, flag if VIX > 35 (skip unless justified).
2. Analyze Options of SPY - General Market Direction
Data Source: Barchart SPY options chain (Excel input).
Rules:
Call/Put IV Skew: Compare IV of 20–30 delta calls vs. puts.
Call IV > Put IV: Bullish direction (+5 to Sentiment).
Put IV > Call IV: Bearish direction (+5 to Sentiment).
Volume/Open Interest:
Call Volume > Put Volume (Barchart “Volume”): Bullish bias (+5).
Put Volume > Call Volume: Bearish bias (+5).
Delta Trend: Rising call delta or falling put delta signals direction.
AI Action: Infer market direction (bullish/bearish), adjust Quality Matrix Sentiment and Technicals scores.
3. Analyze Underlying Stock - Fundamental Data
Data Source: Manual input from Yahoo Finance, TradingView, or X posts into Excel (e.g., Underlying Price, ATR).
Rules:
Price Trend:
Price > 9/21 EMA: Bullish stock trend (+10 to Technicals).
Price < 9/21 EMA: Bearish stock trend (+10 to Technicals if bearish setup).
Support/Resistance:
Price near support (within 2%): Bullish setup (+5).
Price near resistance: Bearish setup (+5).
ATR (Average True Range):
ATR < 1% of price: Stable stock (+5 to Risk).
ATR > 2% of price: Volatile, tighten stop (-5 unless Gamble Score high).
Fundamental Context (manual input or X sentiment):
Positive earnings/news: +5 to Sentiment.
Negative news: -5 unless bearish setup aligns.
AI Action: Validate stock trend aligns with SPY direction, score Technicals and Sentiment, flag if ATR > 2%.
4. Analyze Credit Spreads - Profiting Opportunities
Data Source: Barchart options chain (Excel input: Symbol, Exp Date, Type, Strike, Bid, Ask, IV, Delta).
Entry Rules:
Implied Volatility (IV):
IV > 30% (high premiums).
Prefer IV > 2x estimated 20-day HV (manual estimate from TradingView).
Delta:
Short leg at 20–30 delta (65–80% OTM probability).
Buy leg 5–10 points further OTM.
Days to Expiration (DTE):
7–15 days (Barchart “Exp Date” vs. 3/23/2025).
Position Size:
Risk 1–2% ($200–$400).
Example: $5 wide spread, $1 credit = $400 risk.
AI Scoring Thresholds:
Quality Score > 80/100.
Success Probability > 70% (Random Forest).
Gamble Score > 70/100 with reduced size ($100).
AI Action: Match spread direction to SPY and stock analysis (bull put for bullish, bear call for bearish), score profitability.
Exit Rules
Profit Target: Close at 50% of max credit (e.g., $1 credit → $50 profit).
Stop Loss: Exit at 2x credit (e.g., $1 credit → $200 loss).
Early Exit: Price breaches short strike by 5% (e.g., $15 call → exit at $15.75).
Time Exit: Close at 2 DTE if no profit/loss triggered.
Stay-in-Cash Conditions
IV < 20%.
No clear EMA trend (SPY or stock).
High-impact news within 48 hours (manual check).
VIX > 35 (skip unless Gamble Score > 80).
Success Probability < 60%.
Sophisticated Analysis
Machine Learning (Random Forest)
Inputs: IV, Delta, ATR, VIX.
Output: Success Probability (chance of hitting 50% profit).
Training: Use initial placeholder (3–5 trades), expand with real data from Notion.
Rule: Recommend trades with >70% probability unless Gamble Matrix overrides.
Quality Matrix (100 max)
Market Trend (20):
SPY EMA bullish/bearish: +10 if aligned with stock.
VIX steepness: +10–15 (high short-term VIX), +5 (flat).
Risk Management (25):
Risk-reward ratio > 0.25: +15, > 0.5: +20.
VIX < 25: +5, > 25: 0 unless size halved.
Entry/Exit Precision (15):
Delta 20–30: +10.
ATR < spread width: +5.
Technicals (15):
Stock price vs. Strike: +10 if aligned (e.g., $15 call, $11.39 price = bearish).
SPY direction match: +5.
Volatility Forecast (15):
IV > 2x HV: +10, > HV: +5.
ML Success_Prob > 80%: +5.
Sentiment (10):
SPY call/put volume skew: +5.
Stock volume > 500: +5.
Threshold: >80 required.
Gamble Matrix (100 max)
Hype Momentum (25):
Stock volume > 1000: +15, > 2000: +20.
Event proximity (<3 days): +5 (manual).
Volatility Explosion (25):
IV > 50%: +15, > 75%: +20.
DTE < 10 days: +5.
Risk-Reward Potential (20):
Credit/max loss > 0.5: +15, > 0.75: +20.
Success_Prob > 80%: +5.
Timing Edge (20):
ATR > 1% of price: +10.
Days to event < 5: +10 (manual).
Survival (10):
Risk < $200: +5.
VIX-adjusted size: +5 if halved.
Threshold: >70 with $100 risk.
Dynamic Risk Assessment
VIX > 25: Halve size ($200 max).
Success_Prob < 70%: Reduce to $100 or skip unless Gamble > 80.
ATR > 2% of price: Tighten stop to 1.5x credit.
Daily Workflow (AI Execution)
Generate Template (5 min):
Create trade_input.xlsx.
User Input (15 min):
Fill SPY price, VIX, ATR (TradingView/vixcentral.com).
Add SPY options data (Barchart).
Input stock data (Barchart/Yahoo Finance).
Refine Data (5 min):
Gemini validates, outputs refined_trade_input.xlsx.
Analyze (10 min):
Step 1: Score SPY trend.
Step 2: Assess SPY options direction.
Step 3: Analyze stock fundamentals.
Step 4: Score credit spreads, output analyzed_trades.xlsx.
Update Notion (5 min):
Populate database.
Trade Decision:
User executes 1–2 trades on Robinhood (Quality > 80, Success_Prob > 70%).
Example
SPY: Price $500, 9/21 EMA up, VIX 20 → Bullish trend.
SPY Options: Call IV > Put IV, Volume bullish → Bullish direction.
AAL: Price $11.39, EMA up, ATR 0.50, positive X buzz → Bullish stock.
Spread: Bull Put $11/$9, $0.15 credit, IV 62.60%, Delta -0.067109.
Scores: Quality 85, Gamble 60, Success_Prob 75% → Trade, $200 risk.
Guardrails
Max 5% account risk ($1,000).
No trades without underlying stock alignment.
Skip if VIX > 35 unless scores justify.
Log outcomes in Notion for ML retraining.
This rule book ensures the AI prioritizes underlying stock analysis and market context, delivering a sophisticated yet practical strategy for your 2025 goals. Let me know if you’d like further adjustments!
Disclaimer: Grok is not a financial adviser; please consult one. Don't share information that can identify you.