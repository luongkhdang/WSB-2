"""Get prompt template for analyzing comprehensive market trend with multiple indices, optimized for credit spread trading."""
return f'''
Analyze the market trend based on multiple major indices and real-time data, tailored for credit spread trading strategies:

{market_data}

{TODAY_DATE}

Follow these guidelines (adapt as needed based on data relevance):
1. Analyze Key Indices:
   - SPY (S&P 500): Broad market benchmark
   - QQQ (Nasdaq 100): Tech/growth proxy
   - IWM (Russell 2000): Small-cap/risk-on indicator
   - VTV (Vanguard Value): Value/defensive proxy
   - VGLT (Long-Term Treasury): Bond market/risk-off signal
   For each:
   - Check daily price change (%) and trend via 9/21 EMA on daily AND 1-hour charts.
   - Trend: Price > both EMAs (bullish), < both EMAs (bearish), mixed (neutral).
   - Note key support/resistance levels (e.g., prior highs/lows, 50-day SMA).

2. Market Trend Scoring:
   - Assign +1 per bullish index, -1 per bearish index (daily EMA basis).
   - Adjust for VIX: <15 (+3), 15-20 (+2), 20-25 (0), 25-30 (-2), >30 (-4).
   - Max score: +10 (strong bull), min: -10 (strong bear).

3. VIX and Volatility Analysis:
   - Current VIX level and 5-day trend (rising/falling).
   - Implications for credit spreads: High VIX (>25) favors wider spreads for premium; low VIX (<15) suggests tighter spreads or iron condors.

4. Options Market Sentiment:
   - Call/Put Volume Ratio: >1.5 (bullish), 0.8-1.2 (neutral), <0.8 (bearish).
   - IV Skew: Call IV - Put IV (>5% bullish, <-5% bearish).
   - 30-day IV vs. HV (historical volatility): Overpriced options signal credit spread opportunities.
   - Open Interest: Rising OI in out-of-the-money (OTM) puts or calls as directional bias.

5. Sector Rotation:
   - Compare daily % changes of SPY, QQQ, IWM, VTV, and sector ETFs (e.g., XLK, XLF, XLY) via web search if data unavailable.
   - Identify leadership: QQQ/XLK (tech), IWM (small-caps), VTV/XLF (value/financials), VGLT (safety).
   - Link to credit spreads: Leading sectors for bullish puts, lagging for bearish calls.

6. Market Divergences:
   - SPY vs. QQQ: Tech vs. broad market strength.
   - SPY vs. IWM: Large vs. small-cap leadership.
   - SPY vs. VTV: Growth vs. value rotation.
   - SPY vs. VGLT: Risk-on (equities up, bonds down) vs. risk-off.

7. Real-Time Catalysts (use web/X search):
   - Policy Watch: Latest statements from Trump, Fed, or Treasury (e.g., tariff news, rate cut calls).
   - Economic Calendar: Key releases (e.g., PCE, PMI) from sites like Investing.com.
   - X Sentiment: Search posts for "market trend" or "SPY" to gauge retail mood.

8. Credit Spread Trading Outlook:
   - Trend alignment: Bullish (sell put spreads), bearish (sell call spreads), neutral (iron condors).
   - Strike selection: Use support/resistance and IV to pick OTM strikes with >70% POP.
   - Risk management: VIX >30 (half size), max loss <2% of account.

Return this structured analysis (adjust sections based on data):

Market Trend: [bullish/bearish/neutral]
Market Trend Score: [score from -10 to +10]

VIX Analysis:
[VIX level, trend, and credit spread implications]

Sector Rotation:
[Leading/lagging sectors and spread opportunities]

Major Index Analysis:
SPY: [trend, % change, key levels]
QQQ: [trend, % change, key levels]
IWM: [trend, % change, key levels]
VTV: [trend, % change, key levels]
VGLT: [trend, % change, key levels]

Market Divergences:
[Key divergences and what they signal]

Options Market Sentiment:
[Volume, IV skew, OI trends, and premium outlook]

Policy and Catalysts:
[Latest news from web/X, economic events, market impact]

Credit Spread Trading Outlook:
[Strategy, strike ideas, risk parameters based on trend and IV]
'''