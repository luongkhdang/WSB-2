"""
Market analysis prompt templates for Gemini API.

Note: The system has been updated to handle long responses from Gemini API
by automatically splitting them into multiple messages when sent to Discord,
which has a 2000 character limit per message.
"""

def get_market_trend_prompt(market_data):
    """Get prompt template for analyzing comprehensive market trend with multiple indices."""
    return f'''
    Analyze the market trend based on multiple major indices data:
    
    {market_data}
    
    TODAY'S DATE: [TODAY'S DATE]
    Follow these rules exactly:
    1. Analyze each major index (SPY, QQQ, IWM, VTV, VGLT):
       - SPY (S&P 500): Overall market benchmark
       - QQQ (Nasdaq 100): Technology and growth stocks
       - IWM (Russell 2000): Small-cap stocks
       - VTV (Vanguard Value): Value stocks
       - VGLT (Long-Term Treasury): Bond market indicator
      For each:
       - Check daily price change (%) and trend via 9/21 EMA on daily AND 1-hour charts.
       - Trend: Price > both EMAs (bullish), < both EMAs (bearish), mixed (neutral).
       - Note key support/resistance levels (e.g., prior highs/lows, 50-day SMA).
       
    2. For each index check the 9/21 EMA on 1-hour chart:
       - Price > 9/21 EMA: Bullish trend (+2 to Market Trend score per index)
       - Price < 9/21 EMA: Bearish trend (+2 if bearish setup per index)
       - Flat/No crossover: Neutral (no bonus)
       Market Trend Scoring:
       - Assign +1 per bullish index, -1 per bearish index (daily EMA basis).
       - Adjust for VIX: <15 (+3), 15-20 (+2), 20-25 (0), 25-30 (-2), >30 (-4).
       - Max score: +10 (strong bull), min: -10 (strong bear).
    
    3. Check for market divergences:
       - SPY vs QQQ: Growth vs Broad Market
       - SPY vs IWM: Small Cap vs Large Cap strength 
       - SPY vs VTV: Growth vs Value rotation
       - VGLT trend: Inverse relationship with equities typically
    
    4. Check VIX level:
       - Current VIX level and 5-day trend (rising/falling).
       - Implications for credit spreads: High VIX (>25) favors wider spreads for premium; low VIX (<15) suggests tighter spreads or iron condors.

    
    5. Options sentiment analysis:
       - Call/Put Volume Ratio: >1.5 (bullish), 0.8-1.2 (neutral), <0.8 (bearish).
       - IV Skew: Call IV - Put IV (>5% bullish, <-5% bearish).
       - 30-day IV vs. HV (historical volatility): Overpriced options signal credit spread opportunities.
       - Open Interest: Rising OI in out-of-the-money (OTM) puts or calls as directional bias.
       - Call IV > Put IV by >5%: Strongly bullish direction (+7 to Sentiment)
       - Call IV > Put IV by <5%: Mildly bullish direction (+3 to Sentiment)
       - Put IV > Call IV by <5%: Mildly bearish direction (+3 to Sentiment)
       - Put IV > Call IV by >5%: Strongly bearish direction (+7 to Sentiment)
       - Call/Put Volume Ratio > 1.5: Strong bullish bias (+7)
       - Call/Put Volume Ratio 1.1-1.5: Mild bullish bias (+3)
       - Call/Put Volume Ratio 0.9-1.1: Neutral (0)
       - Call/Put Volume Ratio 0.5-0.9: Mild bearish bias (+3)
       - Call/Put Volume Ratio < 0.5: Strong bearish bias (+7)
    
    6. Sector Rotation Analysis:
       - Compare daily % changes of SPY, QQQ, IWM, VTV, and sector ETFs (e.g., XLK, XLF, XLY) via web search if data unavailable.
       - Identify leadership: QQQ/XLK (tech), IWM (small-caps), VTV/XLF (value/financials), VGLT (safety).
       - Link to credit spreads: Leading sectors for bullish puts, lagging for bearish calls.
       - Identify the best and worst performing indices
       - Analyze what this means for market sentiment
       - QQQ leading: Tech/Growth leadership
       - IWM leading: Small cap strength, risk-on environment
       - VTV leading: Defensive positioning, value over growth
       - VGLT leading: Safety trade, potential risk-off environment
       - Detect if small caps are leading or lagging large caps
       
    7. Market Divergences:
       - SPY vs. QQQ: Tech vs. broad market strength.
       - SPY vs. IWM: Large vs. small-cap leadership.
       - SPY vs. VTV: Growth vs. value rotation.
       - SPY vs. VGLT: Risk-on (equities up, bonds down) vs. risk-off.

    8. Real-Time Catalysts (use web/X search):
       - Policy Watch: Latest statements from Trump, Fed, or Treasury (e.g., tariff news, rate cut calls).
       - Economic Calendar: Key releases (e.g., PCE, PMI) from sites like Investing.com.
       - X Sentiment: Search posts for "market trend" or "SPY" to gauge retail mood.
    !IMPORTANT: Credit Spread Trading Outlook:
       - Trend alignment: Bullish (sell put spreads), bearish (sell call spreads), neutral (iron condors).
       - Strike selection: Use support/resistance and IV to pick OTM strikes with >70% POP.
       - Risk management: VIX >30 (half size), max loss <2% of account.
       
    Return a structured analysis with the following EXACT sections:
    
    Market Trend: [bullish/bearish/neutral]
    Market Trend Score: [number out of 30]
    
    VIX Analysis: 
    [Your assessment of VIX implications]
    
    Risk Adjustment: 
    [standard/half size/skip]
    
    Sector Rotation:
    [Detailed analysis of which sectors are leading/lagging]
    
    Major Index Analysis:
    SPY: [trend and key levels]
    QQQ: [trend and key levels]
    IWM: [trend and key levels]
    VTV: [trend and key levels]
    VGLT: [trend and key levels]
    
    Market Divergences:
    [Identify any significant divergences between indices]
    
    Options Market Sentiment:
    [Analysis of options data and what it suggests]
    
    Trading Outlook:
    [Brief summary of overall trading approach based on all data] 
    
    Credit Spread Trading Outlook:
    [Strategy, strike ideas, risk parameters based on trend and IV]
    
    !IMPORTANT: YOU , THE AI AGENT, ARE THE EXPERT AND THE DECISION MAKER. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: BE CRITICAL AND THOUGHTFUL. YOU ARE NOT A YES MAN. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: WE ARE TRYING TO MAKE A LOT OF MONEY. YOU ARE THE EXPERT AND THE DECISION MAKER. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE FOR THE DECISIONS MADE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE.
    ANALYZER FULL OPINION: [YOUR FULL OPINION HERE]
    
    
    
    
    '''

def get_spy_options_prompt(options_data):
    """Get prompt template for analyzing SPY options data."""
    return f'''
    Analyze SPY options data to determine market direction:
    
    {options_data}
    
    Follow these rules exactly:
    1. Call/Put IV Skew: Compare IV of 20–30 delta calls vs. puts
       - Call IV > Put IV: Bullish direction (+5 to Sentiment)
       - Put IV > Call IV: Bearish direction (+5 to Sentiment)
    
    2. Volume/Open Interest:
       - Call Volume > Put Volume: Bullish bias (+5)
       - Put Volume > Call Volume: Bearish bias (+5)
    
    3. Delta Trend: Rising call delta or falling put delta signals direction
    
    Return:
    1. Overall market direction prediction (bullish/bearish/neutral)
    2. Sentiment score adjustment
    3. Technical score adjustment
    4. Confidence level (high/medium/low)
    '''

def get_market_data_prompt(market_data):
    """Get prompt template for analyzing general market data."""
    return f'''
    Analyze the following market data and provide insights:
    
    {market_data}
    
    Please include:
    1. Key market trends
    2. Potential trading opportunities
    3. Risk factors to consider
    ''' 