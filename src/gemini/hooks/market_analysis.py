"""
Market analysis prompt templates for Gemini API.
"""

def get_market_trend_prompt(market_data):
    """Get prompt template for analyzing comprehensive market trend with multiple indices, optimized for credit spread trading."""
    return f'''
    Analyze the market trend based on multiple major indices data, tailored for credit spread trading strategies:
    
    {market_data}
    
    Follow these rules exactly:
    1. Analyze each major index (SPY, QQQ, IWM, VTV, VGLT, DIA, BND, BTC-USD):
       - SPY (S&P 500): Overall market benchmark
       - QQQ (Nasdaq 100): Technology and growth stocks
       - IWM (Russell 2000): Small-cap stocks
       - VTV (Vanguard Value): Value stocks
       - VGLT (Long-Term Treasury): Bond market indicator
       - DIA (Dow Jones): Industrial/Value large-cap benchmark
       - BND (Total Bond Market): Overall fixed income trend
       - BTC-USD (Bitcoin): Speculative/risk appetite indicator
    
    2. Dual timeframe analysis for each index:
       - Daily chart: Price > 9/21 EMA: Bullish trend (+2 to Market Trend score per index)
       - Daily chart: Price < 9/21 EMA: Bearish trend (+2 if bearish setup per index)
       - 1-hour chart: Confirm daily trend or identify potential reversals/divergences
       - Note key support/resistance levels (prior highs/lows, 50-day SMA)
    
    3. Check for market divergences:
       - SPY vs QQQ: Growth vs Broad Market
       - SPY vs IWM: Small Cap vs Large Cap strength 
       - SPY vs VTV: Growth vs Value rotation
       - SPY vs DIA: Tech vs Industrial/Value
       - SPY vs VGLT/BND: Risk-on vs Risk-off relationship
       - BTC-USD vs SPY: Speculative risk appetite vs conventional equity markets
    
    4. Check VIX level:
       - VIX < 15: Very stable bullish trend (+3 to score)
       - VIX 15-20: Stable bullish trend (+2 to score)
       - VIX 20-25: Neutral volatility (0)
       - VIX 25-30: Elevated volatility, cautious approach (-2 to score)
       - VIX 30-35: High volatility, half position sizes (-4 to score)
       - VIX > 35: Flag as potential skip unless justified by high Gamble Score (-10 to score)
       - 5-day VIX trend: Rising/falling pattern and implications
    
    5. Options sentiment analysis:
       - Call/Put IV Skew: Compare IV of calls vs. puts
       - Call IV > Put IV by >5%: Strongly bullish direction (+7 to Sentiment)
       - Call IV > Put IV by <5%: Mildly bullish direction (+3 to Sentiment)
       - Put IV > Call IV by <5%: Mildly bearish direction (+3 to Sentiment)
       - Put IV > Call IV by >5%: Strongly bearish direction (+7 to Sentiment)
       - Call/Put Volume Ratio > 1.5: Strong bullish bias (+7)
       - Call/Put Volume Ratio 1.1-1.5: Mild bullish bias (+3)
       - Call/Put Volume Ratio 0.9-1.1: Neutral (0)
       - Call/Put Volume Ratio 0.5-0.9: Mild bearish bias (+3)
       - Call/Put Volume Ratio < 0.5: Strong bearish bias (+7)
       - IV vs Historical Volatility: Overpriced options signal credit spread opportunities
       - Open Interest trends: Rising OI in OTM puts or calls as directional indicator
    
    6. Sector Rotation Analysis:
       - Identify the best and worst performing indices
       - Analyze what this means for market sentiment
       - QQQ leading: Tech/Growth leadership (favors bull put spreads in tech)
       - IWM leading: Small cap strength, risk-on environment (wider market for bull put spreads)
       - VTV/DIA leading: Defensive positioning, value over growth (consider bear call spreads in tech)
       - VGLT/BND leading: Safety trade, potential risk-off environment (reduce position sizes)
       - BTC-USD leading: Speculative risk appetite strong (can be a leading indicator for equities)
       - Link to credit spreads: Leading sectors for bullish puts, lagging for bearish calls
    
    7. Credit Spread Trading Outlook:
       - Trend alignment: Bullish (sell put spreads), bearish (sell call spreads), neutral (iron condors)
       - Strike selection guidance based on support/resistance and IV
       - Position sizing recommendations based on VIX
       - Ideal DTE (days to expiration) based on current market conditions
       - Target premium and max loss percentages
    
    Return a structured analysis with the following EXACT sections:
    
    Market Trend: [bullish/bearish/neutral]
    Market Trend Score: [number out of 30]
    
    VIX Analysis: 
    [Your assessment of VIX implications for credit spreads]
    
    Risk Adjustment: 
    [standard/half size/skip]
    
    Sector Rotation:
    [Detailed analysis of which sectors are leading/lagging and credit spread opportunities]
    
    Major Index Analysis:
    SPY: [trend on daily and 1hr charts, key support/resistance levels]
    QQQ: [trend on daily and 1hr charts, key support/resistance levels]
    IWM: [trend on daily and 1hr charts, key support/resistance levels]
    VTV: [trend on daily and 1hr charts, key support/resistance levels]
    DIA: [trend on daily and 1hr charts, key support/resistance levels]
    VGLT: [trend on daily and 1hr charts, key support/resistance levels]
    BND: [trend on daily and 1hr charts, key support/resistance levels]
    BTC-USD: [trend on daily and 1hr charts, key support/resistance levels]
    
    Market Divergences:
    [Identify any significant divergences between indices and trading implications]
    
    Options Market Sentiment:
    [Analysis of IV skew, volume ratios, IV vs HV, and OI trends]
    
    Credit Spread Trading Outlook:
    [Specific recommendations for credit spread strategies, strike selection, position sizing]
    '''

def get_spy_options_prompt(options_data):
    """Get prompt template for analyzing SPY options data for credit spread opportunities."""
    return f'''
    Analyze SPY options data to determine optimal credit spread strategies:
    
    {options_data}
    
    Follow these rules exactly:
    1. IV Skew Analysis:
       - Compare IV of 20-30 delta calls vs. puts
       - Call IV > Put IV: Bullish bias (favor bull put spreads)
       - Put IV > Call IV: Bearish bias (favor bear call spreads)
       - Calculate IV percentile relative to past 30 days (high IV favors credit spreads)
    
    2. Volume/Open Interest Patterns:
       - Call Volume > Put Volume by >1.5x: Strong bullish bias
       - Put Volume > Call Volume by >1.5x: Strong bearish bias
       - Rising OI in OTM puts: Support level / put wall formation
       - Rising OI in OTM calls: Resistance level / call wall formation
    
    3. Delta Analysis:
       - Identify optimal short strike deltas (0.20-0.30 for balanced risk/reward)
       - Find support/resistance levels based on option activity
       - Rising call delta, falling put delta: Bullish momentum
       - Falling call delta, rising put delta: Bearish momentum
    
    4. IV vs Historical Volatility:
       - IV > HV by >20%: Options overpriced, good for selling premium (credit spreads)
       - IV < HV: Options potentially underpriced, caution with credit spreads
    
    5. Credit Spread Recommendations:
       - For bull put spreads: Identify optimal short put strikes below support
       - For bear call spreads: Identify optimal short call strikes above resistance
       - Optimal DTE (days to expiration) based on current IV environment
       - Premium targets (1-2% of collateral optimal)
    
    Return:
    1. Overall market direction prediction (bullish/bearish/neutral)
    2. Sentiment score adjustment
    3. Technical score adjustment
    4. Confidence level (high/medium/low)
    5. Optimal credit spread strategy
    6. Recommended strike selection parameters
    7. Position sizing guidance
    '''

def get_market_data_prompt(market_data):
    """Get prompt template for analyzing general market data tailored for credit spread traders."""
    return f'''
    Analyze the following market data and provide credit spread trading insights:
    
    {market_data}
    
    Please include:
    1. Key market trends and their implications for credit spread strategies
    2. Specific credit spread opportunities (bull puts in bullish markets, bear calls in bearish markets)
    3. Risk factors to consider for spread width and position sizing
    4. Support/resistance levels to use as strike selection guidance
    5. Volatility environment assessment and premium expectations
    6. Optimal DTE (days to expiration) in current conditions
    
    For each potential credit spread opportunity, specify:
    - Underlying index/stock
    - Spread direction (bull put or bear call)
    - Target delta range for short strikes
    - Recommended spread width based on ATR/volatility
    - Position sizing recommendation (% of portfolio)
    - Probability of profit target (70-80% optimal)
    - Target premium as % of spread width (8-12% ideal)
    ''' 