"""
Market analysis prompt templates for Gemini API.
"""

def get_market_trend_prompt(market_data):
    """Get prompt template for analyzing SPY market trend."""
    return f'''
    Analyze SPY market trend based on this data:
    
    {market_data}
    
    Follow these rules exactly:
    1. Check 9/21 EMA on 1-hour chart
       - Price > 9/21 EMA: Bullish market trend (+10 to Market Trend score)
       - Price < 9/21 EMA: Bearish market trend (+10 if bearish setup)
       - Flat/No crossover: Neutral (no bonus)
    
    2. Check VIX level:
       - VIX < 20: Stable bullish trend (+5)
       - VIX 20–25: Neutral volatility
       - VIX > 25: High volatility, cautious approach (-5 unless size halved)
       - VIX > 35: Flag as potential skip unless justified by high Gamble Score
    
    3. Options sentiment analysis:
       - Call/Put IV Skew: Compare IV of calls vs. puts
       - Call IV > Put IV: Bullish direction (+5 to Sentiment)
       - Put IV > Call IV: Bearish direction (+5 to Sentiment)
       - Call/Put Volume Ratio > 1.1: Bullish bias (+5)
       - Call/Put Volume Ratio < 0.9: Bearish bias (+5)
    
    Return:
    1. Overall market trend (bullish/bearish/neutral)
    2. Market Trend score (out of 20)
    3. VIX assessment and impact on trading
    4. Risk management adjustment recommendation
    5. Detailed analysis explaining your reasoning
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