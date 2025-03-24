"""
Stock analysis prompt templates for Gemini API.
"""

def get_stock_analysis_prompt(stock_data, market_context):
    """Get prompt template for analyzing individual stock data."""
    return f'''
    Analyze the underlying stock with this data:
    
    Stock Data: {stock_data}
    Market Context: {market_context}
    
    Follow these rules exactly:
    1. Price Trend:
       - Price > 9/21 EMA: Bullish stock trend (+10 to Technicals)
       - Price < 9/21 EMA: Bearish stock trend (+10 to Technicals if bearish setup)
    
    2. Support/Resistance:
       - Price near support (within 2%): Bullish setup (+5)
       - Price near resistance: Bearish setup (+5)
    
    3. ATR (Average True Range):
       - ATR < 1% of price: Stable stock (+5 to Risk)
       - ATR > 2% of price: Volatile, tighten stop (-5 unless Gamble Score high)
    
    4. Fundamental Context:
       - Positive earnings/news: +5 to Sentiment
       - Negative news: -5 unless bearish setup aligns
    
    5. Market Alignment:
       - Check if stock trend aligns with SPY direction
    
    Return:
    1. Stock trend (bullish/bearish/neutral)
    2. Technical score (out of 15)
    3. Sentiment score (out of 10)
    4. Risk assessment
    5. Market alignment (aligned/contrary/neutral)
    '''

def get_stock_options_prompt(options_data, stock_analysis, market_analysis):
    """Get prompt template for analyzing stock options data."""
    return f'''
    Analyze credit spread opportunities with this data:
    
    Spread Options: {options_data}
    Stock Analysis: {stock_analysis}
    Market Analysis: {market_analysis}
    
    Follow these rules exactly according to the Quality Matrix and Gamble Matrix scoring:
    1. Match spread direction to SPY and stock analysis:
       - Bullish market/stock: Consider Bull Put Spreads
       - Bearish market/stock: Consider Bear Call Spreads
    
    2. Implied Volatility (IV):
       - IV > 30% (high premiums)
       - Prefer IV > 2x estimated 20-day HV
    
    3. Delta:
       - Short leg at 20–30 delta (65–80% OTM probability)
       - Buy leg 5–10 points further OTM
    
    4. Days to Expiration (DTE):
       - 7–15 days preferred
    
    5. Position Size:
       - Risk 1–2% ($200–$400 for $20,000 account)
       - Adjust based on VIX (half size if VIX > 25)
    
    6. Calculate scores based on the Quality Matrix (100 points total):
       - Market Analysis (15): Alignment with overall market trend, sector trends, impact of news
       - Risk Management (25): Clear stop-loss, position sizing, risk-reward ratio, contingency plans
       - Entry and Exit Points (15): Data-driven entry, clear exit strategy
       - Technical Indicators (15): Trend indicators, momentum indicators, volatility indicators
       - Fundamental Analysis (10): Company-specific drivers, macro environment
       - Probability of Success (10): Historical win rate, model prediction
       - Uniqueness and Edge (10): Novel insight, market inefficiency
    
    7. Calculate the Gamble Score if Quality Score is borderline (70-80 points):
       - Hype and Momentum (30): Social media, retail volume, price action
       - Volatility Explosion (25): IV spike, catalyst proximity
       - Risk-Reward Potential (20): Upside vs. downside ratio
       - Timing Edge (15): Freshness of play, entry precision
       - Basic Survival Instinct (10): Max loss cap, quick exit trigger
    
    Return:
    1. Recommended spread type (Bull Put/Bear Call)
    2. Specific strikes and expiration
    3. Quality Score (threshold > 80)
    4. Gamble Score (threshold > 70 with reduced size)
    5. Success Probability (threshold > 70%)
    6. Position size recommendation
    7. Profit target and stop loss levels
    ''' 