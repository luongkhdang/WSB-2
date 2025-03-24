"""
Trade plan prompt templates for Gemini API.
"""

def get_trade_plan_prompt(spy_analysis, options_analysis, stock_analysis, spread_analysis, ticker):
    """Get prompt template for generating a complete trade plan."""
    return f'''
    Create a comprehensive trade plan based on all analyses for ticker: {ticker}
    
    SPY Analysis: {spy_analysis}
    SPY Options Analysis: {options_analysis}
    Stock Analysis: {stock_analysis}
    Spread Analysis: {spread_analysis}
    
    Follow this template based on the Rule Book for AI-Driven Credit Spread Trading Strategy:
    
    1. MARKET CONTEXT
    - SPY Trend: [bullish/bearish/neutral]
    - Market Direction: [direction with evidence from SPY EMAs and VIX]
    - VIX Context: [current VIX and implications for position sizing]
    
    2. UNDERLYING STOCK ANALYSIS ({ticker})
    - Technical Position: [support/resistance, EMA status]
    - Sentiment Factors: [news, earnings, catalysts]
    - Volatility Assessment: [ATR relative to price, stability]
    
    3. CREDIT SPREAD RECOMMENDATION
    - Spread Type: [Bull Put/Bear Call]
    - Strikes and Expiration: [specific strikes and date]
    - Entry Criteria: [exact price levels to enter]
    - Position Size: [$amount based on account risk of 1-2% ($200-$400)]
    
    4. EXIT STRATEGY
    - Profit Target: [exact credit amount to exit at 50% of max credit]
    - Stop Loss: [exit at 2x credit received]
    - Time-based Exit: [exit at 2 days to expiration]
    
    5. RISK ASSESSMENT
    - Quality Score: [score/100 from Quality Matrix]
    - Success Probability: [probability %]
    - Maximum Risk: [$amount and % of account]
    
    6. TRADE EXECUTION CHECKLIST
    - Pre-trade verification steps
    - Order types to use
    - Position monitoring schedule
    
    Make this extremely actionable for a trader with a $20,000 account targeting 40-60% annual returns.
    ''' 