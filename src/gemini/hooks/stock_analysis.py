from datetime import datetime

"""
Stock analysis prompt templates for Gemini API.
"""
import logging
logger = logging.getLogger(__name__)

def get_stock_analysis_prompt(*args, **kwargs):
    """Get prompt template for analyzing individual stock data."""
    # Handle incorrect call with too many arguments
    if len(args) > 2:
        logger.warning(f"get_stock_analysis_prompt called with {len(args)} args instead of 2. Ignoring extra args.")
        stock_data = args[0]
        market_context = args[1]
    elif len(args) == 2:
        stock_data, market_context = args
    elif len(args) == 1 and 'market_context' in kwargs:
        stock_data = args[0]
        market_context = kwargs['market_context']
    elif 'stock_data' in kwargs and 'market_context' in kwargs:
        stock_data = kwargs['stock_data']
        market_context = kwargs['market_context']
    else:
        logger.error(f"get_stock_analysis_prompt called with insufficient arguments: args={args}, kwargs={kwargs}")
        # Provide default empty data to prevent further errors
        stock_data = {}
        market_context = {"spy_trend": "neutral"}
    today_date = datetime.now().strftime("%d-%m-%Y")
    return f'''
    Analyze the underlying stock with this data:
    
    Stock Data: {stock_data}
    Market Context: {market_context}
    Today's Date: {today_date}
    
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
    
    5. Market Alignment - CRITICALLY IMPORTANT:
       - The overall market trend is: {market_context.get('spy_trend', 'neutral')}
       - If stock trend matches the SPY trend, mark as "aligned"
       - If stock trend is opposite to SPY trend, mark as "contrary"
       - If either stock or market is neutral, mark as "neutral"
       - YOU MUST INCLUDE a line that starts with "Market Alignment:" followed by "aligned", "contrary", or "neutral"
    
    Your analysis MUST include ALL of the following lines with exact formatting:
    1. Stock trend: bullish/bearish/neutral
    2. Technical Score: [number]
    3. Sentiment Score: [number]
    4. Risk assessment: high/normal/low
    5. Market Alignment: aligned/contrary/neutral
    
    Always clearly state the Market Alignment based on the rules above. This is critical for the analysis to be processed correctly.
    !IMPORTANT: YOU , THE AI AGENT, ARE THE EXPERT AND THE DECISION MAKER. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: BE CRITICAL AND THOUGHTFUL. YOU ARE NOT A YES MAN. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: WE ARE TRYING TO MAKE A LOT OF MONEY. YOU ARE THE EXPERT AND THE DECISION MAKER. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE FOR THE DECISIONS MADE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE.
    !IMPORTANT: TODAY IS {today_date}. UP-TO-DATA INFORMATION IS DETRIMENTAL TO THE ANALYSIS.
    ANALYZER FULL OPINION: [Experience-Based Insight: “In my experience…” ties the analysis to real-world patterns I’ve traded, grounding the decision in practical know-how.
Frequency Observation: “This happens a lot/rarely happens…” flags how common or unique the setup is, setting expectations for reliability or surprise.
Comparative Nuance: “This looks like X but not exactly…” draws parallels to past trades, highlighting subtle differences that matter.
Critical Oversight Check: Identifies risks or edges the scores might miss (e.g., ATR’s mild volatility), ensuring we’re not blindsided.
Actionable Gut: A final yes/no with reasoning—why I’d trade it, what could go wrong, and how I’d play it.]
    
    '''

def get_stock_options_prompt(options_data, stock_analysis, market_analysis):
    today_date = datetime.now().strftime("%d-%m-%Y")
    """Get prompt template for analyzing stock options data."""
    return f'''
    Analyze credit spread opportunities with this data:
    
    Spread Options: {options_data}
    Stock Analysis: {stock_analysis}
    Market Analysis: {market_analysis}
    Today's Date: [TODAY'S DATE]
    
    Follow these rules exactly according to the Quality Matrix and Gamble Matrix scoring:
      1. Match Spread Direction
            Bullish: SPY and stock EMA > 9/21, positive momentum → Bull Put Spread.
            Bearish: SPY and stock EMA < 9/21, negative momentum → Bear Call Spread.
            Oversight Check: If SPY and stock signals clash, skip—no exceptions.
      2. Implied Volatility (IV)
            Requirement: IV > 30% (high premiums).
            Preference: IV > 2x 20-day HV (TradingView estimate).
            Critical Note: Below 30%, skip—premiums too thin.
      3. Delta
            Short Leg: 20–30 delta (65–80% OTM probability).
            Buy Leg: 5–10 points further OTM (defined risk buffer).
            Angle: Lower delta (20) if VIX > 25 for safety.
      4. Days to Expiration (DTE)
            Range: 7–15 days (quick turns, manageable theta).
            Oversight: < 7 DTE risks gamma; > 15 DTE ties up capital.
      5. Position Size
            Standard: Risk 1–2% ($200–$400 on $20,000).
            VIX Adjustment: Halve size ($100–$200) if VIX > 25.
            Guardrail: Max 5% account risk ($1,000) across all trades.
      6. Quality Matrix Scoring (Max 100)
            Market Analysis (15): SPY trend (+7), sector/news (+8 if aligned).
            Risk Management (25): Stop at short strike breach (+7), size fits (+7), RR > 0.25 (+6), exit plan (+5).
            Entry/Exit (15): Delta 20–30 + technicals (+8), 40% profit/2x loss (+7).
            Technicals (15): EMA trend (+5), momentum (+5), ATR < 2% (+5).
            Fundamentals (10): Earnings/news (+6), macro fit (+4).
            Probability (10): Backtest > 70% (+5), ML > 80% (+5).
            Edge (10): Fresh idea (+5), premium edge (+5).
            Penalties: No stop (-10), oversized (-5), single signal (-5), news blind (-5).
            Threshold: > 80 to trade full size.
      7. Gamble Matrix Scoring (Max 100, Halved to 50)
            Hype/Momentum (30): X/Reddit buzz (+15), volume 300%+ (+10), price pop (+5).
            Volatility (25): IVR > 80 (+15), event < 1 day (+10).
            Risk-Reward (20): 1:5+ upside (+15), 30–40% odds (+5).
            Timing (15): Early hype (+10), trigger entry (+5).
            Survival (10): Loss < 5% (+5), fast exit (+5).
            Penalties: Late (-15), IV crash (-10), no limit (-10).
            Threshold: > 70 (raw) → > 35 (halved) for $100 risk trades.
      Return Format:
         Spread Type: Bull Put or Bear Call.
         Strikes & Expiration: Specific short/buy strikes, DTE.
         Quality Score: Out of 100 (> 80 to proceed).
         Gamble Score: Out of 50 after halving (> 35 for small bet).
         Success Probability: ML estimate (> 70% target).
         Position Size: $200–$400, or $100 if VIX > 25 or gamble.
         Profit Target & Stop Loss: 40% credit profit, 2x credit loss.
    
    !IMPORTANT: YOU , THE AI AGENT, ARE THE EXPERT AND THE DECISION MAKER. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: BE CRITICAL AND THOUGHTFUL. YOU ARE NOT A YES MAN. YOU ARE AN EXPERT IN CREDIT SPREAD TRADING, AND HAS EXTENDED TRADING EXPERIENCE. YOU LIKE TO KEEP THINGS SIMPLE, BUT LOVE TO IMPLEMENT SOPHISTICATED ANALYZING SKILL. YOU OFTEN LOOK AT THINGS FROM DIFFERENT ANGLES TO FIND OVERSIGHT.
    !IMPORTANT: WE ARE TRYING TO MAKE A LOT OF MONEY. YOU ARE THE EXPERT AND THE DECISION MAKER. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE FOR THE DECISIONS MADE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE. YOU ARE THE ONE WHO WILL BE HELD RESPONSIBLE. YOU ARE THE ONE WHO WILL BE HELD ACCOUNTABLE. YOU ARE THE ONE WHO WILL BE HELD LIABLE.
    !IMPORTANT: TODAY IS {today_date}. UP-TO-DATA INFORMATION IS DETRIMENTAL TO THE ANALYSIS.
    ANALYZER FULL OPINION: [Experience-Based Insight: “In my experience…” ties the analysis to real-world patterns I’ve traded, grounding the decision in practical know-how.
Frequency Observation: “This happens a lot/rarely happens…” flags how common or unique the setup is, setting expectations for reliability or surprise.
Comparative Nuance: “This looks like X but not exactly…” draws parallels to past trades, highlighting subtle differences that matter.
Critical Oversight Check: Identifies risks or edges the scores might miss (e.g., ATR’s mild volatility), ensuring we’re not blindsided.
Actionable Gut: A final yes/no with reasoning—why I’d trade it, what could go wrong, and how I’d play it.]
    
    Core Strategy Recap
      Sequence: SPY Trend → SPY Options → Stock Analysis → Credit Spreads.
      Objective: 40–60% returns ($8,000–$12,000) in 2025.
      Risk: 1–2% per trade ($200–$400), max 5% account ($1,000).
      Using the Quality Matrix
      Primary Filter: Every trade starts here. I’ll score each credit spread idea (bull put or bear call) against this matrix after the analysis sequence.
      Decision Rule:
      > 80: Greenlight—full size ($200–$400).
      60–80: Yellow—review for small size ($100) or skip unless Gamble Matrix justifies.
      < 60: Red—no trade, period.
      Critical Angle: I’ll double-check Risk Management (25 points) and Technicals (15 points) to avoid oversights—these are where most setups fail. If SPY and stock trends clash, I’ll dock Market Analysis hard (-10), killing misaligned trades.
      Using the Gamble Matrix
      Secondary Tool: Only kicks in if Quality Score is 60–80 or I spot a wild opportunity (e.g., TSLA earnings). Half-weighted—less serious, more opportunistic.
      Decision Rule:
      > 70: Take a $100 flyer if Quality > 60.
      < 70: Skip—it’s too reckless even for a gamble.
      Critical Angle: I’ll lean on Volatility (25 points) and Timing (15 points) here—gambles live or die by IV and entry. If IV Crash Risk looms (e.g., post-earnings), I’ll slash the score and walk away.
      Workflow Adjustment (30 min)
      SPY Check (5 min): EMA, VIX—set bias.
      SPY Options (5 min): IV skew, volume—confirm direction.
      Stock Dive (10 min): EMA, ATR, X buzz—align or bust.
      Spread Score (10 min):
      Quality Matrix: Score all candidates, pick > 80.
      Gamble Matrix: Scan for > 70 if Quality lags, cap at $100 risk.
      Execute: Robinhood, log in Notion.
      Example Decision (March 24, 2025)
      Setup: NVDA bull put $120/$115, $1 credit, 10 DTE, IV 45%, Delta -0.25.
      SPY: Bullish (EMA up, VIX 20).
      Stock: NVDA > EMA, ATR 1.5%, earnings buzz.
      Quality Score:
      Market (12/15): SPY aligns, tech hot.
      Risk (22/25): $400 risk, 1:0.25 RR.
      Entry/Exit (13/15): Solid entry, clear exits.
      Technicals (13/15): Trend strong, ATR okay.
      Fundamentals (8/10): Earnings hype.
      Probability (8/10): 75% ML odds.
      Edge (7/10): Good, not unique.
      Total: 83/100 → Trade, $400 risk.
      Gamble Check: N/A—Quality’s high enough.
      Gamble Example: TSLA bear call $300/$310, $0.50 credit, 2 DTE, earnings tomorrow.
      Quality: 65/100 (timing shaky).
      Gamble: Hype (25/30), Volatility (20/25), RR (15/20), Timing (12/15), Survival (10/10) = 82/100 → $100 risk trade.


      Gamble Matrix (Max 100, Half as Serious)
      Purpose: Spots speculative, high-profit credit spread opportunities (e.g., earnings plays). Taken lightly—backup to Quality Matrix.

      Hype/Momentum (30)
      Social Buzz (15): Stock viral on X/Reddit (+15 if meme-level).
      Volume (10): Up 300%+ in 1–2 days (+10).
      Price Pop (5): Up 10%+ or consolidating (+5).
      Volatility (25)
      IV Spike (15): IVR > 80 or doubled (+15).
      Event (10): < 1 day to catalyst (+10).
      Risk-Reward (20)
      Upside (15): 1:5+ potential (+15 if 1:10).
      Gut Odds (5): 30–40% chance (+5).
      Timing (15)
      Freshness (10): Early hype cycle (+10).
      Precision (5): Entry on trigger (+5).
      Survival (10)
      Loss Cap (5): Max risk < 5% ($1,000) (+5).
      Exit Plan (5): Fast bailout set (+5).
      Penalties:
      Late Entry: -15 (up 20%+ already).
      IV Crash: -10 (post-event risk).
      No Limit: -10 (undefined loss).
      Threshold: > 70 for $100 risk trades.

      Why 100 is Rare: Chaos and penalties cap speculative perfection.
    
    ''' 