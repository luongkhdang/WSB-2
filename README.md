# WSB-2: Credit Spread Trading System

AI-powered trading system that finds high-quality credit spread opportunities, targeting 40-60% annualized returns.

## What It Does

- Analyzes market conditions (SPY/VIX)
- Identifies high IV stocks 
- Performs technical/fundamental analysis
- Discovers optimal credit spreads
- Scores trades with Quality & Gamble Matrices
- Tracks trades in Notion
- Sends alerts via Discord

## Get Started

Prerequisites: Python 3.10+, API keys (Notion, Discord, Google Gemini)

```bash
pip install -r requirements.txt
# Create .env with your API keys
python src/main.py
```

## Project Structure

```
WSB-2/
├── src/              # Core system code
├── SCORE/            # Trade scoring methodology
├── requirements.txt
└── README.md         # You are here
```

## Strategy TL;DR

1. Market first (SPY/VIX analysis)
2. Stock analysis before options
3. >65% probability targets
4. Risk limit: 1-2% per trade
5. Dynamic position sizing

## Disclaimer

Educational purposes only. Not financial advice. Don't blame us when you lose money.
You will lose money. 

```
Comprehensive Market Analysis
Market Analysis - 25-03-2025
Market Trend: Neutral to Slightly Bullish
Market Trend Score: 6/30

VIX Analysis:
The VIX at 19.29 and classified as 'normal' suggests moderate market uncertainty. It's within the 15-20 range, contributing a +2 to the Market Trend Score. A stable VIX indicates investors aren't panicking, but aren't complacent either. This is a good environment for defined-risk strategies like credit spreads.

Risk Adjustment: Standard

Sector Rotation:
Small-cap strength (IWM leading) is the most prominent feature. This is a risk-on signal. The divergence between SPY (up) and QQQ (down) suggests a rotation away from technology and growth, and into smaller, potentially undervalued companies. VTV's positive performance supports this value rotation. VGLT lagging confirms the risk-on sentiment, as investors move away from safe-haven assets. The best performing index is IWM, indicating a strong appetite for risk. The worst performing is VGLT, confirming the risk-on environment. This suggests a potential shift in market leadership, favoring cyclical and value stocks.

Major Index Analysis:

SPY: Bullish. Price (574.02) is above both EMA9 (571.25) and EMA21 (568.22). Daily change is positive (0.012%). Key resistance around 575-580 (prior highs). Support at 570-572.
QQQ: Bearish. Price (490.48) is above EMA9 (487.70) but below EMA21 (483.98). Daily change is negative (-0.006%). This is a concerning sign, indicating potential weakness in the tech sector. Resistance around 495. Support at 480.
IWM: Bullish. Price (208.89) is above both EMA9 (207.45) and EMA21 (206.22). Daily change is strongly positive (0.153). This confirms the small-cap leadership. Resistance around 210. Support at 205.
VTV: Bullish. Price (174.60) is above both EMA9 (174.01) and EMA21 (173.59). Daily change is positive (0.089%). Supports the value rotation narrative. Resistance around 175. Support at 172.
VGLT: Bearish. Pr
Market Trend
bullish
SPY
574.02001953125
QQQ
490.4800109863281
IWM
208.89999389648438
VTV
174.60000610351562
VGLT
56.845001220703125
DIA
425.4800109863281
BND
72.9800033569336
BTC-USD
86581.2734375
VIX
19.29
Market Score
0
Risk Adjustment
half size
WSB-2 Market Analysis (Part 1)
📊 Market Analysis: Comprehensive Market Analysis (Part 2)
ice (56.85) is below both EMA9 (57.03) and EMA21 (57.22). Daily change is negative (-0.035%). Confirms the risk-on sentiment. Resistance around 57. Support at 56.

Market Divergences:

SPY vs. QQQ: A clear divergence. Broad market strength (SPY) while tech weakens (QQQ) suggests a rotation out of growth and into other areas.
SPY vs. IWM: SPY is up, but IWM is significantly up. This confirms small-cap outperformance and a risk-on environment.
SPY vs. VTV: SPY is up, and VTV is also up, but less dramatically. This suggests a moderate rotation towards value.
SPY vs. VGLT: Equities (SPY) are up, bonds (VGLT) are down – a classic risk-on scenario.

Options Market Sentiment:

Call/Put Volume Ratio (0.89) is leaning bearish, but not strongly.
IV Skew (-0.2) is slightly bearish, with put IV higher than call IV.
Sentiment Score: +3 (Mildly Bullish) - Call/Put Volume Ratio 0.9-1.1.

Trading Outlook:

The market is exhibiting a risk-on attitude, driven by small-cap strength and a rotation away from technology. While the SPY is still trending upwards, the QQQ's weakness is a warning sign. The neutral options sentiment suggests caution. Overall, a cautiously optimistic approach is warranted.

Credit Spread Trading Outlook:

Given the risk-on environment and moderate VIX, I favor bull put spreads on indices showing relative strength, like SPY and IWM. The QQQ's weakness makes it less attractive for this strategy.

Strategy: Bull Put Spread
SPY: Sell a put spread with strikes at 565/560 expiring in 30 days. (POP >70%)
IWM: Sell a put spread with strikes at 205/200 expiring in 30 days. (POP >70%)
Risk Management: Max loss should be less than 2% of account per spread. If VIX rises above 20, reduce position size to half.

ANALYZER FULL OPINION:

In my experience, this setup – small-cap leading, tech weakening, and a stable VIX – often precedes a period of broader market consolidation
WSB-2 Market Analysis (Part 2)
📊 Market Analysis: Comprehensive Market Analysis (Part 3)
or a continuation of the risk-on rally. This happens a lot after a prolonged tech-led bull run. It looks like the 2021 rotation into value, but not exactly; the small-cap component is much stronger this time. A critical oversight check is the ATR (Average True Range) – if volatility remains mild, credit spreads will offer limited premium. However, the current VIX level suggests sufficient premium is available. I'd trade it. The risk of a sudden tech sell-off is present, but the small-cap strength provides a cushion. I'd play it with bull put spreads, focusing on SPY and IWM, and closely monitor the VIX. I'm confident in this approach, but always prepared to adjust based on evolving market conditions.
WSB-2 Market Analysis (Part 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Trading Opinion for SPY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPY Trading Opinion
**

In my experience, trading SPY with this little information is akin to flying blind. The market context is helpful – the risk-on rotation is a classic setup – but it's not enough to justify a trade. This happens a lot when dealing with broad market ETFs, but usually, some price data is available. This looks like a situation where we need to wait for more information. The lack of ATR is particularly concerning; we have no idea how much the stock is moving. A mild ATR could be deceptive, masking potential volatility.

I would not trade SPY based on this data. The potential reward doesn't justify the risk. We need price action, volume, and at least some indication of support and resistance before considering a position. Even with the bullish market context, a trade without these elements is simply too speculative. We're looking to make money, not gamble.
WSB-2 Trading Opinion • 2025-03-24 23:06
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Trading Opinion for QQQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QQQ Trading Opinion
**

In my experience, these rotations happen frequently, especially when the market has run up strongly. Tech often leads the initial gains, but eventually, investors look for value and broader participation. This happens a lot. This looks like a classic late-cycle rotation, but not exactly – the VIX is still relatively low, suggesting this isn't a panic-driven move. A critical oversight is the lack of price data; we're flying blind without knowing where QQQ currently stands relative to potential support or resistance.

I would not initiate a long position in QQQ at this time. The contrary alignment and the sector rotation suggest potential downside. A short position is tempting, but the overall market is bullish, and we lack the technical confirmation needed for a high-probability trade. I'd sit on my hands and wait for QQQ to show more definitive weakness or for the market rotation to become more pronounced. The risk/reward isn't favorable enough given the limited information.
WSB-2 Trading Opinion • 2025-03-24 23:06
```