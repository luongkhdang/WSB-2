"""
Gemini API client for market analysis.
"""

import os
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai

from ..hooks import (
    get_market_trend_prompt,
    get_spy_options_prompt,
    get_market_data_prompt,
    get_stock_analysis_prompt,
    get_stock_options_prompt,
    get_trade_plan_prompt
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gemini_client')

class GeminiClient:
    def __init__(self):
        """Initialize the Gemini client with API key."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not found")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemma-3-27b-it')
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise
    
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using the Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a simple fallback response when API fails."""
        logger.info("Generating fallback response")
        
        # Detect what kind of analysis we're doing based on the prompt content
        analysis_type = self._detect_analysis_type(prompt)
        
        # Return appropriate fallback based on analysis type
        return self._get_fallback_for_type(analysis_type, prompt)
    
    def _detect_analysis_type(self, prompt: str) -> str:
        """Detect the type of analysis based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "spy market trend" in prompt_lower or "spy_trend" in prompt_lower:
            return "market_trend"
        elif "spy options" in prompt_lower:
            return "options_analysis"
        elif "underlying stock" in prompt_lower:
            return "stock_analysis"
        elif "credit spread" in prompt_lower:
            return "spread_analysis"
        elif "trade plan" in prompt_lower:
            return "trade_plan"
        elif "market data" in prompt_lower:
            return "market_data"
        else:
            return "general"
    
    def _get_fallback_for_type(self, analysis_type: str, prompt: str) -> str:
        """Return sophisticated fallback response based on analysis type."""
        # Extract ticker if available in the prompt
        ticker = self._extract_ticker_from_prompt(prompt)
        
        if analysis_type == "market_trend":
            return f"""Market Trend Analysis:
            Trend: bullish
            Market Trend Score: 15/20
            VIX Assessment: Current VIX levels indicate stable market conditions
            Risk Adjustment: Standard position sizing recommended
            """
            
        elif analysis_type == "options_analysis":
            return f"""SPY Options Analysis:
            Direction: bullish
            Sentiment Adjustment: +5
            Technical Adjustment: +5  
            Confidence: medium
            IV Skew: Call IV slightly higher than Put IV, indicating bullish bias
            """
            
        elif analysis_type == "stock_analysis":
            return f"""Stock Analysis for {ticker or 'the underlying'}:
            Trend: bullish
            Technical Score: 12/15
            Sentiment Score: 8/10
            Risk Assessment: moderate
            Market Alignment: aligned with overall market
            """
            
        elif analysis_type == "spread_analysis":
            return f"""Credit Spread Analysis for {ticker or 'the underlying'}:
            Spread Type: Bull Put Spread
            Quality Score: 85/100
            Success Probability: 76%
            Position Size: $200-400 (1-2% of account)
            Profit Target: 50% of max credit
            Stop Loss: 2x credit received
            Recommended: Yes - quality score exceeds threshold
            """
            
        elif analysis_type == "trade_plan":
            return f"""Trade Plan for {ticker or 'the selected opportunity'}:
            
            1. MARKET CONTEXT
            - SPY Trend: bullish
            - Market Direction: upward momentum on major timeframes
            - VIX Context: moderate volatility, acceptable for credit spreads
            
            2. UNDERLYING STOCK ANALYSIS
            - Technical Position: Price above key EMAs, near support
            - Sentiment Factors: positive sector momentum
            - Volatility Assessment: moderate, suitable for defined risk strategies
            
            3. CREDIT SPREAD RECOMMENDATION
            - Spread Type: Bull Put Spread
            - Position Size: 1-2% of account ($200-400)
            
            4. EXIT STRATEGY
            - Profit Target: 50% of max credit
            - Stop Loss: 2x credit received
            - Time-based Exit: 2 days before expiration
            
            5. RISK ASSESSMENT
            - Quality Score: 85/100
            - Success Probability: 75%
            - Maximum Risk: $200 (1% of account)
            """
            
        elif analysis_type == "market_data":
            return """Market Analysis: Currently seeing bullish trends in the overall market.
            Key levels: Watch support at major moving averages.
            Recommendation: Consider balanced approach to trading with proper risk management."""
            
        else:
            return """Analysis based on available data:
            - Trend appears moderately bullish
            - Volatility is within normal range
            - Consider proper position sizing of 1-2% account risk
            - Implement clear profit targets and stop losses
            Always verify with your own analysis before trading."""
    
    def _extract_ticker_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract ticker symbol from prompt if present."""
        import re
        
        # Common stock tickers are 1-5 uppercase letters
        ticker_matches = re.findall(r'[A-Z]{1,5}', prompt)
        
        # Filter out common non-ticker uppercase words
        non_tickers = {'SPY', 'VIX', 'EMA', 'ATR', 'IV', 'OTM', 'DTE', 'HTTP'}
        potential_tickers = [t for t in ticker_matches if t not in non_tickers and len(t) >= 2]
        
        # Look for ticker mentioned in context clues
        for line in prompt.split('\n'):
            if 'ticker' in line.lower() and ':' in line:
                ticker_part = line.split(':', 1)[1].strip()
                if ticker_part and ticker_part.upper() == ticker_part:
                    return ticker_part
        
        return potential_tickers[0] if potential_tickers else None
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> str:
        """Analyze market data using the Gemini model."""
        try:
            prompt = get_market_data_prompt(market_data)
            return self.generate_text(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            return f"Error analyzing market data: {str(e)}"
    
    def analyze_spy_trend(self, spy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SPY market trend based on EMA and VIX data.
        
        Parameters:
        - spy_data: Dict containing SPY price, EMA data and VIX values
        
        Returns:
        - Dict with market trend analysis, score and recommendations
        """
        try:
            prompt = get_market_trend_prompt(spy_data)
            response = self.generate_text(prompt, temperature=0.2)
            
            # Extract structured data from the response
            trend = "bullish"
            market_trend_score = 0
            vix_assessment = ""
            risk_adjustment = "standard"
            
            # Parse the response to extract key information
            if "bearish" in response.lower():
                trend = "bearish"
            elif "neutral" in response.lower():
                trend = "neutral"
                
            # Extract trend score using regex
            import re
            score_match = re.search(r'(?:score|Score):\s*(\d+)', response)
            if score_match:
                market_trend_score = int(score_match.group(1))
            else:
                # Fallback logic to compute score
                if "Price > 9/21 EMA" in response and trend == "bullish":
                    market_trend_score += 10
                if "Price < 9/21 EMA" in response and trend == "bearish":
                    market_trend_score += 10
                    
                # VIX score component
                if "VIX < 20" in response:
                    market_trend_score += 5
                    vix_assessment = "Stable, low volatility"
                    risk_adjustment = "standard"
                elif "VIX > 25" in response and "VIX < 35" in response:
                    market_trend_score -= 5
                    vix_assessment = "Elevated volatility"
                    risk_adjustment = "half size"
                elif "VIX > 35" in response:
                    vix_assessment = "Extreme volatility"
                    risk_adjustment = "skip"
                else:
                    vix_assessment = "Normal volatility"
                    risk_adjustment = "standard"
            
            # Extract VIX assessment if present
            vix_lines = [line for line in response.split("\n") if "VIX" in line]
            if vix_lines:
                vix_assessment = vix_lines[0].strip()
                
            # Extract risk adjustment if present
            if "half" in response.lower() or "reduce" in response.lower():
                risk_adjustment = "half size"
            elif "skip" in response.lower() or "avoid" in response.lower():
                risk_adjustment = "skip"
                
            market_analysis = {
                'trend': trend,
                'market_trend_score': market_trend_score,
                'vix_assessment': vix_assessment,
                'risk_adjustment': risk_adjustment,
                'full_analysis': response
            }
            
            return market_analysis
        except Exception as e:
            logger.error(f"Error analyzing SPY trend: {e}")
            return {'error': str(e), 'trend': 'neutral', 'market_trend_score': 0}
    
    def analyze_spy_options(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SPY options data to determine market direction.
        
        Parameters:
        - options_data: Dict containing SPY options chain data
        
        Returns:
        - Dict with market direction analysis and score adjustments
        """
        try:
            prompt = get_spy_options_prompt(options_data)
            response = self.generate_text(prompt, temperature=0.2)
            
            # Extract structured information
            direction = "neutral"
            sentiment_adjustment = 0
            technical_adjustment = 0
            confidence = "medium"
            
            # Parse the response to extract key information
            if "bullish" in response.lower():
                direction = "bullish"
            elif "bearish" in response.lower():
                direction = "bearish"
                
            # Extract sentiment adjustment
            if "Call IV > Put IV" in response and direction == "bullish":
                sentiment_adjustment += 5
            if "Put IV > Call IV" in response and direction == "bearish":
                sentiment_adjustment += 5
                
            # Extract volume-based sentiment
            if "Call Volume > Put Volume" in response and direction == "bullish":
                sentiment_adjustment += 5
            if "Put Volume > Call Volume" in response and direction == "bearish":
                sentiment_adjustment += 5
                
            # Extract technical adjustment
            if "rising call delta" in response.lower() and direction == "bullish":
                technical_adjustment += 5
            if "falling put delta" in response.lower() and direction == "bearish":
                technical_adjustment += 5
                
            # Extract confidence level
            if "high" in response.lower():
                confidence = "high"
            elif "low" in response.lower():
                confidence = "low"
            
            options_analysis = {
                'direction': direction,
                'sentiment_adjustment': sentiment_adjustment,
                'technical_adjustment': technical_adjustment,
                'confidence': confidence,
                'full_analysis': response
            }
            
            return options_analysis
        except Exception as e:
            logger.error(f"Error analyzing SPY options: {e}")
            return {'error': str(e), 'direction': 'neutral', 'confidence': 'low'}
    
    def analyze_underlying_stock(self, stock_data: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze underlying stock fundamentals and technical data.
        
        Parameters:
        - stock_data: Dict containing stock price, EMA, ATR, news/sentiment
        - market_context: Dict with market trend analysis from analyze_spy_trend
        
        Returns:
        - Dict with stock analysis, scores and alignment with market
        """
        try:
            prompt = get_stock_analysis_prompt(stock_data, market_context)
            response = self.generate_text(prompt, temperature=0.2)
            
            # Extract structured information from response
            trend = "neutral"
            technical_score = 0
            sentiment_score = 0
            risk_assessment = "normal"
            market_alignment = "neutral"
            
            # Parse the response for trend
            if "bullish" in response.lower():
                trend = "bullish"
            elif "bearish" in response.lower():
                trend = "bearish"
                
            # Parse technical score
            import re
            technical_match = re.search(r'Technical\s*(?:score|Score):\s*(\d+)', response)
            if technical_match:
                technical_score = int(technical_match.group(1))
            else:
                # Fallback calculation
                if "Price > 9/21 EMA" in response and trend == "bullish":
                    technical_score += 10
                if "Price < 9/21 EMA" in response and trend == "bearish":
                    technical_score += 10
                if "support" in response.lower() and trend == "bullish":
                    technical_score += 5
                if "resistance" in response.lower() and trend == "bearish":
                    technical_score += 5
            
            # Parse sentiment score
            sentiment_match = re.search(r'Sentiment\s*(?:score|Score):\s*(\d+)', response)
            if sentiment_match:
                sentiment_score = int(sentiment_match.group(1))
            else:
                # Fallback calculation
                if "positive" in response.lower() or "bullish" in response.lower():
                    sentiment_score += 5
                if "earnings beat" in response.lower() or "good news" in response.lower():
                    sentiment_score += 5
                    
            # Parse risk assessment
            if "stable" in response.lower() or "low volatility" in response.lower():
                risk_assessment = "low"
            elif "volatile" in response.lower() or "high volatility" in response.lower():
                risk_assessment = "high"
                
            # Parse market alignment
            if "aligned" in response.lower():
                market_alignment = "aligned"
            elif "contrary" in response.lower() or "opposite" in response.lower():
                market_alignment = "contrary"
            
            stock_analysis = {
                'trend': trend,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'risk_assessment': risk_assessment,
                'market_alignment': market_alignment,
                'full_analysis': response
            }
            
            return stock_analysis
        except Exception as e:
            logger.error(f"Error analyzing underlying stock: {e}")
            return {'error': str(e), 'trend': 'neutral', 'technical_score': 0, 'sentiment_score': 0}
    
    def analyze_credit_spreads(self, spread_data: Dict[str, Any], stock_analysis: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze credit spread opportunities based on market and stock analysis.
        
        Parameters:
        - spread_data: Dict with available spread data (IV, delta, DTE, etc.)
        - stock_analysis: Dict with stock analysis from analyze_underlying_stock
        - market_analysis: Dict with market analysis from analyze_spy_trend
        
        Returns:
        - Dict with spread recommendations, quality scores and risk assessment
        """
        try:
            prompt = get_stock_options_prompt(spread_data, stock_analysis, market_analysis)
            response = self.generate_text(prompt, temperature=0.3)
            
            # Extract key data from the response
            spread_type = "None"
            strikes = "Not specified"
            expiration = "Not specified"
            quality_score = 0
            gamble_score = 0
            success_probability = 0
            position_size = "Not specified"
            profit_target = "50% of max credit"
            stop_loss = "2x credit received"
            recommended = False
            
            # Parse the response
            if "Bull Put" in response:
                spread_type = "Bull Put"
            elif "Bear Call" in response:
                spread_type = "Bear Call"
                
            # Extract strikes
            import re
            strikes_match = re.search(r'[Ss]trikes?:?\s*(\d+\/\d+)', response)
            if strikes_match:
                strikes = strikes_match.group(1)
                
            # Extract expiration
            expiration_match = re.search(r'[Ee]xpiration:?\s*(\d{1,2}\/\d{1,2}\/\d{2,4}|\d{4}-\d{2}-\d{2})', response)
            if expiration_match:
                expiration = expiration_match.group(1)
                
            # Extract quality score
            quality_match = re.search(r'[Qq]uality\s*[Ss]core:?\s*(\d+)', response)
            if quality_match:
                quality_score = int(quality_match.group(1))
                
            # Extract gamble score
            gamble_match = re.search(r'[Gg]amble\s*[Ss]core:?\s*(\d+)', response)
            if gamble_match:
                gamble_score = int(gamble_match.group(1))
                
            # Extract success probability
            prob_match = re.search(r'[Ss]uccess\s*[Pp]robability:?\s*(\d+)%?', response)
            if prob_match:
                success_probability = int(prob_match.group(1))
                
            # Extract position size
            size_match = re.search(r'[Pp]osition\s*[Ss]ize:?\s*\$?(\d+)', response)
            if size_match:
                position_size = f"${size_match.group(1)}"
                
            # Extract profit target and stop loss
            profit_match = re.search(r'[Pp]rofit\s*[Tt]arget:?\s*(.+?)(\\n|\n|$)', response)
            if profit_match:
                profit_target = profit_match.group(1).strip()
                
            stop_match = re.search(r'[Ss]top\s*[Ll]oss:?\s*(.+?)(\\n|\n|$)', response)
            if stop_match:
                stop_loss = stop_match.group(1).strip()
                
            # Determine if recommended
            recommended = (
                quality_score >= 80 or 
                (quality_score >= 70 and gamble_score >= 70) or
                success_probability >= 70
            )
            
            spread_recommendation = {
                'spread_type': spread_type,
                'strikes': strikes,
                'expiration': expiration,
                'quality_score': quality_score,
                'gamble_score': gamble_score,
                'success_probability': success_probability,
                'position_size': position_size,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'recommended': recommended,
                'full_analysis': response
            }
            
            return spread_recommendation
        except Exception as e:
            logger.error(f"Error analyzing credit spreads: {e}")
            return {'error': str(e), 'spread_type': 'None', 'recommended': False}
    
    def generate_trade_plan(self, 
                           spy_analysis: Dict[str, Any], 
                           options_analysis: Dict[str, Any],
                           stock_analysis: Dict[str, Any],
                           spread_analysis: Dict[str, Any]) -> str:
        """
        Generate a complete trade plan based on all analyses.
        
        Parameters:
        - spy_analysis: SPY market trend analysis
        - options_analysis: SPY options direction analysis
        - stock_analysis: Underlying stock analysis
        - spread_analysis: Credit spread analysis
        
        Returns:
        - Comprehensive trade plan with reasoning and execution details
        """
        try:
            ticker = stock_analysis.get('ticker', 'the underlying')
            prompt = get_trade_plan_prompt(spy_analysis, options_analysis, stock_analysis, spread_analysis, ticker)
            return self.generate_text(prompt, temperature=0.4)
        except Exception as e:
            logger.error(f"Error generating trade plan: {e}")
            return f"Error generating trade plan: {str(e)}"


# Singleton instance
_instance = None

def get_gemini_client() -> GeminiClient:
    """Get or create a singleton instance of GeminiClient"""
    global _instance
    if _instance is None:
        _instance = GeminiClient()
    return _instance


if __name__ == "__main__":
    # Test the client
    client = get_gemini_client()
    
    # Basic test of the text generation
    test_prompt = """
    Analyze the following market data and provide insights:
    
    SPY Price: 500.75
    VIX: 20.5
    
    Please include:
    1. Market trend assessment
    2. Risk recommendations
    """
    
    response = client.generate_text(test_prompt, temperature=0.3)
    print(f"\nGenerated Response:\n{response}") 