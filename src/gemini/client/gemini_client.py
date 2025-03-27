#!/usr/bin/env python3
"""
Gemini AI Client Module (src/gemini/client/gemini_client.py)
-----------------------------------------------------------
Client for interacting with Google's Gemini language model API for market analysis.

Class:
  - GeminiClient - Wrapper around Gemini API with specialized analysis methods
  
Methods:
  - generate_text - Core method for generating AI text responses
  - analyze_market_data - Analyzes market data with Gemini
  - analyze_spy_trend - Analyzes SPY market trend
  - analyze_underlying_stock - Analyzes individual stock data
  - analyze_credit_spreads - Evaluates credit spread opportunities
  - plus various utility and fallback methods

Dependencies:
  - google.generativeai - For Gemini API access
  - python-dotenv - For loading API key from .env file
  - src.gemini.hooks - For prompt generation functions
  - Environment variable: GEMINI_API_KEY

Used by:
  - main.py for all AI analysis
  - Various hook functions for specific analytical tasks
"""

import os
import logging
import time
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
import re
import sys


# Add proper imports using relative imports
from src.gemini.hooks import (
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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gemini_client')


class GeminiClient:
    def __init__(self):
        """Initialize the Gemini client with API key."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not found")

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemma-3-27b-it')

            # Compile regex patterns once at initialization
            self._patterns = {
                'score': re.compile(r'(?:score|Score):\s*(\d+)'),
                'tech_score': re.compile(r'(?:technical score|Technical Score):\s*(\d+)'),
                'sentiment_score': re.compile(r'(?:sentiment score|Sentiment Score):\s*(\d+)'),
                'quality_score': re.compile(r'(?:quality score|Quality Score):\s*(\d+)'),
                'probability': re.compile(r'(?:probability|Probability|success probability|Success Probability):\s*(\d+)%?'),
                'position_size': re.compile(r'(?:position size|Position Size):\s*\$?(\d+)'),
                'profit_target': re.compile(r'(?:profit target|Profit Target|target):\s*(\d+)%'),
                'stop_loss': re.compile(r'(?:stop loss|Stop Loss):\s*(\d+)%'),
                'market_alignment': re.compile(r'Market Alignment:\s*(.*?)(?:\n\d\.|\n\n|$)', re.DOTALL),
                'options_section': re.compile(r'(?:Options chain analysis|Options-based directional prediction):\s*(.*?)(?:\n\d\.|\n\n|$)', re.DOTALL),
                'ticker': re.compile(r'[A-Z]{1,5}')
            }

            # Define response templates for more consistent output
            self.response_templates = {
                'market_trend': """
Please respond in this exact JSON format:
{
  "trend": "bullish|bearish|neutral",
  "market_trend_score": <0-30>,
  "vix_assessment": "<your assessment>",
  "risk_adjustment": "standard|half size|skip",
  "sector_rotation": "<your assessment>",
  "explanation": "<detailed analysis>"
}
""",
                'stock_analysis': """
Please respond in this exact JSON format:
{
  "trend": "bullish|bearish|neutral",
  "technical_score": <0-100>,
  "sentiment_score": <0-100>, /* Now represents price movement predictability */
  "risk_assessment": "low|moderate|high",
  "market_alignment": "aligned|contrary|neutral",
  "options_analysis": "<your assessment>",
  "explanation": "<detailed analysis>"
}
""",
                'credit_spread': """
Please respond in this exact JSON format:
{
  "spread_type": "Bull Put|Bear Call|Iron Condor",
  "strikes": "<strike prices>",
  "quality_score": <0-100>,
  "success_probability": <0-100>,
  "position_size": "$<amount>",
  "profit_target": "<percentage or amount>",
  "stop_loss": "<percentage or amount>",
  "greek_assessment": "<your assessment>",
  "recommended": true|false,
  "explanation": "<detailed analysis>"
}
"""
            }

            logger.info("Gemini client initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise

    def generate_text(self, prompt: str, temperature: float = 0.7, structured: bool = False,
                      response_format: str = None) -> str:
        """
        Generate text using the Gemini model.

        Args:
            prompt: The text prompt to send to the model
            temperature: Controls randomness (0.0-1.0)
            structured: Whether to request structured JSON output
            response_format: A specific format template to use ('market_trend', 'stock_analysis', 'credit_spread')

        Returns:
            The generated text response
        """
        max_retries = 5  # Increased from 3 to 5
        retry_delay = 10  # Fixed 10-second delay for 429 errors

        # Add structured format instruction if requested
        if structured:
            if response_format and response_format in self.response_templates:
                # Use a specific template
                format_instruction = self.response_templates[response_format]
            else:
                # Generic JSON instruction
                format_instruction = "Please respond in valid JSON format."

            prompt = f"{prompt}\n\n{format_instruction}"
            logger.debug(
                f"Using structured format: {response_format or 'generic JSON'}")

        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                result = response.text

                # Try to extract JSON if the response contains it
                if structured:
                    result = self._extract_json_from_response(result)

                return result

            except Exception as e:
                error_str = str(e)
                logger.error(f"Error generating text: {e}")

                # Check if it's a rate limit error (429)
                if "429" in error_str and attempt < max_retries:
                    logger.warning(
                        f"Rate limit exceeded (429). Waiting {retry_delay} seconds before retry #{attempt+1}")
                    time.sleep(retry_delay)
                    continue

                # For other errors or if we've exhausted retries, return fallback
                return self._generate_fallback_response(prompt, structured, response_format)

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from a response that might contain markdown and other text."""
        try:
            # Try to find JSON within triple backticks
            json_match = re.search(
                r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
                # Validate it's proper JSON
                json.loads(json_str)
                return json_str

            # Try to find JSON with curly braces
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                json_str = json_match.group(1).strip()
                # Validate it's proper JSON
                json.loads(json_str)
                return json_str

            logger.warning("Could not extract JSON from response")
            return response

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            return response

    def _generate_fallback_response(self, prompt: str, structured: bool = False,
                                    response_format: str = None) -> str:
        """Generate a simple fallback response when API fails."""
        logger.info("Generating fallback response")

        # Detect what kind of analysis we're doing based on the prompt content
        analysis_type = self._detect_analysis_type(prompt)

        # Get fallback based on analysis type
        fallback = self._get_fallback_for_type(analysis_type, prompt)

        # Convert to JSON if structured was requested
        if structured:
            try:
                if isinstance(fallback, dict):
                    return json.dumps(fallback)
                else:
                    # Try to parse the fallback text as JSON
                    key_values = {}
                    for line in fallback.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key_values[key.strip()] = value.strip()

                    return json.dumps(key_values)
            except Exception as e:
                logger.error(f"Error converting fallback to JSON: {e}")

        return fallback

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
        # Common stock tickers are 1-5 uppercase letters
        ticker_matches = re.findall(r'[A-Z]{1,5}', prompt)

        # Filter out common non-ticker uppercase words
        non_tickers = {'SPY', 'VIX', 'EMA', 'ATR', 'IV', 'OTM', 'DTE', 'HTTP'}
        potential_tickers = [
            t for t in ticker_matches if t not in non_tickers and len(t) >= 2]

        # Look for ticker mentioned in context clues
        for line in prompt.split('\n'):
            if 'ticker' in line.lower() and ':' in line:
                ticker_part = line.split(':', 1)[1].strip()
                if ticker_part and ticker_part.upper() == ticker_part:
                    return ticker_part

        return potential_tickers[0] if potential_tickers else None

    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using the Gemini model."""
        try:
            prompt = get_market_data_prompt(market_data)
            response_text = self.generate_text(
                prompt, temperature=0.3, structured=True)

            # Try to parse the response as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fall back to text parsing if JSON parsing fails
                logger.warning(
                    "Failed to parse market data response as JSON, falling back to text parsing")
                parsed_result = self._parse_market_analysis_from_text(
                    response_text)
                return parsed_result

        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            return {'error': str(e), 'trend': 'neutral', 'market_trend_score': 0, 'message': 'Market trend score can be 0-30, with higher values indicating stronger trends.'}

    def _parse_market_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Parse market analysis from text response using regex patterns."""
        result = {
            'trend': 'neutral',
            'market_trend_score': 0,
            'vix_assessment': '',
            'risk_adjustment': 'standard',
            'full_analysis': text
        }

        # Extract trend
        if "bullish" in text.lower():
            result['trend'] = "bullish"
        elif "bearish" in text.lower():
            result['trend'] = "bearish"

        # Try multiple patterns to extract the market trend score
        # First check for exact "Market Trend Score" format
        score_match = re.search(
            r'Market\s+Trend\s+Score:\s*(\d+)', text, re.IGNORECASE)
        if score_match:
            result['market_trend_score'] = int(score_match.group(1))
        else:
            # Try the compiled pattern which looks for score/Score
            score_match = self._patterns['score'].search(text)
            if score_match:
                result['market_trend_score'] = int(score_match.group(1))
            else:
                # Try another variation with "trend score"
                score_match = re.search(
                    r'trend\s+score:\s*(\d+)', text, re.IGNORECASE)
                if score_match:
                    result['market_trend_score'] = int(score_match.group(1))
                else:
                    # Calculate a fallback score based on market cues in the text
                    market_score = 0

                    # Add points for bullish/bearish indicators
                    if result['trend'] == "bullish":
                        market_score += 15  # Base score for bullish trend

                        # Check for indicators of strength
                        if any(term in text.lower() for term in ["strong bull", "strongly bullish", "very bullish"]):
                            market_score += 10
                        elif any(term in text.lower() for term in ["moderate bull", "moderately bullish"]):
                            market_score += 5

                    elif result['trend'] == "bearish":
                        market_score += 15  # Base score for bearish trend

                        # Check for indicators of strength
                        if any(term in text.lower() for term in ["strong bear", "strongly bearish", "very bearish"]):
                            market_score += 10
                        elif any(term in text.lower() for term in ["moderate bear", "moderately bearish"]):
                            market_score += 5

                    # Add points for EMA analysis
                    if "Price > 9/21 EMA" in text and result['trend'] == "bullish":
                        market_score += 5
                    if "Price < 9/21 EMA" in text and result['trend'] == "bearish":
                        market_score += 5

                    # Add points for VIX assessment
                    if "VIX < 15" in text or "low volatility" in text.lower():
                        market_score += 3
                    elif "VIX < 20" in text:
                        market_score += 2
                    elif "VIX > 30" in text or "high volatility" in text.lower():
                        market_score -= 3

                    # Set the calculated score
                    result['market_trend_score'] = market_score

        # Extract VIX assessment if present
        vix_lines = [line for line in text.split("\n") if "VIX" in line]
        if vix_lines:
            result['vix_assessment'] = vix_lines[0].strip()

        # Extract risk adjustment if present
        if "half" in text.lower() or "reduce" in text.lower():
            result['risk_adjustment'] = "half size"
        elif "skip" in text.lower() or "avoid" in text.lower():
            result['risk_adjustment'] = "skip"

        return result

    def analyze_spy_trend(self, spy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SPY market trend based on EMA and VIX data.

        Parameters:
        - spy_data: Dict containing SPY price, EMA data and VIX values

        Returns:
        - Dict with market trend analysis, score and recommendations
        """
        try:
            start_time = time.time()
            prompt = get_market_trend_prompt(spy_data)

            # Request structured JSON output
            response_text = self.generate_text(
                prompt, temperature=0.2, structured=True, response_format='market_trend'
            )

            # Try to parse the response as JSON
            try:
                market_analysis = json.loads(response_text)
                logger.debug("Successfully parsed JSON response for SPY trend")

                # Ensure the response has full_analysis for compatibility
                if 'explanation' in market_analysis and 'full_analysis' not in market_analysis:
                    market_analysis['full_analysis'] = market_analysis['explanation']

                return market_analysis

            except json.JSONDecodeError:
                # Fall back to regex parsing if JSON fails
                logger.warning(
                    "Failed to parse market trend response as JSON, falling back to regex parsing")

                # Use the existing parsing logic with compiled patterns
                trend = "neutral"
                if "bullish" in response_text.lower():
                    trend = "bullish"
                elif "bearish" in response_text.lower():
                    trend = "bearish"

                # Extract trend score using regex
                score_match = self._patterns['score'].search(response_text)
                market_trend_score = 0
                if score_match:
                    market_trend_score = int(score_match.group(1))
                else:
                    # Fallback logic to compute score
                    if "Price > 9/21 EMA" in response_text and trend == "bullish":
                        market_trend_score += 10
                    if "Price < 9/21 EMA" in response_text and trend == "bearish":
                        market_trend_score += 10

                    # VIX score component
                    if "VIX < 20" in response_text:
                        market_trend_score += 5

                # Extract VIX assessment if present
                vix_lines = [line for line in response_text.split(
                    "\n") if "VIX" in line]
                vix_assessment = ""
                if vix_lines:
                    vix_assessment = vix_lines[0].strip()

                # Extract risk adjustment if present
                risk_adjustment = "standard"
                if "half" in response_text.lower() or "reduce" in response_text.lower():
                    risk_adjustment = "half size"
                elif "skip" in response_text.lower() or "avoid" in response_text.lower():
                    risk_adjustment = "skip"

                market_analysis = {
                    'trend': trend,
                    'market_trend_score': market_trend_score,
                    'vix_assessment': vix_assessment,
                    'risk_adjustment': risk_adjustment,
                    'full_analysis': response_text
                }

                return market_analysis

        except Exception as e:
            logger.error(f"Error analyzing SPY trend: {e}", exc_info=True)
            return {'error': str(e), 'trend': 'neutral', 'market_trend_score': 0, 'message': 'Market trend score can be 0-30, with higher values indicating stronger trends.'}

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
        Analyze individual stock for trading opportunities based on technical and price movement predictability.

        Parameters:
        - stock_data: Dict containing stock price, EMA data and volume
        - market_context: Dict with market trend analysis

        Returns:
        - Dict with stock analysis, scores and recommendations.
          Note: The sentiment_score now represents price movement predictability
          rather than news sentiment.
        """
        try:
            start_time = time.time()
            prompt = get_stock_analysis_prompt(stock_data, market_context)

            # Request structured JSON output
            response_text = self.generate_text(
                prompt, temperature=0.3, structured=True, response_format='stock_analysis'
            )

            # Try to parse the response as JSON
            try:
                stock_analysis = json.loads(response_text)

                # Add ticker from input data
                stock_analysis['ticker'] = stock_data.get('ticker', 'Unknown')

                # Calculate overall score if not provided
                if 'overall_score' not in stock_analysis and 'technical_score' in stock_analysis and 'sentiment_score' in stock_analysis:
                    stock_analysis['overall_score'] = stock_analysis['technical_score'] + \
                        stock_analysis['sentiment_score']

                # Ensure the response has analysis field for compatibility
                if 'explanation' in stock_analysis and 'analysis' not in stock_analysis:
                    stock_analysis['analysis'] = stock_analysis['explanation']

                return stock_analysis

            except json.JSONDecodeError:
                # Fall back to regex parsing if JSON fails
                logger.warning(
                    "Failed to parse stock analysis response as JSON, falling back to regex parsing")

                # Extract structured data from the response using compiled patterns
                trend = "neutral"
                if "bullish" in response_text.lower():
                    trend = "bullish"
                elif "bearish" in response_text.lower():
                    trend = "bearish"

                # Extract technical score
                tech_score_match = self._patterns['tech_score'].search(
                    response_text)
                technical_score = 0
                if tech_score_match:
                    technical_score = int(tech_score_match.group(1))

                # Extract sentiment score
                sent_score_match = self._patterns['sentiment_score'].search(
                    response_text)
                sentiment_score = 0
                if sent_score_match:
                    sentiment_score = int(sent_score_match.group(1))

                # Extract risk assessment
                risk_assessment = "moderate"
                if "high risk" in response_text.lower():
                    risk_assessment = "high"
                elif "low risk" in response_text.lower():
                    risk_assessment = "low"

                # Extract market alignment
                market_alignment = "aligned"
                if "misaligned" in response_text.lower() or "not aligned" in response_text.lower() or "contrary" in response_text.lower():
                    market_alignment = "contrary"

                # Calculate overall score
                overall_score = technical_score + sentiment_score

                stock_analysis = {
                    'ticker': stock_data.get('ticker', 'Unknown'),
                    'trend': trend,
                    'technical_score': technical_score,
                    'sentiment_score': sentiment_score,
                    'overall_score': overall_score,
                    'risk_assessment': risk_assessment,
                    'market_alignment': market_alignment,
                    'analysis': response_text
                }

                return stock_analysis

        except Exception as e:
            logger.error(
                f"Error analyzing stock data for {stock_data.get('ticker', 'Unknown')}: {e}", exc_info=True)
            return {
                'ticker': stock_data.get('ticker', 'Unknown'),
                'trend': 'neutral',
                'technical_score': 0,
                'sentiment_score': 0,
                'overall_score': 0,
                'risk_assessment': 'high',
                'market_alignment': 'misaligned',
                'analysis': f"Error analyzing stock: {str(e)}"
            }

    def analyze_credit_spreads(self, spread_data: Dict[str, Any], stock_analysis: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze credit spread opportunities based on options data.

        Parameters:
        - spread_data: Dict containing options, stock price and IV data
        - stock_analysis: Dict with stock trend and analysis
        - market_analysis: Dict with market trend and analysis

        Returns:
        - Dict with spread recommendations, quality score and risk assessment
        """
        try:
            ticker = spread_data.get('ticker', 'Unknown')
            prompt = get_stock_options_prompt(
                spread_data, stock_analysis, market_analysis)

            # Request structured JSON output using the credit_spread template
            response_text = self.generate_text(
                prompt, temperature=0.2, structured=True, response_format='credit_spread'
            )

            # Try to parse the response as JSON
            try:
                spread_analysis = json.loads(response_text)

                # Add ticker from input data
                spread_analysis['ticker'] = ticker

                # Ensure the response has analysis field for compatibility
                if 'explanation' in spread_analysis and 'analysis' not in spread_analysis:
                    spread_analysis['analysis'] = spread_analysis['explanation']

                return spread_analysis

            except json.JSONDecodeError as json_err:
                # Log the specific JSON error and response for debugging
                logger.warning(f"JSON decode error for {ticker}: {json_err}")

                # Fall back to regex parsing if JSON fails
                logger.warning(
                    f"Failed to parse credit spread response as JSON for {ticker}, falling back to regex parsing")

                # Extract structured data from the response using compiled patterns
                spread_type = "Bull Put"
                if "bear call" in response_text.lower():
                    spread_type = "Bear Call"
                elif "iron condor" in response_text.lower():
                    spread_type = "Iron Condor"

                # Extract quality score
                quality_score = 0
                quality_match = self._patterns['quality_score'].search(
                    response_text)
                if quality_match:
                    quality_score = int(quality_match.group(1))

                # Extract success probability
                success_prob = 0
                prob_match = self._patterns['probability'].search(
                    response_text)
                if prob_match:
                    success_prob = int(prob_match.group(1))

                # Extract position size recommendation
                position_size = "skip"
                size_match = self._patterns['position_size'].search(
                    response_text)
                if size_match:
                    position_size = f"${size_match.group(1)}"
                elif "$200" in response_text:
                    position_size = "$200"
                elif "$100" in response_text:
                    position_size = "$100"
                elif "$300" in response_text:
                    position_size = "$300"
                elif "$400" in response_text:
                    position_size = "$400"
                elif "$500" in response_text:
                    position_size = "$500"

                # Determine if recommended
                recommended = False
                if "recommended: yes" in response_text.lower() or "recommendation: yes" in response_text.lower():
                    recommended = True
                elif quality_score >= 70:  # Default threshold
                    recommended = True

                # Extract profit target if present
                profit_target = "50%"
                profit_match = self._patterns['profit_target'].search(
                    response_text)
                if profit_match:
                    profit_target = f"{profit_match.group(1)}%"

                # Extract stop loss if present
                stop_loss = "100%"
                stop_match = self._patterns['stop_loss'].search(response_text)
                if stop_match:
                    stop_loss = f"{stop_match.group(1)}%"

                # Extract strikes
                strikes = "Not specified"
                strikes_match = re.search(
                    r'[Ss]trikes?:?\s*(\d+\/\d+|\d+[\s-]+\d+)', response_text)
                if strikes_match:
                    strikes = strikes_match.group(1).replace(
                        " ", "/").replace("-", "/")

                spread_analysis = {
                    'ticker': ticker,
                    'spread_type': spread_type,
                    'strikes': strikes,
                    'quality_score': quality_score,
                    'success_probability': success_prob,
                    'position_size': position_size,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'recommended': recommended,
                    'analysis': response_text
                }

                return spread_analysis

        except Exception as e:
            logger.error(
                f"Error analyzing credit spreads for {spread_data.get('ticker', 'Unknown')}: {e}", exc_info=True)
            # Return a detailed error response
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'phase': 'Unknown'  # Default phase
            }

            # Try to determine which phase of processing caused the error
            if 'prompt' in locals():
                if 'response_text' not in locals():
                    error_details['phase'] = 'API_CALL'
                elif 'json_err' in locals():
                    error_details['phase'] = 'JSON_PARSING'
                else:
                    error_details['phase'] = 'TEXT_PARSING'
            else:
                error_details['phase'] = 'PROMPT_GENERATION'

            return {
                'ticker': spread_data.get('ticker', 'Unknown'),
                'error': True,
                'error_details': error_details,
                'spread_type': 'None',
                'recommended': False
            }

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
            prompt = get_trade_plan_prompt(
                spy_analysis, options_analysis, stock_analysis, spread_analysis, ticker)

            # We don't use structured output for trade plans as they're meant to be human-readable
            return self.generate_text(prompt, temperature=0.4)

        except Exception as e:
            logger.error(
                f"Error generating trade plan for {stock_analysis.get('ticker', 'Unknown')}: {e}", exc_info=True)
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

    # Test structured response
    test_prompt = """
    Analyze the following market data and provide insights:
    
    SPY Price: 500.75
    VIX: 20.5
    
    Please include:
    1. Market trend assessment
    2. Risk recommendations
    """

    response = client.generate_text(
        test_prompt, temperature=0.3, structured=True)
    print(f"\nGenerated Structured Response:\n{response}")

    # Try to parse as JSON to validate
    try:
        parsed = json.loads(response)
        print(
            f"\nSuccessfully parsed as JSON:\n{json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError:
        print("\nResponse could not be parsed as JSON.")
