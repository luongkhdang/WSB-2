import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_stock_analysis(analysis_text):
    """
    Parse stock analysis text from AI into structured data
    
    Parameters:
    - analysis_text: Full analysis text
    
    Returns:
    - Dictionary with parsed analysis
    """
    logger.debug("Parsing stock analysis text")
    
    # Initialize default values
    analysis = {
        "trend": "neutral",
        "technical_score": 0,
        "fundamental_score": 0,
        "predictability_score": 0,
        "sentiment_score": 0,
        "total_score": 0,
        "market_alignment": "neutral",
        "strength": "neutral",
        "support_levels": [],
        "resistance_levels": [],
        "volume_analysis": "",
        "price_targets": {},
        "risk_factors": [],
        "opportunity_factors": [],
        "pattern_recognition": "",
        "pattern_type": ""
    }
    
    try:
        # Extract trend
        if "bullish" in analysis_text.lower():
            analysis["trend"] = "bullish"
        elif "bearish" in analysis_text.lower():
            analysis["trend"] = "bearish"
        
        # Extract technical score
        tech_score_match = re.search(r'(?:Technical|technical|TECHNICAL)(?:\s+analysis)?(?:\s+score)?:?\s*(\d+)', analysis_text)
        if tech_score_match:
            analysis["technical_score"] = int(tech_score_match.group(1))
        
        # Extract fundamental score
        fund_score_match = re.search(r'(?:Fundamental|fundamental|FUNDAMENTAL)(?:\s+analysis)?(?:\s+score)?:?\s*(\d+)', analysis_text)
        if fund_score_match:
            analysis["fundamental_score"] = int(fund_score_match.group(1))
        
        # Extract predictability score or sentiment score (try both labels)
        predict_score_match = re.search(r'(?:Predictability|predictability|PREDICTABILITY)(?:\s+analysis)?(?:\s+score)?:?\s*(\d+)', analysis_text)
        if predict_score_match:
            analysis["predictability_score"] = int(predict_score_match.group(1))
            # For backward compatibility, also store this as sentiment score
            analysis["sentiment_score"] = analysis["predictability_score"]
        else:
            # If no predictability score, look for sentiment score
            sent_score_match = re.search(r'(?:Sentiment|sentiment|SENTIMENT)(?:\s+analysis)?(?:\s+score)?:?\s*(\d+)', analysis_text)
            if sent_score_match:
                analysis["sentiment_score"] = int(sent_score_match.group(1))
                # Also store as predictability score for forward compatibility
                analysis["predictability_score"] = analysis["sentiment_score"]
        
        # Calculate total score
        analysis["total_score"] = (
            analysis["technical_score"] + 
            analysis["fundamental_score"] + 
            max(analysis["predictability_score"], analysis["sentiment_score"])
        )
        
        # Extract market alignment
        alignment_match = re.search(r'(?:Market\s+Alignment|market\s+alignment):\s*(\w+)', analysis_text, re.IGNORECASE)
        if alignment_match:
            alignment = alignment_match.group(1).lower()
            if alignment in ["aligned", "following", "leading", "divergent", "contrary"]:
                analysis["market_alignment"] = alignment
        else:
            if "aligned with market" in analysis_text.lower() or "following market" in analysis_text.lower():
                analysis["market_alignment"] = "aligned"
            elif "diverging from market" in analysis_text.lower() or "divergent from market" in analysis_text.lower():
                analysis["market_alignment"] = "divergent"
            elif "leading market" in analysis_text.lower():
                analysis["market_alignment"] = "leading"
        
        # Extract strength
        if "strong" in analysis_text.lower() and analysis["trend"] == "bullish":
            analysis["strength"] = "strong bullish"
        elif "weak" in analysis_text.lower() and analysis["trend"] == "bullish":
            analysis["strength"] = "weak bullish"
        elif "strong" in analysis_text.lower() and analysis["trend"] == "bearish":
            analysis["strength"] = "strong bearish"
        elif "weak" in analysis_text.lower() and analysis["trend"] == "bearish":
            analysis["strength"] = "weak bearish"
        
        # Extract support levels
        support_match = re.search(r'Support(?:\s+levels)?:?\s*([^:]+)(?:\n|$)', analysis_text, re.IGNORECASE)
        if support_match:
            support_text = support_match.group(1).strip()
            support_values = re.findall(r'\$?(\d+\.?\d*)', support_text)
            analysis["support_levels"] = [float(val) for val in support_values]
        
        # Extract resistance levels
        resistance_match = re.search(r'Resistance(?:\s+levels)?:?\s*([^:]+)(?:\n|$)', analysis_text, re.IGNORECASE)
        if resistance_match:
            resistance_text = resistance_match.group(1).strip()
            resistance_values = re.findall(r'\$?(\d+\.?\d*)', resistance_text)
            analysis["resistance_levels"] = [float(val) for val in resistance_values]
        
        # Extract pattern recognition
        pattern_section = ""
        pattern_match = re.search(r'PATTERN RECOGNITION:\s*(.*?)(?:\n\n|\n[A-Z]|$)', analysis_text, re.IGNORECASE | re.DOTALL)
        if pattern_match:
            pattern_section = pattern_match.group(1).strip()
            analysis["pattern_recognition"] = pattern_section
            
            # Identify specific pattern types
            pattern_types = ["triangle", "flag", "wedge", "channel", "head and shoulders", 
                            "double top", "double bottom", "cup and handle", "rectangle",
                            "fibonacci", "support", "resistance", "consolidation"]
            
            for pattern in pattern_types:
                if pattern.lower() in pattern_section.lower():
                    if analysis["pattern_type"]:
                        analysis["pattern_type"] += f", {pattern}"
                    else:
                        analysis["pattern_type"] = pattern
        
        # Extract volume analysis
        volume_match = re.search(r'Volume(?:\s+analysis)?:?\s*([^:]+)(?:\n|$)', analysis_text, re.IGNORECASE)
        if volume_match:
            analysis["volume_analysis"] = volume_match.group(1).strip()
        
        # Extract price targets
        short_target_match = re.search(r'(?:Short[- ]term|1[- ]day)(?:\s+target)?:?\s*\$?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
        if short_target_match:
            analysis["price_targets"]["short_term"] = float(short_target_match.group(1))
        
        medium_target_match = re.search(r'(?:Medium[- ]term|1[- ]week)(?:\s+target)?:?\s*\$?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
        if medium_target_match:
            analysis["price_targets"]["medium_term"] = float(medium_target_match.group(1))
        
        long_target_match = re.search(r'(?:Long[- ]term|1[- ]month)(?:\s+target)?:?\s*\$?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
        if long_target_match:
            analysis["price_targets"]["long_term"] = float(long_target_match.group(1))
        
        # Extract risk factors
        risk_section = ""
        risk_keywords = ["risk", "risks", "risk factors", "concerns", "warning signs"]
        for keyword in risk_keywords:
            match = re.search(fr'{keyword}:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', analysis_text, re.IGNORECASE | re.DOTALL)
            if match:
                risk_section = match.group(1).strip()
                break
        
        if risk_section:
            # Split by bullet points or new lines
            risks = re.split(r'•|\*|\n-|\n\d+\.', risk_section)
            analysis["risk_factors"] = [risk.strip() for risk in risks if risk.strip()]
        
        # Extract opportunity factors
        opp_section = ""
        opp_keywords = ["opportunity", "opportunities", "opportunity factors", "positive factors", "strengths"]
        for keyword in opp_keywords:
            match = re.search(fr'{keyword}:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', analysis_text, re.IGNORECASE | re.DOTALL)
            if match:
                opp_section = match.group(1).strip()
                break
        
        if opp_section:
            # Split by bullet points or new lines
            opportunities = re.split(r'•|\*|\n-|\n\d+\.', opp_section)
            analysis["opportunity_factors"] = [opp.strip() for opp in opportunities if opp.strip()]
        
        # Add the full analysis text for reference
        analysis["full_analysis"] = analysis_text
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error parsing stock analysis: {e}")
        return analysis

def parse_reflection(reflection_text):
    """
    Parse reflection text from AI into structured data
    
    Parameters:
    - reflection_text: Full reflection text
    
    Returns:
    - Dictionary with parsed reflection
    """
    logger.debug("Parsing reflection text")
    
    # Initialize default values
    reflection = {
        "prediction_accuracy": 0,
        "correct_direction": False,
        "magnitude_error": 0,
        "lessons_learned": [],
        "adjustment_factor": 0
    }
    
    try:
        # Extract prediction accuracy
        accuracy_match = re.search(r'accuracy(?:\s+score)?:?\s*(\d+)', reflection_text, re.IGNORECASE)
        if accuracy_match:
            reflection["prediction_accuracy"] = int(accuracy_match.group(1))
        
        # Extract correct direction
        if "correct direction" in reflection_text.lower() or "correctly predicted" in reflection_text.lower():
            reflection["correct_direction"] = True
        
        # Extract magnitude error
        error_match = re.search(r'(?:magnitude|price)(?:\s+error)?:?\s*(\d+\.?\d*)%?', reflection_text, re.IGNORECASE)
        if error_match:
            reflection["magnitude_error"] = float(error_match.group(1))
        
        # Extract lessons learned
        lessons_section = ""
        lessons_keywords = ["lessons", "lessons learned", "insights", "takeaways", "learnings"]
        for keyword in lessons_keywords:
            match = re.search(fr'{keyword}:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', reflection_text, re.IGNORECASE | re.DOTALL)
            if match:
                lessons_section = match.group(1).strip()
                break
        
        if lessons_section:
            # Split by bullet points or new lines
            lessons = re.split(r'•|\*|\n-|\n\d+\.', lessons_section)
            reflection["lessons_learned"] = [lesson.strip() for lesson in lessons if lesson.strip()]
        
        # Extract adjustment factor
        adjustment_match = re.search(r'adjustment(?:\s+factor)?:?\s*([+-]?\d+\.?\d*)', reflection_text, re.IGNORECASE)
        if adjustment_match:
            reflection["adjustment_factor"] = float(adjustment_match.group(1))
        
        return reflection
    
    except Exception as e:
        logger.error(f"Error parsing reflection: {e}")
        return reflection

def parse_market_analysis(market_analysis_text):
    """
    Parse market analysis text from AI into structured data
    
    Parameters:
    - market_analysis_text: Full market analysis text
    
    Returns:
    - Dictionary with parsed market analysis
    """
    logger.debug("Parsing market analysis text")
    
    # Initialize market trend
    trend = "neutral"
    market_trend_score = 0
    vix_assessment = ""
    risk_adjustment = "standard"
    sector_rotation_analysis = ""
    
    try:
        # Extract trend
        if "bullish" in market_analysis_text.lower():
            trend = "bullish"
        elif "bearish" in market_analysis_text.lower():
            trend = "bearish"
            
        # Extract trend score using regex - more flexible pattern matching
        # Try multiple pattern variations to increase chance of finding the score
        score_patterns = [
            r'(?:Market\s+Trend\s+Score|market\s+trend\s+score):\s*(\d+)',
            r'(?:trend\s+score|Trend\s+Score):\s*(\d+)',
            r'(?:score|Score):\s*(\d+)',
        ]
        
        for pattern in score_patterns:
            score_match = re.search(pattern, market_analysis_text)
            if score_match:
                market_trend_score = int(score_match.group(1))
                break
                
        # If no score found, try to extract a numeric value following "Market Trend:" section
        if market_trend_score == 0:
            market_trend_section = re.search(r'Market\s+Trend:.*?(\d+)', market_analysis_text, re.IGNORECASE | re.DOTALL)
            if market_trend_section:
                market_trend_score = int(market_trend_section.group(1))
            
        # Extract VIX assessment if present
        vix_lines = [line for line in market_analysis_text.split("\n") if "VIX" in line]
        if vix_lines:
            vix_assessment = vix_lines[0].strip()
            
        # Extract risk adjustment if present
        if "half" in market_analysis_text.lower() or "reduce" in market_analysis_text.lower():
            risk_adjustment = "half size"
        elif "skip" in market_analysis_text.lower() or "avoid" in market_analysis_text.lower():
            risk_adjustment = "skip"
            
        # Extract sector rotation analysis if present
        sector_rotation_match = re.search(r'Sector Rotation:[\s\n]*(.*?)(?:[\n][\n]|$)', market_analysis_text, re.DOTALL)
        if sector_rotation_match:
            sector_rotation_analysis = sector_rotation_match.group(1).strip()
        
        # Create structured market analysis
        market_trend = {
            'trend': trend,
            'market_trend_score': market_trend_score,
            'vix_assessment': vix_assessment,
            'risk_adjustment': risk_adjustment,
            'sector_rotation': sector_rotation_analysis,
            'full_analysis': market_analysis_text
        }
        
        return market_trend
    
    except Exception as e:
        logger.error(f"Error parsing market analysis: {e}")
        return {
            'trend': trend,
            'market_trend_score': market_trend_score,
            'vix_assessment': vix_assessment,
            'risk_adjustment': risk_adjustment,
            'sector_rotation': sector_rotation_analysis,
            'full_analysis': market_analysis_text
        } 