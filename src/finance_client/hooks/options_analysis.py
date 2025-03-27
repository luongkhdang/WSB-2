import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta

# Import utilities
from src.finance_client.utilities.options_processor import (
    calculate_option_greeks,
    calculate_iv_skew,
    identify_spread_opportunities,
    enrich_options_chain
)

logger = logging.getLogger(__name__)

def analyze_options_chain(options_chain: Dict, 
                         price_history: Optional[pd.DataFrame] = None,
                         volatility_data: Optional[Dict] = None,
                         strategy_filter: Optional[str] = None) -> Dict:
    """
    Analyze options chain and return insights
    
    Args:
        options_chain: Options chain dictionary returned from get_options_chain
        price_history: Historical price data for the underlying (optional)
        volatility_data: Market volatility data (optional)
        strategy_filter: Filter for specific strategy types (e.g., 'bullish', 'bearish', 'neutral')
        
    Returns:
        Dictionary with options analysis results
    """
    logger.info(f"Analyzing options chain for {options_chain.get('symbol')}")
    
    if not options_chain or 'calls' not in options_chain or 'puts' not in options_chain:
        logger.warning("Invalid options chain data provided")
        return {"error": "Invalid options chain data"}
    
    try:
        # Make a copy to avoid modifying the original
        result = options_chain.copy()
        
        # Enrich with greeks and other calculations
        result = enrich_options_chain(result)
        
        # Get IV skew data and implied market direction
        iv_skew = calculate_iv_skew(result)
        result['iv_skew'] = iv_skew
        
        # Add market bias based on skew
        if iv_skew.get('put_call_skew', 1.0) > 1.1:
            result['market_bias'] = 'bearish'
            result['skew_strength'] = (iv_skew.get('put_call_skew', 1.0) - 1.0) * 100
        elif iv_skew.get('put_call_skew', 1.0) < 0.9:
            result['market_bias'] = 'bullish'
            result['skew_strength'] = (1.0 - iv_skew.get('put_call_skew', 1.0)) * 100
        else:
            result['market_bias'] = 'neutral'
            result['skew_strength'] = 0
        
        # Find trading opportunities
        opportunities = identify_spread_opportunities(result, strategy_filter)
        result['opportunities'] = opportunities
        
        # Analyze the risk profile
        risk_profile = _analyze_risk_profile(result, price_history, volatility_data)
        result['risk_profile'] = risk_profile
        
        # Add overall analysis summary
        result['analysis'] = _generate_options_summary(result)
        
        logger.info(f"Completed options analysis for {options_chain.get('symbol')}")
        return result
    
    except Exception as e:
        logger.error(f"Error during options analysis: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "original_data": options_chain
        }

def _analyze_risk_profile(options_data: Dict, 
                         price_history: Optional[pd.DataFrame] = None, 
                         volatility_data: Optional[Dict] = None) -> Dict:
    """
    Analyze the risk profile of the options chain
    
    Args:
        options_data: Enriched options chain data
        price_history: Historical price data for the underlying
        volatility_data: Market volatility data 
        
    Returns:
        Dictionary with risk analysis
    """
    symbol = options_data.get('symbol')
    current_price = options_data.get('current_price')
    expiration_date = options_data.get('expiration_date')
    
    # Calculate days until expiration
    if expiration_date:
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        days_to_expiration = (exp_date - datetime.now()).days
    else:
        days_to_expiration = 30  # Default assumption
    
    # Default risk profile
    risk_profile = {
        "days_to_expiration": days_to_expiration,
        "implied_move": None,
        "expected_range": {},
        "theta_risk": "low",
        "gamma_risk": "low",
        "vega_risk": "low"
    }
    
    # Get ATM option for implied move calculation
    atm_options = _find_atm_options(options_data, current_price)
    
    if atm_options['call'] is not None and 'iv' in atm_options['call']:
        # Calculate implied move until expiration based on ATM IV
        atm_iv = atm_options['call']['iv']
        implied_move_pct = atm_iv * np.sqrt(days_to_expiration / 365)
        implied_move_price = current_price * implied_move_pct
        
        risk_profile['implied_move'] = {
            'percent': float(implied_move_pct * 100),  # Convert to percentage
            'price': float(implied_move_price),
            'expected_range': {
                'lower': float(current_price - implied_move_price),
                'upper': float(current_price + implied_move_price)
            }
        }
    
    # Assess theta risk based on days to expiration
    if days_to_expiration < 7:
        risk_profile['theta_risk'] = "extreme"
    elif days_to_expiration < 14:
        risk_profile['theta_risk'] = "high"
    elif days_to_expiration < 30:
        risk_profile['theta_risk'] = "moderate"
    else:
        risk_profile['theta_risk'] = "low"
    
    # Assess gamma risk (highest near expiration and ATM)
    if days_to_expiration < 7:
        risk_profile['gamma_risk'] = "high"
    elif days_to_expiration < 21:
        risk_profile['gamma_risk'] = "moderate"
    else:
        risk_profile['gamma_risk'] = "low"
    
    # Assess vega risk based on market volatility
    if volatility_data and 'stability' in volatility_data:
        if volatility_data['stability'] in ['extreme', 'elevated']:
            risk_profile['vega_risk'] = "high"
            risk_profile['volatility_note'] = "Market volatility is elevated, increasing vega risk"
        else:
            risk_profile['vega_risk'] = "moderate"
    
    # Adjust for historical volatility if available
    if price_history is not None and len(price_history) >= 20:
        # Calculate historical volatility
        returns = np.log(price_history['Close'] / price_history['Close'].shift(1))
        hist_vol = returns.std() * np.sqrt(252)  # Annualized
        
        if atm_options['call'] is not None and 'iv' in atm_options['call']:
            iv_ratio = atm_options['call']['iv'] / hist_vol
            
            risk_profile['iv_vs_hv'] = {
                'iv': float(atm_options['call']['iv']),
                'hv': float(hist_vol),
                'ratio': float(iv_ratio),
                'assessment': "overpriced" if iv_ratio > 1.2 else "underpriced" if iv_ratio < 0.8 else "fair"
            }
    
    return risk_profile

def _find_atm_options(options_data: Dict, current_price: float) -> Dict[str, Optional[Dict]]:
    """
    Find the at-the-money options
    
    Args:
        options_data: Options chain data
        current_price: Current price of the underlying
        
    Returns:
        Dictionary with ATM call and put
    """
    atm_options = {"call": None, "put": None}
    
    # Find closest call to current price
    if 'calls' in options_data:
        calls_df = options_data['calls']
        if not calls_df.empty:
            # Find strike closest to current price
            calls_df['strike_diff'] = abs(calls_df['strike'] - current_price)
            atm_call_idx = calls_df['strike_diff'].idxmin()
            atm_options['call'] = calls_df.loc[atm_call_idx].to_dict()
    
    # Find closest put to current price
    if 'puts' in options_data:
        puts_df = options_data['puts']
        if not puts_df.empty:
            # Find strike closest to current price
            puts_df['strike_diff'] = abs(puts_df['strike'] - current_price)
            atm_put_idx = puts_df['strike_diff'].idxmin()
            atm_options['put'] = puts_df.loc[atm_put_idx].to_dict()
    
    return atm_options

def _generate_options_summary(options_data: Dict) -> Dict:
    """
    Generate a summary of the options analysis
    
    Args:
        options_data: Analyzed options data
        
    Returns:
        Dictionary with options summary
    """
    summary = {
        "symbol": options_data.get('symbol'),
        "expiration_date": options_data.get('expiration_date'),
        "current_price": options_data.get('current_price'),
        "implied_volatility": {}
    }
    
    # Extract IV data
    if 'iv_skew' in options_data:
        iv_skew = options_data['iv_skew']
        summary['implied_volatility'] = {
            "average": iv_skew.get('average_iv'),
            "skew": iv_skew.get('put_call_skew'),
            "term_structure": iv_skew.get('term_structure', {})
        }
    
    # Market sentiment based on IV skew
    summary['market_sentiment'] = {
        "bias": options_data.get('market_bias', 'neutral'),
        "strength": options_data.get('skew_strength', 0)
    }
    
    # Top opportunities
    if 'opportunities' in options_data:
        opportunities = options_data['opportunities']
        
        # Get top bull, bear, and neutral strategies
        top_bull = [op for op in opportunities if op.get('sentiment') == 'bullish']
        top_bear = [op for op in opportunities if op.get('sentiment') == 'bearish']
        top_neutral = [op for op in opportunities if op.get('sentiment') == 'neutral']
        
        # Sort by expected return
        top_bull = sorted(top_bull, key=lambda x: x.get('expected_return', 0), reverse=True)
        top_bear = sorted(top_bear, key=lambda x: x.get('expected_return', 0), reverse=True)
        top_neutral = sorted(top_neutral, key=lambda x: x.get('expected_return', 0), reverse=True)
        
        # Get top recommendation for each
        summary['top_opportunities'] = {
            "bullish": top_bull[0] if top_bull else None,
            "bearish": top_bear[0] if top_bear else None,
            "neutral": top_neutral[0] if top_neutral else None
        }
        
        # Overall recommendation based on market bias
        bias = options_data.get('market_bias', 'neutral')
        if bias == 'bullish' and top_bull:
            summary['recommended_strategy'] = top_bull[0]
        elif bias == 'bearish' and top_bear:
            summary['recommended_strategy'] = top_bear[0]
        else:
            summary['recommended_strategy'] = top_neutral[0] if top_neutral else None
    
    # Risk profile
    if 'risk_profile' in options_data:
        risk = options_data['risk_profile']
        summary['risk_profile'] = {
            "implied_move": risk.get('implied_move', {}).get('percent'),
            "expected_range": risk.get('implied_move', {}).get('expected_range'),
            "days_to_expiration": risk.get('days_to_expiration')
        }
    
    return summary

def find_option_spread_strategies(options_data: Dict, strategy_type: Optional[str] = None) -> List[Dict]:
    """
    Find viable option spread strategies
    
    Args:
        options_data: Analyzed options data
        strategy_type: Type of strategy ('bullish', 'bearish', 'neutral')
        
    Returns:
        List of spread strategies
    """
    logger.info(f"Finding {strategy_type or 'all'} spread strategies for {options_data.get('symbol')}")
    
    if not options_data or 'opportunities' not in options_data:
        logger.warning("Options data not properly analyzed")
        return []
    
    opportunities = options_data['opportunities']
    
    # Filter by strategy type if specified
    if strategy_type:
        opportunities = [op for op in opportunities if op.get('sentiment') == strategy_type]
    
    # Sort by risk-adjusted return
    opportunities = sorted(
        opportunities, 
        key=lambda x: (x.get('expected_return', 0) / max(0.01, x.get('max_risk', 1))),
        reverse=True
    )
    
    # Limit to top 5 strategies
    return opportunities[:5]

def analyze_options_sentiment(options_data: Dict) -> Dict:
    """
    Analyze market sentiment from options data
    
    Args:
        options_data: Analyzed options data
        
    Returns:
        Dictionary with sentiment analysis
    """
    if 'iv_skew' not in options_data or 'market_bias' not in options_data:
        logger.warning("Options data not properly analyzed for sentiment")
        return {"error": "Missing required data for sentiment analysis"}
    
    iv_skew = options_data.get('iv_skew', {})
    market_bias = options_data.get('market_bias', 'neutral')
    skew_strength = options_data.get('skew_strength', 0)
    
    # Get put/call volume ratio if available
    put_volume = options_data.get('puts', pd.DataFrame()).get('volume', pd.Series()).sum()
    call_volume = options_data.get('calls', pd.DataFrame()).get('volume', pd.Series()).sum()
    
    volume_ratio = None
    if put_volume is not None and call_volume is not None and call_volume > 0:
        volume_ratio = put_volume / call_volume
    
    # Get open interest ratio if available
    put_oi = options_data.get('puts', pd.DataFrame()).get('openInterest', pd.Series()).sum()
    call_oi = options_data.get('calls', pd.DataFrame()).get('openInterest', pd.Series()).sum()
    
    oi_ratio = None
    if put_oi is not None and call_oi is not None and call_oi > 0:
        oi_ratio = put_oi / call_oi
    
    # Determine sentiment score (-100 to 100)
    sentiment_score = 0
    
    # IV skew component (-50 to 50)
    pc_skew = iv_skew.get('put_call_skew', 1.0)
    if pc_skew > 1.0:
        # Bearish (put premium > call premium)
        sentiment_score -= min(50, (pc_skew - 1.0) * 100)
    else:
        # Bullish (call premium > put premium)
        sentiment_score += min(50, (1.0 - pc_skew) * 100)
    
    # Volume ratio component (-25 to 25)
    if volume_ratio is not None:
        if volume_ratio > 1.0:
            # Higher put volume (potentially bearish)
            sentiment_score -= min(25, (volume_ratio - 1.0) * 25)
        else:
            # Higher call volume (potentially bullish)
            sentiment_score += min(25, (1.0 - volume_ratio) * 25)
    
    # OI ratio component (-25 to 25)
    if oi_ratio is not None:
        if oi_ratio > 1.0:
            # Higher put OI (potentially bearish)
            sentiment_score -= min(25, (oi_ratio - 1.0) * 25)
        else:
            # Higher call OI (potentially bullish)
            sentiment_score += min(25, (1.0 - oi_ratio) * 25)
    
    # Determine sentiment category
    sentiment_category = "neutral"
    if sentiment_score >= 50:
        sentiment_category = "strongly_bullish"
    elif sentiment_score >= 20:
        sentiment_category = "bullish"
    elif sentiment_score <= -50:
        sentiment_category = "strongly_bearish"
    elif sentiment_score <= -20:
        sentiment_category = "bearish"
    
    return {
        "sentiment_score": float(sentiment_score),
        "sentiment_category": sentiment_category,
        "iv_skew": float(pc_skew),
        "put_call_volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
        "put_call_open_interest_ratio": float(oi_ratio) if oi_ratio is not None else None,
        "skew_strength": float(skew_strength)
    } 