"""
Test script for the enhanced market analysis with multiple indices
"""
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_market_analysis')

def test_market_analysis():
    """Test the enhanced market analysis with SPY, QQQ, IWM, VTV, and VGLT"""
    try:
        # Attempt to import the required modules
        from src.main import WSBTradingApp
        
        logger.info("Testing enhanced market analysis...")
        
        # Initialize the app
        app = WSBTradingApp()
        
        # Run the market analysis
        market_analysis = app.analyze_market()
        
        # Print the results
        print("\n=== MARKET ANALYSIS RESULTS ===")
        print(f"Overall Market Trend: {market_analysis.get('trend', 'Unknown')}")
        print(f"Market Trend Score: {market_analysis.get('market_trend_score', 0)}/30")
        print(f"Risk Adjustment: {market_analysis.get('risk_adjustment', 'Unknown')}")
        
        # Print sector rotation if available
        sector_rotation = market_analysis.get('sector_rotation', '')
        if sector_rotation:
            print(f"\nSector Rotation: {sector_rotation}")
        
        # Print VIX assessment if available
        vix_assessment = market_analysis.get('vix_assessment', '')
        if vix_assessment:
            print(f"\nVIX Assessment: {vix_assessment}")
        
        # Print included indices
        print("\n=== INDICES ANALYZED ===")
        raw_data = market_analysis.get('raw_data', {})
        
        # SPY data
        spy_price = raw_data.get('spy_price', 'N/A')
        spy_change = raw_data.get('spy_daily_pct_change', 'N/A')
        print(f"SPY: ${spy_price} ({spy_change:.2f}% daily change)" if isinstance(spy_change, float) else f"SPY: ${spy_price}")
        
        # QQQ data
        qqq_price = raw_data.get('qqq', {}).get('regularMarketPrice', 'N/A')
        qqq_change = raw_data.get('qqq', {}).get('dailyPctChange', 'N/A')
        print(f"QQQ: ${qqq_price} ({qqq_change:.2f}% daily change)" if isinstance(qqq_change, float) else f"QQQ: ${qqq_price}")
        
        # IWM data
        iwm_price = raw_data.get('iwm', {}).get('regularMarketPrice', 'N/A')
        iwm_change = raw_data.get('iwm', {}).get('dailyPctChange', 'N/A')
        print(f"IWM: ${iwm_price} ({iwm_change:.2f}% daily change)" if isinstance(iwm_change, float) else f"IWM: ${iwm_price}")
        
        # VTV data
        vtv_price = raw_data.get('vtv', {}).get('regularMarketPrice', 'N/A')
        vtv_change = raw_data.get('vtv', {}).get('dailyPctChange', 'N/A')
        print(f"VTV: ${vtv_price} ({vtv_change:.2f}% daily change)" if isinstance(vtv_change, float) else f"VTV: ${vtv_price}")
        
        # VGLT data
        vglt_price = raw_data.get('vglt', {}).get('regularMarketPrice', 'N/A')
        vglt_change = raw_data.get('vglt', {}).get('dailyPctChange', 'N/A')
        print(f"VGLT: ${vglt_price} ({vglt_change:.2f}% daily change)" if isinstance(vglt_change, float) else f"VGLT: ${vglt_price}")
        
        # VIX data
        vix_price = raw_data.get('vix_price', 'N/A')
        print(f"VIX: {vix_price}")
        
        print("\n=== ANALYSIS SUMMARY ===")
        full_analysis = market_analysis.get('full_analysis', 'No analysis available')
        # Print first 500 characters of the full analysis
        print(f"{full_analysis[:500]}...")
        
        return market_analysis
        
    except ModuleNotFoundError as e:
        logger.error(f"Missing module: {e}")
        print(f"\nERROR: Missing module - {e}")
        print("Please ensure all required modules are installed.")
        print("Required modules may include: gemini, notion_client, etc.")
        return None
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        print(f"\nERROR: {e}")
        return None

if __name__ == "__main__":
    test_market_analysis()