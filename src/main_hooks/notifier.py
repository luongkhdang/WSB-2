import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_results_to_notion(notion_client, market_analysis, stock_analyses, credit_spreads=None):
    """
    Log analysis results to Notion database
    
    Parameters:
    - notion_client: Notion client
    - market_analysis: Market analysis results
    - stock_analyses: Stock analyses results
    - credit_spreads: Credit spread opportunities (optional)
    
    Returns:
    - Boolean indicating success
    """
    logger.info("Logging results to Notion database")
    
    try:
        if not notion_client:
            logger.warning("Notion client not initialized, skipping logging")
            return False
        
        # Log market analysis
        market_properties = {
            "Date": {"date": {"start": datetime.now().isoformat()}},
            "Trend": {"select": {"name": market_analysis.get("trend", "Neutral")}},
            "VIX": {"number": market_analysis.get("raw_data", {}).get("vix_price", 0)},
            "SPY Price": {"number": market_analysis.get("raw_data", {}).get("spy_price", 0)},
            "Technical Score": {"number": market_analysis.get("market_trend_score", 0)},
            "Risk Adjustment": {"select": {"name": market_analysis.get("risk_adjustment", "Standard")}}
        }
        
        notion_client.add_market_log_entry(market_properties)
        
        # Log stock analyses - focus on top stocks by technical score
        sorted_stocks = sorted(
            [(sym, analysis) for sym, analysis in stock_analyses.items() if 'technical_score' in analysis],
            key=lambda x: x[1].get('technical_score', 0),
            reverse=True
        )
        
        # Log top 5 stocks
        for symbol, analysis in sorted_stocks[:5]:
            stock_properties = {
                "Symbol": {"title": [{"text": {"content": symbol}}]},
                "Date": {"date": {"start": datetime.now().isoformat()}},
                "Price": {"number": analysis.get("raw_data", {}).get("current_price", 0)},
                "Trend": {"select": {"name": analysis.get("trend", "Neutral")}},
                "Technical Score": {"number": analysis.get("technical_score", 0)},
                "Fundamental Score": {"number": analysis.get("fundamental_score", 0)},
                "Sentiment Score": {"number": analysis.get("sentiment_score", 0)},
                "ATR %": {"number": analysis.get("raw_data", {}).get("atr_percent", 0)},
                "Market Alignment": {"select": {"name": analysis.get("market_alignment", "Neutral")}}
            }
            
            if "options_summary" in analysis and analysis["options_summary"]:
                # Add first expiration's IV data if available
                for exp_date, exp_data in list(analysis["options_summary"].items())[:1]:
                    stock_properties["Call IV"] = {"number": exp_data.get("avg_call_iv", 0) * 100}
                    stock_properties["Put IV"] = {"number": exp_data.get("avg_put_iv", 0) * 100}
                    stock_properties["IV Skew"] = {"number": exp_data.get("iv_skew", 0)}
                    break
            
            notion_client.add_stock_log_entry(stock_properties)
        
        # Log top credit spread opportunities if available
        if credit_spreads and len(credit_spreads) > 0:
            # Sort by total score and take top 5
            sorted_spreads = sorted(credit_spreads, key=lambda x: x.get("total_score", 0), reverse=True)[:5]
            
            for spread in sorted_spreads:
                if not hasattr(notion_client, 'trade_log_page_id') or not notion_client.trade_log_page_id:
                    logger.warning("No trade log page ID configured, skipping credit spread logging")
                    break
                    
                properties = {
                    "Symbol": {"title": [{"text": {"content": spread.get('symbol', '')}}]},
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                    "Type": {"select": {"name": spread.get("spread_type", "Unknown").title()}},
                    "Direction": {"select": {"name": spread.get("direction", "Neutral").title()}},
                    "Strikes": {"rich_text": [{"text": {"content": spread.get("strikes", "")}}]},
                    "Expiration": {"date": {"start": spread.get("expiration", "2025-04-01")}},
                    "Quality Score": {"number": spread.get("quality_score", 0)},
                    "Gamble Score": {"number": spread.get("gamble_score", 0)},
                    "Total Score": {"number": spread.get("total_score", 0)},
                    "Success Probability": {"number": spread.get("success_probability", 0)},
                    "Status": {"select": {"name": "Identified"}}
                }
                
                notion_client.add_trade_log_entry(properties)
            
            logger.info(f"Logged {len(sorted_spreads)} credit spread opportunities to Notion")
        
        logger.info(f"Successfully logged market and {len(sorted_stocks[:5])} stock analyses to Notion")
        return True
        
    except Exception as e:
        logger.error(f"Error logging results to Notion: {e}")
        return False

def log_pretraining_evaluation(notion_client, discord_client, ticker, evaluation):
    """
    Log pretraining evaluation results to Notion and Discord
    
    Parameters:
    - notion_client: Notion client
    - discord_client: Discord client
    - ticker: Stock symbol evaluated
    - evaluation: Evaluation results
    
    Returns:
    - Boolean indicating success
    """
    logger.info(f"Logging pretraining evaluation for {ticker}")
    
    try:
        success = False
        
        # Log to Notion if available
        if notion_client:
            try:
                metrics = evaluation.get("metrics", {})
                
                # Create properties for Notion
                properties = {
                    "Symbol": {"title": [{"text": {"content": ticker}}]},
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                    "Evaluation Period": {"rich_text": [{"text": {"content": evaluation.get("evaluation_period", "")}}]},
                    "Predictions": {"number": evaluation.get("prediction_count", 0)}
                }
                
                # Add metrics for each horizon
                for horizon, results in metrics.items():
                    horizon_name = horizon.replace("_", " ").title()
                    directional_accuracy = float(results.get("directional_accuracy", "0%").replace("%", ""))
                    magnitude_error = float(results.get("avg_magnitude_error", "0%").replace("%", ""))
                    
                    properties[f"{horizon_name} Accuracy"] = {"number": directional_accuracy}
                    properties[f"{horizon_name} Error"] = {"number": magnitude_error}
                
                # Add to Notion database
                if hasattr(notion_client, 'pretraining_database_id') and notion_client.pretraining_database_id:
                    notion_client.create_pretraining_evaluation(properties)
                    logger.info(f"Logged pretraining evaluation for {ticker} to Notion")
                    success = True
            
            except Exception as notion_error:
                logger.error(f"Error logging to Notion: {notion_error}")
        
        # Log to Discord if available
        if discord_client:
            try:
                metrics = evaluation.get("metrics", {})
                
                # Format evaluation message
                message = f"**Pretraining Evaluation for {ticker}**\n"
                message += f"Period: {evaluation.get('evaluation_period', '')}\n"
                message += f"Total predictions: {evaluation.get('prediction_count', 0)}\n\n"
                
                for horizon, results in metrics.items():
                    message += f"**{horizon.replace('_', ' ').title()} Horizon**\n"
                    message += f"• Directional Accuracy: {results.get('directional_accuracy')}\n"
                    message += f"• Avg. Magnitude Error: {results.get('avg_magnitude_error')}\n"
                    message += f"• Sample Size: {results.get('sample_size')}\n\n"
                
                discord_client.send_message(message)
                logger.info(f"Sent pretraining evaluation for {ticker} to Discord")
                success = True
            
            except Exception as discord_error:
                logger.error(f"Error sending to Discord: {discord_error}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error logging pretraining evaluation: {e}")
        return False 