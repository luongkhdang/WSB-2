"""
Gemini prompt hooks package.
"""

from .market_analysis import (
    get_market_trend_prompt,
    get_spy_options_prompt,
    get_market_data_prompt
)

from .stock_analysis import (
    get_stock_analysis_prompt,
    get_stock_options_prompt
)

from .trade_plan import get_trade_plan_prompt

__all__ = [
    'get_market_trend_prompt',
    'get_spy_options_prompt',
    'get_market_data_prompt',
    'get_stock_analysis_prompt',
    'get_stock_options_prompt',
    'get_trade_plan_prompt'
] 