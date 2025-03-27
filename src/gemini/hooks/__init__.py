#!/usr/bin/env python3
"""
Gemini Prompts Package (src/gemini/hooks/__init__.py)
---------------------------------------------------
Provides prompt generation functions for different AI analysis types.

Exports:
  - Market Analysis: get_market_trend_prompt, get_spy_options_prompt, get_market_data_prompt
  - Stock Analysis: get_stock_analysis_prompt, get_stock_options_prompt
  - Trade Planning: get_trade_plan_prompt, get_trade_plan_prompt_from_context
  - Pretraining: get_pretraining_prompt, generate_pretraining_context

Dependencies:
  - Individual modules for each prompt type

Used by:
  - main.py and main_hooks modules for generating prompts for Gemini AI
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

from .trade_plan import (
    get_trade_plan_prompt,
    get_trade_plan_prompt_from_context
)

from .gemini_pretraining import (
    get_pretraining_prompt,
    generate_pretraining_context
)

__all__ = [
    'get_market_trend_prompt',
    'get_spy_options_prompt',
    'get_market_data_prompt',
    'get_stock_analysis_prompt',
    'get_stock_options_prompt',
    'get_trade_plan_prompt',
    'get_trade_plan_prompt_from_context',
    'get_pretraining_prompt',
    'generate_pretraining_context'
]
