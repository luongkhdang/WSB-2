#!/usr/bin/env python3
"""
Pretraining Integration Module (src/main_hooks/pretraining_integration.py)
-------------------------------------------------------------------------
Integrates multiple pretraining implementations and provides selection logic.

Functions:
  - use_optimized_pretraining - Determines whether to use optimized implementation
  - use_six_step_pretraining - Determines whether to use six-step process
  - integrated_pretrain_analyzer - Wrapper for single ticker pretraining
  - integrated_batch_pretrain_analyzer - Wrapper for batch pretraining
  - calculate_pretraining_improvement - Compares results between implementations

Dependencies:
  - src.main_hooks.pretraining - Traditional pretraining implementation
  - src.main_hooks.optimized_pretraining - Optimized implementation
  - src.main_hooks.six_step_pretraining - Six-step implementation
  - Environment variables: WSB_USE_OPTIMIZED_PRETRAINING, WSB_USE_SIX_STEP_PRETRAINING

Used by:
  - main.py for selecting the appropriate pretraining implementation
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Import both traditional and optimized pretraining
from src.main_hooks.pretraining import pretrain_analyzer, batch_pretrain_analyzer
from src.main_hooks.optimized_pretraining import optimized_pretrain_analyzer, optimized_batch_pretrain_analyzer
from src.main_hooks.six_step_pretraining import six_step_pretrain_analyzer, six_step_batch_pretrain_analyzer

logger = logging.getLogger(__name__)


def use_optimized_pretraining():
    """Determine whether to use optimized pretraining based on environment variables."""
    # Check environment variable
    return os.getenv('WSB_USE_OPTIMIZED_PRETRAINING', 'true').lower() == 'true'


def use_six_step_pretraining():
    """Determine whether to use the six-step pretraining process."""
    # Check environment variable - default to true for enhanced performance
    return os.getenv('WSB_USE_SIX_STEP_PRETRAINING', 'true').lower() == 'true'


def integrated_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    ticker,
    pretraining_prompt_hook=None,  # This parameter is ignored in the optimized version
    start_date=None,
    end_date=None,
    save_results=True,
    callback=None,
    discord_client=None  # Add discord_client parameter
):
    """
    Intelligent wrapper for pretraining that selects either the original, optimized, or six-step implementation.

    Parameters:
    - Same as pretrain_analyzer, with pretraining_prompt_hook becoming optional
    - discord_client: Optional Discord client for tracking Gemini responses

    Returns:
    - Dictionary containing pretraining results and context for future analysis
    """
    if use_six_step_pretraining():
        logger.info(f"Using six-step pretraining for {ticker}")
        return six_step_pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            ticker,
            start_date=start_date,
            end_date=end_date,
            save_results=save_results,
            callback=callback,
            discord_client=discord_client,
            strict_validation=False  # Never use strict validation - always fall back to defaults
        )
    elif use_optimized_pretraining():
        logger.info(f"Using optimized pretraining for {ticker}")
        return optimized_pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            ticker,
            start_date=start_date,
            end_date=end_date,
            save_results=save_results,
            callback=callback,
            discord_client=discord_client
        )
    else:
        logger.info(f"Using traditional pretraining for {ticker}")
        return pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            ticker,
            pretraining_prompt_hook,
            start_date=start_date,
            end_date=end_date,
            save_results=save_results,
            callback=callback
        )


def integrated_batch_pretrain_analyzer(
    yfinance_client,
    gemini_client,
    pretraining_dir,
    tickers,
    pretraining_prompt_hook=None,  # This parameter is ignored in the optimized version
    discord_client=None,  # Add discord_client parameter
    **kwargs
):
    """
    Intelligent wrapper for batch pretraining that selects either the original, optimized, or six-step implementation.

    Parameters:
    - Same as batch_pretrain_analyzer, with pretraining_prompt_hook becoming optional
    - discord_client: Optional Discord client for tracking Gemini responses

    Returns:
    - Dictionary with results for each ticker
    """
    if use_six_step_pretraining():
        logger.info(
            f"Using six-step batch pretraining for {len(tickers)} tickers")

        # Make a copy of kwargs and ensure strict_validation is False
        batch_kwargs = kwargs.copy()
        # Always use fallback data instead of raising errors
        batch_kwargs['strict_validation'] = False

        return six_step_batch_pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            tickers,
            discord_client=discord_client,
            **batch_kwargs
        )
    elif use_optimized_pretraining():
        logger.info(
            f"Using optimized batch pretraining for {len(tickers)} tickers")
        return optimized_batch_pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            tickers,
            discord_client=discord_client,
            **kwargs
        )
    else:
        logger.info(
            f"Using traditional batch pretraining for {len(tickers)} tickers")
        return batch_pretrain_analyzer(
            yfinance_client,
            gemini_client,
            pretraining_dir,
            tickers,
            pretraining_prompt_hook,
            **kwargs
        )


def calculate_pretraining_improvement(original_result, optimized_result):
    """
    Calculate the improvement metrics between original and optimized pretraining results.

    Parameters:
    - original_result: Result dict from traditional pretraining
    - optimized_result: Result dict from optimized pretraining

    Returns:
    - Dict with improvement metrics
    """
    if "error" in original_result or "error" in optimized_result:
        return {"error": "Cannot calculate improvement due to errors in results"}

    # Calculate time improvement
    original_time = original_result.get("processing_time", 0)
    optimized_time = optimized_result.get("processing_time", 0)

    time_improvement = 0
    if original_time > 0:
        time_improvement = (original_time - optimized_time) / \
            original_time * 100

    # Calculate API call reduction
    original_api_calls = len(original_result.get("results", []))
    optimized_api_calls = len(optimized_result.get("results", []))

    api_call_reduction = 0
    if original_api_calls > 0:
        api_call_reduction = (original_api_calls -
                              optimized_api_calls) / original_api_calls * 100

    # Check for memory context improvement
    original_memory = original_result.get("memory_context", {})
    optimized_memory = optimized_result.get("memory_context", {})

    # Check pattern recognition improvement
    original_patterns = 0
    if "pattern_library" in original_memory:
        original_patterns = sum(len(patterns) for _, patterns in original_memory.get(
            "pattern_library", {}).items())

    optimized_patterns = 0
    if "pattern_library" in optimized_memory:
        optimized_patterns = sum(len(patterns) for _, patterns in optimized_memory.get(
            "pattern_library", {}).items())

    pattern_improvement = 0
    if original_patterns > 0:
        pattern_improvement = (optimized_patterns -
                               original_patterns) / original_patterns * 100

    # Compare prediction accuracy if available
    original_accuracy = original_result.get("accuracy_rate", 0)
    optimized_accuracy = optimized_result.get("accuracy_rate", 0)

    accuracy_improvement = 0
    if original_accuracy > 0:
        accuracy_improvement = (optimized_accuracy -
                                original_accuracy) / original_accuracy * 100

    return {
        "time_improvement": time_improvement,
        "api_call_reduction": api_call_reduction,
        "pattern_improvement": pattern_improvement,
        "accuracy_improvement": accuracy_improvement,
        "original_time": original_time,
        "optimized_time": optimized_time,
        "original_api_calls": original_api_calls,
        "optimized_api_calls": optimized_api_calls,
        "original_patterns": original_patterns,
        "optimized_patterns": optimized_patterns,
        "original_accuracy": original_accuracy,
        "optimized_accuracy": optimized_accuracy
    }
