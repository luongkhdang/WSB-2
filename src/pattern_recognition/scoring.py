"""
Pattern Scoring and Validation

This module provides functions for scoring detected patterns,
validating pattern predictions, and calculating pattern reliability.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)


def score_pattern(pattern_info: Dict[str, Any], data: pd.DataFrame) -> float:
    """
    Score a detected pattern based on clarity and other factors

    Parameters:
    - pattern_info: Dictionary with pattern details
    - data: DataFrame with price data

    Returns:
    - Score from 0-100 indicating pattern quality
    """
    logger.info(f"Scoring {pattern_info.get('pattern', 'unknown')} pattern")

    base_score = 50  # Start with neutral score
    adjustments = 0

    pattern_type = pattern_info.get('pattern', '')

    # Skip scoring if essential pattern details are missing
    if not pattern_type:
        logger.warning("Cannot score pattern: missing pattern type")
        return 0

    # 1. Pattern Clarity Score
    clarity_score = calculate_pattern_clarity(pattern_info, data)

    # 2. Volume confirmation
    volume_score = calculate_volume_confirmation(pattern_info, data)

    # 3. Timeframe consideration (patterns on longer timeframes are more reliable)
    tf_multiplier = pattern_info.get('timeframe_multiplier', 1.0)

    # 4. Pattern symmetry (for patterns where symmetry matters)
    symmetry_score = 0
    if pattern_type in ['head_and_shoulders', 'double_top', 'double_bottom']:
        symmetry_score = calculate_pattern_symmetry(pattern_info, data)

    # 5. Apply adjustments for timeframe and specific pattern factors
    adjustments = (clarity_score + volume_score +
                   symmetry_score) * tf_multiplier

    # Calculate final score, clamped to 0-100 range
    final_score = min(max(base_score + adjustments, 0), 100)

    logger.debug(f"Pattern scored {final_score:.1f}/100")
    return final_score


def calculate_pattern_clarity(pattern_info: Dict[str, Any], data: pd.DataFrame) -> float:
    """
    Calculate how clearly defined a pattern is

    Parameters:
    - pattern_info: Dictionary with pattern details
    - data: DataFrame with price data

    Returns:
    - Clarity score adjustment (-20 to +20)
    """
    clarity_score = 0

    # Check if pattern has clear pivot points
    if pattern_info.get('pivot_strength', 0) > 0:
        clarity_score += min(pattern_info['pivot_strength'] * 5, 10)

    # Check if pattern has clean breakout/breakdown
    if pattern_info.get('clean_break', False):
        clarity_score += 5

    # Penalize if there's excessive noise in the pattern
    if pattern_info.get('noise_level', 0) > 0:
        clarity_score -= min(pattern_info['noise_level'] * 5, 15)

    return clarity_score


def calculate_volume_confirmation(pattern_info: Dict[str, Any], data: pd.DataFrame) -> float:
    """
    Calculate volume confirmation score for the pattern

    Parameters:
    - pattern_info: Dictionary with pattern details
    - data: DataFrame with price data

    Returns:
    - Volume confirmation score adjustment (-10 to +10)
    """
    volume_score = 0

    # Check if we have volume data
    if 'Volume' not in data.columns:
        return 0

    pattern_type = pattern_info.get('pattern', '')

    # Check for increasing volume on breakout
    if pattern_info.get('breakout_idx') is not None:
        breakout_idx = pattern_info['breakout_idx']

        if breakout_idx > 0 and breakout_idx < len(data):
            avg_volume = data['Volume'].iloc[max(
                0, breakout_idx-5):breakout_idx].mean()
            breakout_volume = data['Volume'].iloc[breakout_idx]

            if breakout_volume > avg_volume * 1.5:
                volume_score += 10
            elif breakout_volume > avg_volume * 1.2:
                volume_score += 5

    return volume_score


def calculate_pattern_symmetry(pattern_info: Dict[str, Any], data: pd.DataFrame) -> float:
    """
    Calculate symmetry score for applicable patterns

    Parameters:
    - pattern_info: Dictionary with pattern details
    - data: DataFrame with price data

    Returns:
    - Symmetry score adjustment (-10 to +10)
    """
    symmetry_score = 0
    pattern_type = pattern_info.get('pattern', '')

    if pattern_type == 'head_and_shoulders':
        if all(k in pattern_info for k in ['left_shoulder', 'head', 'right_shoulder']):
            ls_idx = pattern_info['left_shoulder']
            h_idx = pattern_info['head']
            rs_idx = pattern_info['right_shoulder']

            # Calculate heights
            ls_height = data['High'].iloc[ls_idx] - \
                pattern_info.get('neckline', 0)
            rs_height = data['High'].iloc[rs_idx] - \
                pattern_info.get('neckline', 0)

            # Calculate time symmetry
            ls_time_dist = h_idx - ls_idx
            rs_time_dist = rs_idx - h_idx

            # Height symmetry (shoulders should be similar height)
            height_ratio = min(ls_height, rs_height) / \
                max(ls_height, rs_height)

            # Time symmetry
            time_ratio = min(ls_time_dist, rs_time_dist) / \
                max(ls_time_dist, rs_time_dist)

            # Score based on symmetry (perfect symmetry = 1.0 ratio)
            if height_ratio > 0.8 and time_ratio > 0.8:
                symmetry_score += 10
            elif height_ratio > 0.6 and time_ratio > 0.6:
                symmetry_score += 5
            else:
                symmetry_score -= 5

    return symmetry_score


def validate_pattern_prediction(pattern_info: Dict[str, Any],
                                future_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate how well a pattern predicted future price action

    Parameters:
    - pattern_info: Dictionary with pattern details and prediction
    - future_data: DataFrame with future price data after the pattern

    Returns:
    - Dictionary with validation results
    """
    logger.info(
        f"Validating prediction for {pattern_info.get('pattern', 'unknown')} pattern")

    # Initialize validation result
    validation = {
        'pattern': pattern_info.get('pattern', 'unknown'),
        'predicted_direction': pattern_info.get('predicted_direction', 'unknown'),
        'predicted_target': pattern_info.get('target_price', 0),
        'predicted_time': pattern_info.get('target_time', 0),
        'actual_direction': 'unknown',
        'actual_change': 0,
        'actual_change_pct': 0,
        'time_to_target': 0,
        'direction_correct': False,
        'target_reached': False,
        'magnitude_error': 0,
        'timing_error': 0,
        'overall_accuracy': 0
    }

    # Skip validation if we don't have future data
    if future_data is None or len(future_data) == 0:
        logger.warning("Cannot validate pattern: no future data provided")
        return validation

    try:
        # Get pattern end point and price
        end_idx = pattern_info.get('end_idx', 0)
        end_price = pattern_info.get('end_price', future_data['Close'].iloc[0])

        # Get target price and predicted direction
        target_price = pattern_info.get('target_price', 0)
        predicted_direction = pattern_info.get('predicted_direction', '')

        # Determine actual direction and change
        if len(future_data) > 0:
            last_price = future_data['Close'].iloc[-1]
            actual_change = last_price - end_price
            actual_change_pct = (actual_change / end_price) * 100

            if actual_change > 0:
                actual_direction = 'bullish'
            elif actual_change < 0:
                actual_direction = 'bearish'
            else:
                actual_direction = 'neutral'

            validation['actual_direction'] = actual_direction
            validation['actual_change'] = actual_change
            validation['actual_change_pct'] = actual_change_pct

            # Check if direction was correct
            direction_correct = (
                (predicted_direction == 'bullish' and actual_direction == 'bullish') or
                (predicted_direction == 'bearish' and actual_direction == 'bearish') or
                (predicted_direction == 'neutral' and actual_direction == 'neutral')
            )
            validation['direction_correct'] = direction_correct

            # Check if target was reached
            target_reached = False
            time_to_target = 0

            if predicted_direction == 'bullish' and target_price > 0:
                # For bullish prediction, check if price rose to target
                for i, row in enumerate(future_data.iterrows()):
                    if row[1]['High'] >= target_price:
                        target_reached = True
                        time_to_target = i + 1
                        break

            elif predicted_direction == 'bearish' and target_price > 0:
                # For bearish prediction, check if price fell to target
                for i, row in enumerate(future_data.iterrows()):
                    if row[1]['Low'] <= target_price:
                        target_reached = True
                        time_to_target = i + 1
                        break

            validation['target_reached'] = target_reached
            validation['time_to_target'] = time_to_target

            # Calculate errors
            predicted_pct_change = (
                (target_price - end_price) / end_price) * 100
            validation['magnitude_error'] = abs(
                predicted_pct_change - actual_change_pct)

            # Calculate overall accuracy (0-100)
            # 60% weight on direction, 40% weight on magnitude error (inversely)
            direction_score = 60 if direction_correct else 0
            max_magnitude_error = 20  # Cap magnitude error at 20%
            magnitude_score = 40 * \
                (1 - min(validation['magnitude_error'],
                 max_magnitude_error) / max_magnitude_error)

            validation['overall_accuracy'] = direction_score + magnitude_score

        return validation

    except Exception as e:
        logger.error(f"Error validating pattern prediction: {e}")
        return validation


def calculate_pattern_reliability(pattern_type: str, validation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate reliability statistics for a pattern type based on historical validations

    Parameters:
    - pattern_type: Type of pattern to analyze
    - validation_history: List of validation dictionaries from validate_pattern_prediction

    Returns:
    - Dictionary with reliability statistics
    """
    logger.info(f"Calculating reliability for {pattern_type} pattern")

    # Filter to just this pattern type
    pattern_validations = [
        v for v in validation_history if v.get('pattern') == pattern_type]

    if not pattern_validations:
        logger.warning(f"No validation history for {pattern_type} pattern")
        return {
            'pattern': pattern_type,
            'sample_size': 0,
            'direction_accuracy': 0,
            'target_reached_rate': 0,
            'avg_magnitude_error': 0,
            'avg_accuracy': 0,
            'reliability_score': 0
        }

    # Calculate statistics
    sample_size = len(pattern_validations)
    direction_correct_count = sum(
        1 for v in pattern_validations if v.get('direction_correct', False))
    target_reached_count = sum(
        1 for v in pattern_validations if v.get('target_reached', False))

    direction_accuracy = (direction_correct_count / sample_size) * 100
    target_reached_rate = (target_reached_count / sample_size) * 100

    # Calculate average errors and accuracy
    magnitude_errors = [v.get('magnitude_error', 0)
                        for v in pattern_validations]
    overall_accuracies = [v.get('overall_accuracy', 0)
                          for v in pattern_validations]

    avg_magnitude_error = sum(magnitude_errors) / sample_size
    avg_accuracy = sum(overall_accuracies) / sample_size

    # Calculate overall reliability score (0-100)
    # 50% based on direction accuracy, 30% on target reached rate, 20% on sample size
    sample_size_score = min(20, sample_size) / 20 * 20  # Max out at 20 samples
    reliability_score = (direction_accuracy * 0.5) + \
        (target_reached_rate * 0.3) + sample_size_score

    return {
        'pattern': pattern_type,
        'sample_size': sample_size,
        'direction_accuracy': direction_accuracy,
        'target_reached_rate': target_reached_rate,
        'avg_magnitude_error': avg_magnitude_error,
        'avg_accuracy': avg_accuracy,
        'reliability_score': reliability_score
    }
