"""
Pattern Visualization

This module provides functions for visualizing detected chart patterns,
generating pattern images, and calculating pattern similarity.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
from dtw import dtw  # Dynamic Time Warping for pattern similarity

logger = logging.getLogger(__name__)


def visualize_pattern(data: pd.DataFrame, pattern_info: Dict[str, Any],
                      save_path: Optional[str] = None) -> Optional[Figure]:
    """
    Visualize a detected pattern on price chart

    Parameters:
    - data: DataFrame with price data
    - pattern_info: Dictionary with pattern details
    - save_path: Optional path to save the visualization

    Returns:
    - Matplotlib Figure object if successful, None otherwise
    """
    logger.info(
        f"Visualizing {pattern_info.get('pattern', 'unknown')} pattern")

    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price data
        ax.plot(data.index, data['Close'], 'k-', label='Price')

        # Plot zigzag if available
        if 'zigzag' in data.columns:
            # Only plot non-NaN values and connect them
            zigzag_data = data.dropna(subset=['zigzag'])
            ax.plot(zigzag_data.index,
                    zigzag_data['zigzag'], 'r-', linewidth=2, label='Pivots')

            # Mark pivot points
            ax.scatter(zigzag_data.index, zigzag_data['zigzag'],
                       color='red', s=50, zorder=5)

        # Highlight pattern region if available
        if 'start_idx' in pattern_info and 'end_idx' in pattern_info:
            start_idx = pattern_info['start_idx']
            end_idx = pattern_info['end_idx']
            ax.axvspan(data.index[start_idx], data.index[end_idx],
                       alpha=0.2, color='green', label='Pattern Region')

        # Add pattern-specific visualization elements
        pattern_type = pattern_info.get('pattern', '')

        if pattern_type == 'head_and_shoulders':
            # Mark left shoulder, head, right shoulder if available
            if all(k in pattern_info for k in ['left_shoulder', 'head', 'right_shoulder']):
                ls_idx = pattern_info['left_shoulder']
                h_idx = pattern_info['head']
                rs_idx = pattern_info['right_shoulder']

                # Mark points
                ax.scatter([data.index[ls_idx], data.index[h_idx], data.index[rs_idx]],
                           [data['High'].iloc[ls_idx], data['High'].iloc[h_idx],
                               data['High'].iloc[rs_idx]],
                           color='blue', s=100, zorder=5)

                # Add labels
                ax.annotate('LS', (data.index[ls_idx], data['High'].iloc[ls_idx]),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom', fontweight='bold')
                ax.annotate('H', (data.index[h_idx], data['High'].iloc[h_idx]),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom', fontweight='bold')
                ax.annotate('RS', (data.index[rs_idx], data['High'].iloc[rs_idx]),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom', fontweight='bold')

        # Add neckline for head and shoulders or double top/bottom
        if 'neckline' in pattern_info:
            neckline = pattern_info['neckline']
            ax.axhline(y=neckline, color='blue', linestyle='--',
                       label='Neckline')

        # Add target price if available
        if 'target_price' in pattern_info:
            target = pattern_info['target_price']
            ax.axhline(y=target, color='green', linestyle='-.',
                       label=f'Target: {target:.2f}')

        # Add title and legend
        confidence = pattern_info.get('confidence', 0)
        ax.set_title(
            f"{pattern_type.replace('_', ' ').title()} Pattern (Confidence: {confidence}%)")
        ax.legend()

        # Format x-axis to show dates nicely
        fig.autofmt_xdate()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved pattern visualization to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error visualizing pattern: {e}")
        return None


def generate_pattern_image(data: pd.DataFrame, pattern_info: Dict[str, Any]) -> str:
    """
    Generate a base64 encoded image of the pattern for embedding

    Parameters:
    - data: DataFrame with price data
    - pattern_info: Dictionary with pattern details

    Returns:
    - Base64 encoded string of the pattern image
    """
    try:
        fig = visualize_pattern(data, pattern_info)
        if fig is None:
            return ""

        # Convert plot to PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Convert PNG to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return image_base64

    except Exception as e:
        logger.error(f"Error generating pattern image: {e}")
        return ""


def quantify_pattern_similarity(pattern1: pd.Series, pattern2: pd.Series) -> float:
    """
    Compare two patterns for similarity using Dynamic Time Warping

    Parameters:
    - pattern1: First pattern series (usually normalized price data)
    - pattern2: Second pattern series

    Returns:
    - Similarity score (0-100, higher is more similar)
    """
    try:
        # Ensure we have numeric data
        p1 = np.array(pattern1.values, dtype=float)
        p2 = np.array(pattern2.values, dtype=float)

        # Normalize both patterns to range [0, 1]
        p1_norm = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))
        p2_norm = (p2 - np.min(p2)) / (np.max(p2) - np.min(p2))

        # Use Dynamic Time Warping to calculate distance
        # Lower distance = more similar
        d, _, _, _ = dtw(p1_norm, p2_norm, dist=lambda x, y: np.abs(x - y))

        # Convert distance to similarity score (0-100)
        # A perfect match would have distance 0
        # The maximum possible distance for normalized data is the length
        # We'll use a simple exponential decay function to convert distance to similarity
        similarity = 100 * np.exp(-d)

        return similarity

    except Exception as e:
        logger.error(f"Error calculating pattern similarity: {e}")
        return 0.0
