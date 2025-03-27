"""
Pattern Recognition Module for WSB Trading App

This module provides tools for technical chart pattern recognition, 
detection and analysis of various chart patterns.
"""

from src.pattern_recognition.detection import (
    detect_head_and_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triangle,
    detect_flag,
    detect_wedge,
    implement_zigzag
)

from src.pattern_recognition.visualization import (
    visualize_pattern,
    generate_pattern_image,
    quantify_pattern_similarity
)

from src.pattern_recognition.scoring import (
    score_pattern,
    validate_pattern_prediction,
    calculate_pattern_reliability
)
