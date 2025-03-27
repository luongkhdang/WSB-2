import logging
import pandas as pd
import numpy as np
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List, Any, Union

logger = logging.getLogger(__name__)

def validate_historical_data(data: pd.DataFrame, 
                            ticker: str, 
                            min_rows: int = 20,
                            require_volume: bool = True) -> Tuple[bool, str, float]:
    """
    Validate historical stock data to ensure quality
    
    Args:
        data: DataFrame with historical data
        ticker: Stock ticker symbol
        min_rows: Minimum number of rows required
        require_volume: Whether volume data is required
        
    Returns:
        Tuple of (is_valid, message, quality_score)
    """
    if data is None or data.empty:
        return False, "Data is empty", 0.0
    
    # Start with perfect score and subtract for issues
    quality_score = 100.0
    issues = []
    
    # Check number of rows
    if len(data) < min_rows:
        quality_score -= 30.0
        issues.append(f"Insufficient data: {len(data)} rows")
        
        if len(data) < 5:  # Critical threshold
            return False, "Too few data points", quality_score
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    if require_volume:
        required_columns.append('Volume')
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        quality_score -= 20.0 * len(missing_columns) / len(required_columns)
        issues.append(f"Missing columns: {', '.join(missing_columns)}")
        
        if 'Close' in missing_columns:  # Critical column
            return False, "Missing Close prices", quality_score
    
    # Check for NaN values
    nan_counts = data[data.columns.intersection(required_columns)].isna().sum()
    total_required_values = len(data) * len(data.columns.intersection(required_columns))
    nan_percentage = nan_counts.sum() / total_required_values if total_required_values > 0 else 0
    
    if nan_percentage > 0:
        quality_score -= min(40.0, nan_percentage * 100)
        issues.append(f"Contains {nan_percentage:.1%} NaN values")
        
        if nan_percentage > 0.2:  # More than 20% NaNs
            return False, f"Too many NaN values ({nan_percentage:.1%})", quality_score
    
    # Check for duplicate indices
    duplicate_indices = data.index.duplicated().sum()
    if duplicate_indices > 0:
        quality_score -= min(20.0, duplicate_indices / len(data) * 100)
        issues.append(f"Contains {duplicate_indices} duplicate timestamps")
    
    # Check for price anomalies (e.g., zero or negative prices)
    if 'Close' in data.columns:
        zero_prices = (data['Close'] <= 0).sum()
        if zero_prices > 0:
            quality_score -= min(30.0, zero_prices / len(data) * 100)
            issues.append(f"Contains {zero_prices} zero/negative prices")
            
            if zero_prices > len(data) * 0.1:  # More than 10% bad prices
                return False, "Too many zero/negative prices", quality_score
    
    # Check for price gaps (more than 50% change between consecutive prices)
    if 'Close' in data.columns and len(data) > 1:
        price_changes = abs(data['Close'].pct_change())
        large_gaps = (price_changes > 0.5).sum()  # 50% change threshold
        
        if large_gaps > 0:
            quality_score -= min(30.0, large_gaps * 5)  # Each large gap costs 5 points
            issues.append(f"Contains {large_gaps} large price gaps (>50%)")
            
            if large_gaps > 3:  # More than 3 large gaps
                return False, "Too many large price gaps", quality_score
    
    # Volume checks
    if 'Volume' in data.columns:
        zero_volume = (data['Volume'] == 0).sum()
        zero_volume_pct = zero_volume / len(data)
        
        if zero_volume_pct > 0.3:  # More than 30% zero volume
            quality_score -= min(20.0, zero_volume_pct * 50)
            issues.append(f"High percentage of zero volume: {zero_volume_pct:.1%}")
    
    # Check for out-of-order timestamps
    if not data.index.is_monotonic_increasing:
        quality_score -= 15.0
        issues.append("Timestamps not in chronological order")
    
    # Final validation decision
    is_valid = quality_score >= 60.0  # Arbitrary threshold
    
    if is_valid and issues:
        message = f"Passed validation with {len(issues)} minor issues: {'; '.join(issues)}"
    elif is_valid:
        message = "Passed validation with no issues"
    else:
        message = f"Failed validation: {'; '.join(issues)}"
    
    return is_valid, message, quality_score

def add_data_metadata(data: pd.DataFrame, 
                     quality_score: float, 
                     validation_message: str) -> pd.DataFrame:
    """
    Add metadata to a DataFrame using the attrs property
    
    Args:
        data: DataFrame to add metadata to
        quality_score: Data quality score
        validation_message: Validation message
        
    Returns:
        DataFrame with metadata
    """
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Add metadata
    result.attrs['quality_score'] = float(quality_score)
    result.attrs['validation_message'] = validation_message
    result.attrs['validation_timestamp'] = datetime.now().isoformat()
    
    # Add data shape info
    result.attrs['data_shape'] = {
        'rows': len(result),
        'columns': len(result.columns),
        'start_date': result.index[0].isoformat() if len(result) > 0 and hasattr(result.index[0], 'isoformat') else str(result.index[0]),
        'end_date': result.index[-1].isoformat() if len(result) > 0 and hasattr(result.index[-1], 'isoformat') else str(result.index[-1])
    }
    
    return result

def save_to_cache(data: pd.DataFrame, 
                 key: str, 
                 start: Optional[str] = None, 
                 end: Optional[str] = None,
                 interval: Optional[str] = None,
                 cache_dir: str = "./data-cache") -> bool:
    """
    Save data to cache
    
    Args:
        data: DataFrame to cache
        key: Cache key (usually ticker symbol)
        start: Start date of the data
        end: End date of the data
        interval: Data interval
        cache_dir: Directory for cache files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache filename
        if start and end and interval:
            # For historical data with date range
            filename = f"{key}_{start}_{end}_{interval}.pkl"
        else:
            # For other data
            filename = f"{key}_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        filepath = os.path.join(cache_dir, filename)
        
        # Add cache metadata
        data.attrs['cache_timestamp'] = datetime.now().isoformat()
        if start:
            data.attrs['start_date'] = start
        if end:
            data.attrs['end_date'] = end
        if interval:
            data.attrs['interval'] = interval
        
        # Save to pickle file
        data.to_pickle(filepath)
        logger.info(f"Saved data to cache: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
        return False

def load_from_cache(key: str, 
                   start: Optional[str] = None, 
                   end: Optional[str] = None,
                   interval: Optional[str] = None,
                   cache_dir: str = "./data-cache",
                   max_age_days: float = 1.0) -> Optional[pd.DataFrame]:
    """
    Load data from cache if available and not expired
    
    Args:
        key: Cache key (usually ticker symbol)
        start: Start date of the data
        end: End date of the data
        interval: Data interval
        cache_dir: Directory for cache files
        max_age_days: Maximum age of cache in days
        
    Returns:
        DataFrame if cache hit, None otherwise
    """
    try:
        # Create filename pattern
        if start and end and interval:
            # For historical data with date range
            filename = f"{key}_{start}_{end}_{interval}.pkl"
        else:
            # Try to find newest file matching the key
            if not os.path.exists(cache_dir):
                return None
                
            matching_files = [f for f in os.listdir(cache_dir) if f.startswith(f"{key}_") and f.endswith('.pkl')]
            if not matching_files:
                return None
                
            # Get the newest file
            matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)), reverse=True)
            filename = matching_files[0]
        
        filepath = os.path.join(cache_dir, filename)
        
        # Check if file exists and is recent enough
        if not os.path.exists(filepath):
            logger.info(f"Cache miss: {filepath} does not exist")
            return None
        
        # Check file age
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))).total_seconds() / (24 * 3600)
        if file_age_days > max_age_days:
            logger.info(f"Cache expired: {filepath} is {file_age_days:.1f} days old (max: {max_age_days})")
            return None
        
        # Load the data
        data = pd.read_pickle(filepath)
        logger.info(f"Cache hit: {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None

def check_data_consistency(data1: pd.DataFrame, 
                          data2: pd.DataFrame, 
                          key_columns: List[str] = None,
                          tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Check consistency between two DataFrames
    
    Args:
        data1: First DataFrame
        data2: Second DataFrame
        key_columns: Columns to check for consistency
        tolerance: Tolerance for numerical differences
        
    Returns:
        Dictionary with consistency check results
    """
    if data1 is None or data2 is None:
        return {"consistent": False, "reason": "One or both DataFrames are None"}
    
    if data1.empty or data2.empty:
        return {"consistent": False, "reason": "One or both DataFrames are empty"}
    
    # If no specific columns provided, check all common columns
    if key_columns is None:
        key_columns = list(set(data1.columns).intersection(set(data2.columns)))
        if not key_columns:
            return {"consistent": False, "reason": "No common columns to compare"}
    
    # Check that all key columns exist in both DataFrames
    missing_in_1 = [col for col in key_columns if col not in data1.columns]
    missing_in_2 = [col for col in key_columns if col not in data2.columns]
    
    if missing_in_1 or missing_in_2:
        return {
            "consistent": False, 
            "reason": f"Missing columns: {', '.join(missing_in_1)} in first DataFrame, {', '.join(missing_in_2)} in second DataFrame"
        }
    
    # Check row counts
    row_diff = abs(len(data1) - len(data2))
    row_diff_pct = row_diff / max(len(data1), len(data2))
    
    if row_diff_pct > 0.1:  # More than 10% difference
        return {
            "consistent": False,
            "reason": f"Row count differs significantly: {len(data1)} vs {len(data2)} ({row_diff_pct:.1%} difference)"
        }
    
    # For each numerical column, check the consistency of values
    inconsistencies = []
    
    for col in key_columns:
        if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col]):
            # For numerical columns, compare averages
            avg1 = data1[col].mean()
            avg2 = data2[col].mean()
            
            if avg1 == 0 and avg2 == 0:
                continue  # Both are zero, considered consistent
            
            if avg1 != 0:
                rel_diff = abs(avg1 - avg2) / abs(avg1)
                
                if rel_diff > tolerance:
                    inconsistencies.append({
                        "column": col,
                        "avg1": float(avg1),
                        "avg2": float(avg2),
                        "relative_diff": float(rel_diff)
                    })
    
    # Overall consistency result
    result = {
        "consistent": len(inconsistencies) == 0,
        "row_count_1": len(data1),
        "row_count_2": len(data2),
        "row_difference": row_diff,
        "row_difference_pct": float(row_diff_pct),
        "column_inconsistencies": inconsistencies
    }
    
    if not result["consistent"]:
        result["reason"] = f"Found {len(inconsistencies)} column inconsistencies"
    
    return result

def merge_with_validation(main_data: pd.DataFrame, 
                         additional_data: pd.DataFrame, 
                         join_how: str = 'left',
                         validate_consistency: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge two DataFrames with validation
    
    Args:
        main_data: Primary DataFrame
        additional_data: Secondary DataFrame to merge
        join_how: How to join ('left', 'right', 'inner', 'outer')
        validate_consistency: Whether to validate data consistency
        
    Returns:
        Tuple of (merged_data, merge_info)
    """
    if main_data is None or main_data.empty:
        return additional_data, {"status": "main_data_empty"}
    
    if additional_data is None or additional_data.empty:
        return main_data, {"status": "additional_data_empty"}
    
    # Store original shapes
    main_shape = main_data.shape
    additional_shape = additional_data.shape
    
    # Make sure DataFrames have compatible indices
    if not isinstance(main_data.index, pd.DatetimeIndex):
        try:
            main_data.index = pd.to_datetime(main_data.index)
        except:
            pass
    
    if not isinstance(additional_data.index, pd.DatetimeIndex):
        try:
            additional_data.index = pd.to_datetime(additional_data.index)
        except:
            pass
    
    # Check for overlapping columns
    overlapping_columns = set(main_data.columns).intersection(set(additional_data.columns))
    
    # Validate consistency of overlapping columns
    consistency_check = None
    if validate_consistency and overlapping_columns:
        # Check only rows that exist in both DataFrames
        common_indices = main_data.index.intersection(additional_data.index)
        
        if len(common_indices) > 0:
            main_subset = main_data.loc[common_indices]
            additional_subset = additional_data.loc[common_indices]
            
            consistency_check = check_data_consistency(
                main_subset, 
                additional_subset,
                key_columns=list(overlapping_columns)
            )
    
    # Rename overlapping columns in additional_data to avoid conflicts
    additional_data_renamed = additional_data.copy()
    renamed_columns = {}
    
    for col in overlapping_columns:
        new_name = f"{col}_2"
        additional_data_renamed = additional_data_renamed.rename(columns={col: new_name})
        renamed_columns[col] = new_name
    
    # Merge the DataFrames
    merged_data = main_data.join(additional_data_renamed, how=join_how)
    
    # Prepare merge info
    merge_info = {
        "status": "success",
        "main_data_shape": main_shape,
        "additional_data_shape": additional_shape,
        "merged_data_shape": merged_data.shape,
        "renamed_columns": renamed_columns,
        "consistency_check": consistency_check
    }
    
    return merged_data, merge_info

def detect_data_anomalies(data: pd.DataFrame, 
                         check_columns: Optional[List[str]] = None,
                         z_score_threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect anomalies in data using statistical methods
    
    Args:
        data: DataFrame to check
        check_columns: Columns to check (defaults to all numeric columns)
        z_score_threshold: Z-score threshold for anomaly detection
        
    Returns:
        Dictionary with anomaly detection results
    """
    if data is None or data.empty:
        return {"status": "error", "message": "Empty data"}
    
    # If no columns specified, use all numeric columns
    if check_columns is None:
        check_columns = data.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to only include columns that exist and are numeric
        check_columns = [col for col in check_columns if col in data.columns and 
                          pd.api.types.is_numeric_dtype(data[col])]
    
    if not check_columns:
        return {"status": "error", "message": "No numeric columns to check"}
    
    anomalies = {}
    
    for col in check_columns:
        col_data = data[col].dropna()
        
        if len(col_data) < 5:  # Need enough data for meaningful statistics
            continue
        
        # Calculate mean and standard deviation
        mean = col_data.mean()
        std = col_data.std()
        
        if std == 0:  # All values are the same
            continue
        
        # Calculate z-scores
        z_scores = abs((col_data - mean) / std)
        
        # Find anomalies
        anomaly_indices = z_scores[z_scores > z_score_threshold].index.tolist()
        
        if anomaly_indices:
            # Get the actual anomalous values
            anomaly_values = col_data.loc[anomaly_indices]
            
            # Convert to list of (index, value, z_score) tuples
            anomaly_details = []
            for idx in anomaly_indices:
                value = float(col_data.loc[idx])
                z = float(z_scores.loc[idx])
                
                # Convert index to string if it's a datetime
                if isinstance(idx, pd.Timestamp):
                    idx_str = idx.isoformat()
                else:
                    idx_str = str(idx)
                
                anomaly_details.append({
                    "index": idx_str,
                    "value": value,
                    "z_score": z
                })
            
            anomalies[col] = {
                "count": len(anomaly_indices),
                "mean": float(mean),
                "std": float(std),
                "threshold": float(z_score_threshold),
                "details": anomaly_details
            }
    
    return {
        "status": "success",
        "anomaly_count": sum(anomalies[col]["count"] for col in anomalies),
        "columns_checked": len(check_columns),
        "columns_with_anomalies": len(anomalies),
        "anomalies": anomalies
    } 