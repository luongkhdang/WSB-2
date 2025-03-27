import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from src.main_utilities.data_processor import format_stock_data_for_analysis


def create_mock_data(length=100, with_nan=False):
    """Create mock stock data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(length)]
    dates.sort()

    data = {
        'Open': np.linspace(100, 120, length),
        'High': np.linspace(105, 125, length),
        'Low': np.linspace(95, 115, length),
        'Close': np.linspace(102, 122, length),
        'Volume': np.random.randint(1000, 10000, length)
    }

    # Add some noise to make it more realistic
    for key in ['Open', 'High', 'Low', 'Close']:
        data[key] = data[key] + np.random.normal(0, 1, length)

    # Ensure High is always highest, Low is always lowest
    for i in range(length):
        maxval = max(data['Open'][i], data['Close'][i])
        minval = min(data['Open'][i], data['Close'][i])
        data['High'][i] = max(data['High'][i], maxval + 0.5)
        data['Low'][i] = min(data['Low'][i], minval - 0.5)

    # Add NaN values if requested
    if with_nan:
        # Add some NaN values to test handling
        nan_indices = np.random.choice(
            length, int(length * 0.1), replace=False)
        for idx in nan_indices:
            col = np.random.choice(['Open', 'High', 'Low', 'Close'], 1)[0]
            data[col][idx] = np.nan

    return pd.DataFrame(data, index=dates)


def test_format_stock_data_basic():
    """Test basic formatting of stock data"""
    data = create_mock_data(length=100)
    result = format_stock_data_for_analysis(data, 'TEST', '2023-03-24')

    # Check that required fields are present
    assert 'ticker' in result
    assert 'date' in result
    assert 'current_price' in result
    assert 'atr' in result
    assert 'atr_percent' in result
    assert 'rsi' in result
    assert 'macd' in result

    # Check for correct ticker and non-zero values
    assert result['ticker'] == 'TEST'
    assert result['current_price'] > 0
    assert result['atr'] > 0
    assert result['atr_percent'] > 0
    assert 0 <= result['rsi'] <= 100  # RSI is always between 0 and 100

    # Check that MACD is correctly calculated
    assert result['macd_available'] == True
    assert 'macd_error' not in result
    assert result['macd'] is not None


def test_format_stock_data_with_nans():
    """Test formatting stock data with NaN values"""
    data = create_mock_data(length=100, with_nan=True)
    result = format_stock_data_for_analysis(data, 'TEST', '2023-03-24')

    # Even with NaNs, we should still get valid results
    assert result['current_price'] > 0
    assert result['atr'] > 0

    # MACD should still be calculated
    assert result['macd'] is not None


def test_format_stock_data_insufficient_data():
    """Test formatting stock data with insufficient data points"""
    # Create data with only 10 points (insufficient for many indicators)
    data = create_mock_data(length=10)
    result = format_stock_data_for_analysis(data, 'TEST', '2023-03-24')

    # Check that the function handles insufficient data gracefully
    assert result['ticker'] == 'TEST'
    assert result['current_price'] > 0

    # MACD should be unavailable due to insufficient data
    assert result.get('macd_available', True) == False
    assert 'macd_error' in result


def test_atr_calculation():
    """Test ATR calculation specifically"""
    data = create_mock_data(length=50)
    result = format_stock_data_for_analysis(data, 'TEST', '2023-03-24')

    # ATR should be reasonable (typically 0.5-5% for most stocks)
    assert 0.1 <= result['atr_percent'] <= 10.0

    # Let's create a high-volatility dataset
    high_vol_data = data.copy()
    high_vol_data['High'] = high_vol_data['High'] * 1.5
    high_vol_data['Low'] = high_vol_data['Low'] * 0.5

    high_vol_result = format_stock_data_for_analysis(
        high_vol_data, 'TEST', '2023-03-24')

    # ATR should be higher but still reasonable (capped at 10%)
    assert result['atr_percent'] < high_vol_result['atr_percent']
    assert high_vol_result['atr_percent'] <= 10.0


def test_empty_data_handling():
    """Test handling of empty data"""
    # Create empty dataframe
    empty_data = pd.DataFrame()
    result = format_stock_data_for_analysis(empty_data, 'TEST', '2023-03-24')

    # Should return error info
    assert 'error' in result
    assert result['ticker'] == 'TEST'
    assert result['date'] == '2023-03-24'


def test_missing_columns():
    """Test handling of dataframe with missing required columns"""
    # Create data with missing columns
    dates = [datetime.now() - timedelta(days=i) for i in range(50)]
    dates.sort()

    incomplete_data = pd.DataFrame({
        'Open': np.linspace(100, 120, 50),
        'Close': np.linspace(102, 122, 50),
        # Missing High and Low columns
    }, index=dates)

    result = format_stock_data_for_analysis(
        incomplete_data, 'TEST', '2023-03-24')

    # Should note missing columns
    assert 'error' in result
    assert 'Missing required data columns' in result['error']
