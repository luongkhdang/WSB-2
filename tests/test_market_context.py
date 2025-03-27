import os
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.main_utilities.file_operations import get_historical_market_context


class MockYFinanceClient:
    """Mock YFinance client for testing"""

    def get_historical_data(self, symbol, start=None, end=None, interval="1d"):
        """Return mock historical data for testing"""
        dates = pd.date_range(start=start, end=end, freq='D')

        if symbol == "SPY":
            data = {
                'Open': np.linspace(400, 410, len(dates)),
                'High': np.linspace(405, 415, len(dates)),
                'Low': np.linspace(395, 405, len(dates)),
                'Close': np.linspace(402, 412, len(dates)),
                'Volume': np.random.randint(10000, 100000, len(dates))
            }
        elif symbol == "^VIX":
            data = {
                'Open': np.linspace(18, 22, len(dates)),
                'High': np.linspace(19, 23, len(dates)),
                'Low': np.linspace(17, 21, len(dates)),
                'Close': np.linspace(18.5, 22.5, len(dates)),
                'Volume': np.random.randint(5000, 50000, len(dates))
            }
        else:
            # Return empty dataframe for unsupported symbols
            return pd.DataFrame()

        return pd.DataFrame(data, index=dates)


@pytest.fixture
def temp_market_dir(tmp_path):
    """Create a temporary directory for market context data"""
    return tmp_path / "market_data"


def test_get_market_context_with_file(temp_market_dir):
    """Test getting market context from existing file"""
    # Create directory
    temp_market_dir.mkdir(exist_ok=True)

    # Create a date to test
    test_date = '2023-03-24'

    # Create and save a test context file
    test_context = {
        "date": test_date,
        "spy_trend": "bullish",
        "market_trend": "bullish",
        "market_trend_score": 65,
        "spy_change_percent": 1.25,
        "vix_level": 19.5,
        "vix_assessment": "VIX is at 19.5 indicating moderate volatility",
        "risk_adjustment": "standard",
        "market_indices": {
            "SPY": {"trend": "bullish", "change": 1.25}
        },
        "data_source": "test"
    }

    # Create date-specific file
    with open(temp_market_dir / f"market_context_{test_date}.json", 'w') as f:
        json.dump(test_context, f)

    # Call function
    result = get_historical_market_context(temp_market_dir, test_date)

    # Check result
    assert result['date'] == test_date
    assert result['spy_trend'] == 'bullish'
    assert result['market_trend'] == 'bullish'
    assert result['spy_change_percent'] == 1.25
    assert result['data_source'] == 'test'


def test_get_market_context_with_yfinance(temp_market_dir):
    """Test getting market context using YFinance client when no file exists"""
    # Create directory
    temp_market_dir.mkdir(exist_ok=True)

    # Create a date to test
    test_date = '2023-03-24'

    # Call function with mock YFinance client
    mock_client = MockYFinanceClient()
    result = get_historical_market_context(
        temp_market_dir, test_date, mock_client)

    # Check that a real result was returned, not a default
    assert result['date'] == test_date
    assert 'spy_trend' in result
    assert 'vix_level' in result
    assert result.get('data_source') == 'yfinance_realtime'

    # Verify file was created
    assert (temp_market_dir / f"market_context_{test_date}.json").exists()


def test_get_market_context_default(temp_market_dir):
    """Test getting default market context when no file exists and no YFinance client"""
    # Create directory
    temp_market_dir.mkdir(exist_ok=True)

    # Create a date to test
    test_date = '2023-03-24'

    # Call function without YFinance client
    result = get_historical_market_context(temp_market_dir, test_date)

    # Check that default was returned
    assert result['date'] == test_date
    assert result['spy_trend'] == 'neutral'
    assert result.get('data_source') == 'default'
    assert 'note' in result

    # Verify file was created
    assert (temp_market_dir / f"market_context_{test_date}.json").exists()


def test_get_market_context_month_file(temp_market_dir):
    """Test getting market context from month file"""
    # Create directory
    temp_market_dir.mkdir(exist_ok=True)

    # Create dates to test
    test_date1 = '2023-03-24'
    test_date2 = '2023-03-25'
    month = '2023-03'

    # Create a monthly context file with multiple dates
    month_data = {
        test_date1: {
            "date": test_date1,
            "spy_trend": "bullish",
            "market_trend": "bullish",
            "market_trend_score": 65,
            "data_source": "month_file"
        },
        test_date2: {
            "date": test_date2,
            "spy_trend": "bearish",
            "market_trend": "bearish",
            "market_trend_score": 35,
            "data_source": "month_file"
        }
    }

    # Create month file
    with open(temp_market_dir / f"market_context_{month}.json", 'w') as f:
        json.dump(month_data, f)

    # Call function for first date
    result1 = get_historical_market_context(temp_market_dir, test_date1)

    # Check result
    assert result1['date'] == test_date1
    assert result1['spy_trend'] == 'bullish'
    assert result1['data_source'] == 'month_file'

    # Call function for second date
    result2 = get_historical_market_context(temp_market_dir, test_date2)

    # Check result
    assert result2['date'] == test_date2
    assert result2['spy_trend'] == 'bearish'
    assert result2['data_source'] == 'month_file'


def test_get_market_context_error_handling(temp_market_dir):
    """Test error handling in get_historical_market_context"""
    # Try to get context for a directory that doesn't exist
    non_existent_dir = Path("not_a_real_directory")
    result = get_historical_market_context(non_existent_dir, '2023-03-24')

    # Since the implementation creates the directory and returns default values,
    # check that we got a default result
    assert result['spy_trend'] == 'neutral'
    assert result.get('data_source') == 'default'
    assert 'note' in result
