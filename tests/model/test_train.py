"""Tests for training pipeline"""

import pandas as pd

from app.model.train import add_datetime_features


def test_universe_is_functioning_correctly():
    """Just making sure"""
    assert True


def test_add_datetime_features():
    """Test datetime features added correctly"""
    data = pd.DataFrame(
        {"date": ["2014-09-08", "2014-10-29"], "store": [1, 2], "item": [3, 4]}
    )
    assert set(["month", "day", "year"]) - set(data.columns) == set(
        ["month", "day", "year"]
    ), "Features added before processing"
    processed = add_datetime_features(data)
    assert (
        set(["month", "day", "year"]) - set(processed.columns) == set()
    ), "Features not added"
