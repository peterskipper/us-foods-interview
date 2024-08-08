"""Training pipeline for sales forecasts"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


INPUT_FEATURES = ["month", "day", "year", "store", "item"]
TARGET = "sales"
SEED = 42  # life, the universe, everything
BENCHMARK = 0.8  # we expect a model with at least this performance on a test set


def add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add datetime features to a training dataset.

    This modifies the training data in place.

    Args:
        data: training data for the sales forecast model

    Returns:
        Training data with datetime columns added
    """
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.dayofweek
    data["year"] = data["date"].dt.year
    return data


def split_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training data into train/test

    Args:
        data: Training data

    Returns:
         tuple of train_features, train_target, test_features, test_target
    """
    return train_test_split(
        data[INPUT_FEATURES], data[TARGET], test_size=0.2, random_state=SEED
    )


def train_model(
    train_x: pd.DataFrame, train_y: pd.Series, log: logging.Logger
) -> RandomForestRegressor:
    """
    Train the sales forecast model

    Args:
        train_x: Input features to train the model
        train_y: Target for model predictions (i.e. sales)

    Returns:
         trained RandomForestRegressor model
    """
    rf = RandomForestRegressor(random_state=SEED)
    log.info("Beginning model training...")
    rf.fit(train_x, train_y)
    log.info("Finished model training")
    return rf
