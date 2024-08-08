"""Script to train a new model"""

from pathlib import Path
import pickle

import pandas as pd

# pylint: disable=import-error
from log_utils import _instantiate_log
from model.train import (
    BENCHMARK,
    add_datetime_features,
    split_data,
    train_model,
)

# pylint: enable=import-error

log = _instantiate_log(__file__)


def main():
    """Model training pipeline for sales forecasts"""
    parent_dir = Path(__file__).resolve().parent
    training_path = parent_dir / "model/data/train.csv"
    if training_path.is_file():
        train = pd.read_csv(training_path)
    else:
        msg = (
            f"Training data is required at {training_path}. "
            "Move training data to that path and re-run"
        )
        raise RuntimeError(msg)
    train_ = add_datetime_features(train)
    log.info("Added datetime features")
    train_x, test_x, train_y, test_y = split_data(train_)
    log.info("Split data into train and test")
    model = train_model(train_x, train_y, log)
    r2 = model.score(test_x, test_y)
    if r2 < BENCHMARK:
        log.warning("Model test set R^2 performs below benchmark %f: %f", BENCHMARK, r2)
    else:
        log.info("Model test set R^2: %f", r2)
    with open(parent_dir / "model.pkl", "wb") as model_ckpt:
        pickle.dump(model, model_ckpt)
        log.info("Model persisted at %s", model_ckpt)
    log.info("Finished pipeline")


if __name__ == "__main__":
    main()
