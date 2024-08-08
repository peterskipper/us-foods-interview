"""App to run Sales Forecast Model"""

from pathlib import Path
import pickle

from fastapi import FastAPI, status
import pandas as pd
from pydantic import BaseModel

# pylint: disable=import-error
from log_utils import _instantiate_log
from model.train import INPUT_FEATURES

# pylint: enable=import-error

log = _instantiate_log(__file__)

app = FastAPI()
ROOT_DIR = Path(__file__).parent
with open(ROOT_DIR / "model.pkl", "rb") as model_path:
    model = pickle.load(model_path)


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class SalesPrediction(BaseModel):
    """Response model for sales forecast"""

    sales: float


class SalesModelInput(BaseModel):
    """Request model to get sales forecast"""

    date: str
    store: int
    item: int


@app.get(
    "/status",
    summary="Perform a health check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def _status() -> HealthCheck:
    """Check that app is running"""
    return HealthCheck(status="OK")


def _parse_datetime_features(date: str):
    """Parse datetime features for model"""
    date_ = pd.to_datetime(date)
    return {"month": date_.month, "day": date_.dayofweek, "year": date_.year}


def _process_inputs_for_model(date: str, store: int, item: int) -> pd.DataFrame:
    """Format features to pass to model"""
    feature_dict = {"store": store, "item": item}
    datetime_features = _parse_datetime_features(date)
    features = {**feature_dict, **datetime_features}
    log.info("Saw features: %s", features)
    return pd.DataFrame(features, columns=INPUT_FEATURES, index=[0])


@app.post(
    "/predict",
    summary="Forecast sales",
    response_model=SalesPrediction,
    status_code=status.HTTP_200_OK,
)
def predict(payload: SalesModelInput) -> SalesPrediction:
    """Forecast sales for a given date/store/item

    Args:
        payload: dict of SalesModelInput
    """
    model_inputs = _process_inputs_for_model(
        date=payload.date, store=payload.store, item=payload.item
    )
    sales = model.predict(model_inputs)[0]
    log.info("Made prediction: %f", sales)
    return SalesPrediction(sales=sales)
