from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="House Price Predictor")
@app.get("/")
def health_check():
    return {"status": "ok", "message": "House Price Predictor API is running"}


class HouseFeatures(BaseModel):
    property_type: str = Field(..., examples=["House"])
    city: str = Field(..., examples=["Lahore"])
    province_name: str = Field(..., examples=["Punjab"])
    latitude: float = Field(..., examples=[31.5204])
    longitude: float = Field(..., examples=[74.3587])
    baths: int = Field(..., ge=0, examples=[3])
    bedrooms: int = Field(..., ge=0, examples=[3])
    area_marla: float = Field(..., gt=0, examples=[8.0])
    
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, one level above src/
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

KNOWN_CITIES = {"Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad"}
KNOWN_PROPERTY_TYPES = {"House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", "Penthouse"}


@app.post("/predict")
def predict(features: HouseFeatures):
    warnings = []
    if features.city not in KNOWN_CITIES:
        warnings.append(f"City '{features.city}' was not seen during training; prediction may be unreliable.")
    if features.property_type not in KNOWN_PROPERTY_TYPES:
        warnings.append(f"Property type '{features.property_type}' was not seen during training; prediction may be unreliable.")

    input_df = pd.DataFrame([features.model_dump()])
    input_encoded = pd.get_dummies(
        input_df, columns=["property_type", "city", "province_name"]
    )
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    pred_log = model.predict(input_encoded)[0]
    pred_price = float(np.expm1(pred_log))

    response = {"predicted_price_pkr": round(pred_price, 0)}
    if warnings:
        response["warnings"] = warnings

    return response