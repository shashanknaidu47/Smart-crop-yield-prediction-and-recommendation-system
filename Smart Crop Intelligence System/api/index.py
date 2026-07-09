from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import joblib
import pandas as pd

# Add root directory to sys path so we can import utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import recommend_crops, MODEL_DIR, CROP_REQUIREMENTS

app = FastAPI()

# Pre-load artefacts
artefacts = {}
try:
    if os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        artefacts["model"] = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "encoder.pkl")):
        artefacts["encoder"] = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
        artefacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "feature_cols.pkl")):
        artefacts["features"] = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
except Exception as e:
    print(f"Error loading models: {e}")


class PredictRequest(BaseModel):
    temp: float
    rainfall: float
    humidity: float
    solar: float
    N: float
    P: float
    K: float
    pH: float
    crop: str
    area: float
    pesticide: float
    season: str


class RecommendRequest(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    temp: float
    rainfall: float
    humidity: float


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": "model" in artefacts}


@app.post("/api/predict")
def predict_yield(req: PredictRequest):
    model_loaded = all(k in artefacts for k in ["model", "scaler", "features"])
    
    if model_loaded:
        try:
            features = artefacts["features"]
            encoder = artefacts.get("encoder")
            scaler = artefacts["scaler"]

            # Map crop to encoded value
            crop_encoded = 0
            if encoder and "Crop" in encoder.label_encoders:
                try:
                    crop_encoded = encoder.label_encoders["Crop"].transform([req.crop])[0]
                except Exception:
                    crop_encoded = 0

            # Season OHE columns
            season_map = {f"Season_{s}": 0 for s in ["Kharif", "Rabi", "Whole Year", "Zaid"]}
            season_key = f"Season_{req.season}"
            if season_key in season_map:
                season_map[season_key] = 1

            input_dict = {
                "Temperature": req.temp,
                "Rainfall": req.rainfall,
                "Humidity": req.humidity,
                "pH": req.pH,
                "N": req.N,
                "P": req.P,
                "K": req.K,
                "Pesticide_usage": req.pesticide,
                "Area": req.area,
                "Year": 2024.0,
                "Crop": crop_encoded,
                "solar_radiation": req.solar,
                "wind_speed": 3.0,
                "soil_moisture": 45.0,
                **season_map,
            }

            row = pd.DataFrame([{f: input_dict.get(f, 0.0) for f in features}])
            row_scaled = scaler.transform(row)
            
            # Predict
            model = artefacts["model"]
            pred_val = model.predict(row_scaled.values if hasattr(row_scaled, "values") else row_scaled)[0]
            pred_yield = float(pred_val)
            
        except Exception as e:
            print("Prediction error:", e)
            pred_yield = None
    else:
        pred_yield = None

    if pred_yield is None:
        # Fallback rule-of-thumb estimate
        score = (
            0.25 * min(req.N / 120, 1.5) +
            0.20 * min(req.P / 60, 1.5) +
            0.20 * min(req.K / 50, 1.5) +
            0.20 * min(req.rainfall / 1000, 2.0) +
            0.10 * (1 - abs(req.temp - 27) / 20) +
            0.05 * min(req.humidity / 70, 1.5)
        ) * (1 - req.pesticide / 200)
        pred_yield = round(score * 8, 2)

    production = pred_yield * req.area
    
    if pred_yield < 5:
        category = "Low 🟡"
    elif pred_yield < 10:
        category = "Medium 🟠"
    else:
        category = "High 🟢"

    return {
        "yield": round(pred_yield, 2),
        "production": round(production, 2),
        "category": category,
        "is_fallback": not model_loaded
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    recs = recommend_crops(
        N=req.N, P=req.P, K=req.K,
        temperature=req.temp, rainfall=req.rainfall, humidity=req.humidity,
        pH=req.pH, top_n=3
    )
    return {"recommendations": recs}

@app.get("/api/crops")
def get_crops():
    return {"crops": sorted(list(CROP_REQUIREMENTS.keys()))}
