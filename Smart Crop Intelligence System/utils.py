"""
utils.py
--------
Shared constants, helper functions, and utilities used across the project.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "saved")
VIZ_DIR    = os.path.join(BASE_DIR, "visualization", "outputs")

for _d in (MODEL_DIR, VIZ_DIR):
    os.makedirs(_d, exist_ok=True)

DATASET_PATH  = os.path.join(DATA_DIR, "crop_yield_dataset.csv")
FALLBACK_PATH = os.path.join(DATA_DIR, "crop_dataset.csv")

# ─────────────────────────────────────────────
# DOMAIN CONSTANTS
# ─────────────────────────────────────────────
CROP_NAMES = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean",
    "Groundnut", "Barley", "Chickpea", "Mustard", "Potato",
    "Tomato", "Onion", "Jowar", "Bajra", "Sunflower",
    "Turmeric", "Ginger", "Tea", "Coffee",
]

SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]

NUMERIC_FEATURES = [
    "Temperature", "Rainfall", "Humidity", "pH",
    "N", "P", "K", "Pesticide_usage",
    "Area", "Year",
]

CATEGORICAL_FEATURES = ["Crop", "State", "District", "Season"]

TARGET_REGRESSION     = "Yield"
TARGET_CLASSIFICATION = "Crop"

# Soil / weather ranges for the crop recommendation engine
CROP_REQUIREMENTS = {
    "Rice":         dict(N=(80,120), P=(40,60),  K=(40,60),  temp=(22,32), rain=(1000,2500), hum=(70,90),  pH=(5.5,7.0)),
    "Wheat":        dict(N=(60,100), P=(40,60),  K=(40,60),  temp=(10,25), rain=(400,1000),  hum=(40,70),  pH=(6.0,7.5)),
    "Maize":        dict(N=(80,120), P=(40,70),  K=(40,70),  temp=(20,30), rain=(600,1200),  hum=(50,80),  pH=(5.8,7.0)),
    "Cotton":       dict(N=(60,100), P=(25,50),  K=(25,50),  temp=(21,35), rain=(500,1200),  hum=(40,75),  pH=(5.8,8.0)),
    "Sugarcane":    dict(N=(100,160),P=(50,80),  K=(70,110), temp=(24,35), rain=(1000,2000), hum=(65,90),  pH=(6.0,7.5)),
    "Soybean":      dict(N=(20,50),  P=(40,60),  K=(40,60),  temp=(20,30), rain=(600,1200),  hum=(50,80),  pH=(6.0,7.0)),
    "Groundnut":    dict(N=(15,40),  P=(30,50),  K=(60,90),  temp=(25,35), rain=(400,900),   hum=(50,75),  pH=(5.5,7.0)),
    "Barley":       dict(N=(50,90),  P=(35,55),  K=(35,55),  temp=(12,25), rain=(300,800),   hum=(35,65),  pH=(6.0,7.5)),
    "Chickpea":     dict(N=(15,40),  P=(40,60),  K=(20,40),  temp=(15,30), rain=(300,700),   hum=(40,70),  pH=(6.0,7.5)),
    "Mustard":      dict(N=(50,90),  P=(30,50),  K=(30,50),  temp=(10,25), rain=(250,700),   hum=(40,70),  pH=(5.8,7.5)),
    "Potato":       dict(N=(80,130), P=(60,100), K=(100,150),temp=(15,25), rain=(500,1000),  hum=(65,85),  pH=(5.0,6.5)),
    "Tomato":       dict(N=(80,120), P=(60,90),  K=(80,120), temp=(20,30), rain=(600,1200),  hum=(60,80),  pH=(6.0,7.0)),
    "Onion":        dict(N=(60,100), P=(40,70),  K=(60,100), temp=(15,28), rain=(300,700),   hum=(50,75),  pH=(6.0,7.5)),
    "Jowar":        dict(N=(50,90),  P=(25,50),  K=(25,50),  temp=(25,35), rain=(400,900),   hum=(40,70),  pH=(6.0,7.5)),
    "Bajra":        dict(N=(40,80),  P=(20,40),  K=(20,40),  temp=(25,38), rain=(200,600),   hum=(30,65),  pH=(5.5,7.0)),
    "Sunflower":    dict(N=(60,100), P=(40,60),  K=(40,60),  temp=(20,32), rain=(400,800),   hum=(40,70),  pH=(5.8,7.5)),
    "Turmeric":     dict(N=(60,100), P=(40,60),  K=(100,150),temp=(20,35), rain=(1200,2500), hum=(65,90),  pH=(5.5,7.0)),
    "Ginger":       dict(N=(60,100), P=(40,60),  K=(80,120), temp=(20,30), rain=(1500,3000), hum=(70,90),  pH=(5.5,6.5)),
    "Tea":          dict(N=(80,140), P=(20,40),  K=(40,80),  temp=(15,28), rain=(1500,3000), hum=(75,95),  pH=(4.5,5.5)),
    "Coffee":       dict(N=(80,140), P=(40,70),  K=(80,130), temp=(18,28), rain=(1500,2500), hum=(70,90),  pH=(5.5,6.5)),
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """Load the main dataset; fall back to the smaller CSV if needed."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Loaded dataset: {df.shape}  ← {DATASET_PATH}")
    elif os.path.exists(FALLBACK_PATH):
        df = pd.read_csv(FALLBACK_PATH)
        logger.info(f"Loaded fallback dataset: {df.shape}  ← {FALLBACK_PATH}")
    else:
        raise FileNotFoundError(
            "No dataset found. Run: python data/generate_dataset.py"
        )
    return df


def save_model(model, filename: str) -> str:
    """Persist a model with joblib and return its path."""
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")
    return path


def load_model(filename: str):
    """Load a persisted model by filename."""
    path = os.path.join(MODEL_DIR, filename)
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_yield_category(yield_val: float) -> str:
    """Bin a continuous yield value into Low / Medium / High."""
    if yield_val < 5:
        return "Low"
    elif yield_val < 10:
        return "Medium"
    else:
        return "High"


def recommend_crops(N: float, P: float, K: float, temperature: float,
                    rainfall: float, humidity: float, pH: float,
                    top_n: int = 3) -> list[dict]:
    """
    Rule-based crop suitability scoring.

    Each nutrient / weather parameter is scored 0–1 based on how well the
    supplied value falls within the crop's ideal range.  The overall score
    is the mean of all individual scores.  Returns the `top_n` crops with
    the highest scores.
    """
    scores = {}
    params = {
        "N": N, "P": P, "K": K,
        "temp": temperature, "rain": rainfall,
        "hum": humidity, "pH": pH,
    }

    def _score_param(val: float, lo: float, hi: float) -> float:
        if lo <= val <= hi:
            # Linearly highest at the midpoint
            mid = (lo + hi) / 2
            return 1.0 - abs(val - mid) / ((hi - lo) / 2 + 1e-9)
        else:
            dist = min(abs(val - lo), abs(val - hi))
            rng  = max(hi - lo, 1)
            return max(0.0, 1.0 - dist / rng)

    for crop, req in CROP_REQUIREMENTS.items():
        s = []
        for key in ("N", "P", "K", "temp", "rain", "hum", "pH"):
            lo, hi = req[key]
            s.append(_score_param(params[key], lo, hi))
        scores[crop] = round(float(np.mean(s)), 4)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"crop": c, "score": s, "suitability": f"{s*100:.1f}%"}
            for c, s in ranked]


def get_feature_names(df: pd.DataFrame, drop_cols: list = None) -> list:
    """Return usable feature column names from a DataFrame."""
    drop = set(drop_cols or [])
    return [c for c in df.columns if c not in drop]
