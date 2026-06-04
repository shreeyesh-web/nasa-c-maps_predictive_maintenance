from fastapi   import FastAPI, HTTPException
from pydantic  import BaseModel
import pickle
import numpy as np
import uvicorn
import os

# Configuration for Database (Environment variables)
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "nasa")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI(
    title       = "NASA Predictive Maintenance API",
    description = "Predicts engine failure using XGBoost on NASA CMAPSS data",
    version     = "1.0"
)

# Load model once when server starts
try:
    xgb_model  = pickle.load(open("models/xgb_model.pkl",    "rb"))
    xgb_scaler = pickle.load(open("models/xgb_scaler.pkl",   "rb"))
    xgb_feats  = pickle.load(open("models/xgb_features.pkl", "rb"))
    print("Models loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Model files not found. Ensure the 'models/' folder contains your .pkl files. {e}")

# Fixed Schema: One field per line
class SensorInput(BaseModel):
    unit_id:  int
    cycles:   int
    life_pct: float
    s2:  float
    s3:  float
    s4:  float
    s7:  float
    s8:  float
    s9:  float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s17: float
    s20: float
    s21: float
    s11_mean10:    float = 0.0
    s4_mean10:     float = 0.0
    s12_mean10:    float = 0.0
    s11_std10:     float = 0.0
    s4_std10:      float = 0.0
    s11_delta5:    float = 0.0
    s4_delta5:     float = 0.0
    s12_delta5:    float = 0.0
    s11_drift:     float = 0.0
    s4_drift:      float = 0.0
    s4_x_s11:      float = 0.0
    s12_minus_s11: float = 0.0

@app.get("/")
def home():
    return {
        "status":  "running",
        "project": "NASA Predictive Maintenance",
        "model":   "XGBoost",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": xgb_model is not None}

@app.post("/predict")
def predict(data: SensorInput):
    try:
        # Extract features in the exact order the model expects
        row = {f: getattr(data, f, 0.0) for f in xgb_feats}
        X   = np.array([[row[f] for f in xgb_feats]])
        
        # Scale and Predict
        X_s = xgb_scaler.transform(X)
        pred  = int(xgb_model.predict(X_s)[0])
        proba = float(xgb_model.predict_proba(X_s)[0][pred])

        if pred == 1:
            label   = "ANOMALY"
            message = "Engine approaching failure. Schedule maintenance immediately."
        else:
            label   = "NORMAL"
            message = "Engine operating within normal parameters."

        return {
            "unit_id":    data.unit_id,
            "cycles":     data.cycles,
            "prediction": pred,
            "label":      label,
            "confidence": round(proba, 4),
            "message":    message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    # Ensure this string matches your filename (api.py -> "api:app")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)