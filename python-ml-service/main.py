from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class TrainingData(BaseModel):
    date: str
    close: float
    volume: float
    sma_20: float
    volatility: float
    fed_funds: float
    cpi: float

class PredictionRequest(BaseModel):
    training_data: List[TrainingData]
    current_features: Dict[str, float]
    days_ahead: int

@app.get("/")
def read_root():
    return {"status": "ok", "service": "stock-predictor-ml"}

@app.post("/train-and-predict")
def train_and_predict(request: PredictionRequest):
    try:
        # 1. Prepare Data
        df = pd.DataFrame([vars(d) for d in request.training_data])
        
        # Features and Target
        # We need to construct X and y ensuring alignment
        # y is the close price 'days_ahead' in the future
        
        feature_cols = ['close', 'volume', 'sma_20', 'volatility', 'fed_funds', 'cpi']
        
        X_raw = df[feature_cols].values
        y_raw = df['close'].shift(-request.days_ahead).dropna().values
        X_raw = X_raw[:-request.days_ahead] # Align X with y length
        
        if len(X_raw) < 50:
            raise HTTPException(status_code=400, detail="Insufficient training data")

        # 2. Z-Score Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # 3. Train Ridge Regression
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_raw)

        # 4. Predict
        current_feats = [
            request.current_features['close'],
            request.current_features['volume'],
            request.current_features['sma_20'],
            request.current_features['volatility'],
            request.current_features['fed_funds'],
            request.current_features['cpi']
        ]
        
        current_scaled = scaler.transform([current_feats])
        prediction = model.predict(current_scaled)[0]

        # 5. Metrics & Feature Importance
        # R-squared score
        r_squared = model.score(X_scaled, y_raw)
        
        # Feature Importance (Absolute coefficients as percentage)
        importance = np.abs(model.coef_)
        total_importance = importance.sum()
        importance_pct = (importance / total_importance) * 100 if total_importance > 0 else importance

        feature_importance_dict = {
            "Price": importance_pct[0],
            "Volume": importance_pct[1],
            "SMA_20": importance_pct[2],
            "Volatility": importance_pct[3],
            "Fed Funds": importance_pct[4],
            "CPI": importance_pct[5]
        }

        return {
            "predicted_price": float(prediction),
            "r_squared": float(r_squared),
            "feature_importance": feature_importance_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
