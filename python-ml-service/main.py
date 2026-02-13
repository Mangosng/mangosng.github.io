from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
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

        # 2. Train/Test Split (80/20) for honest out-of-sample metrics
        split_idx = int(len(X_raw) * 0.8)
        X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
        y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

        # 3. Z-Score Standardization (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 4. Train Ridge Regression with cross-validated alpha on train set
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_train_scaled, y_train)

        # 5. Metrics (computed on held-out test set)
        # Out-of-sample R² for price CHANGE predicted by non-close features
        # This measures how well the contextual features explain price movement
        change_feature_cols = ['volume', 'sma_20', 'volatility', 'fed_funds', 'cpi']
        X_change_raw = df[change_feature_cols].values[:-request.days_ahead]
        current_prices_all = df['close'].values[:len(y_raw)]
        y_change_raw = y_raw - current_prices_all  # actual price change

        X_change_train, X_change_test = X_change_raw[:split_idx], X_change_raw[split_idx:]
        y_change_train, y_change_test = y_change_raw[:split_idx], y_change_raw[split_idx:]

        change_scaler = StandardScaler()
        X_change_train_scaled = change_scaler.fit_transform(X_change_train)
        X_change_test_scaled = change_scaler.transform(X_change_test)

        change_model = RidgeCV(alphas=alphas, cv=5)
        change_model.fit(X_change_train_scaled, y_change_train)
        r_squared = change_model.score(X_change_test_scaled, y_change_test)

        # Hit Rate (Directional Accuracy) on test set
        y_pred_test = model.predict(X_test_scaled)
        # current_prices_test[i] = df['close'][split_idx + i], the "today" price for each test sample
        current_prices_test = df['close'].values[split_idx:split_idx + len(y_test)]
        actual_dir_test = np.sign(y_test - current_prices_test)
        pred_dir_test = np.sign(y_pred_test - current_prices_test)
        # Only count non-flat days
        non_flat = actual_dir_test != 0
        if non_flat.sum() > 0:
            hit_rate = float(np.mean(actual_dir_test[non_flat] == pred_dir_test[non_flat]))
        else:
            hit_rate = 0.0

        # 6. Re-fit on ALL data for the final prediction (use all available info)
        X_all_scaled = scaler.fit_transform(X_raw)
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_all_scaled, y_raw)

        # 7. Predict
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

        # 8. Feature Importance (Absolute coefficients as percentage)
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
            "hit_rate": float(hit_rate),
            "feature_importance": feature_importance_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
