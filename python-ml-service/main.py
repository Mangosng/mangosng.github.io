from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
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


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float):
    """
    Closed-form Ridge Regression: β̂ = (XᵀX + λI)⁻¹ Xᵀy
    Returns (coefficients, intercept).
    Assumes X is already standardized.
    """
    n, p = X.shape
    # Add intercept by centering y (since X is standardized, mean is 0)
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # β̂ = (XᵀX + λI)⁻¹ Xᵀy
    XtX = X.T @ X                        # (p x p)
    regularization = lam * np.eye(p)     # λI (p x p)
    Xty = X.T @ y_centered               # (p x 1)
    beta = np.linalg.solve(XtX + regularization, Xty)  # More stable than inverse

    intercept = y_mean  # Since X is centered (standardized), intercept = mean(y)
    return beta, intercept


def ridge_predict(X: np.ndarray, beta: np.ndarray, intercept: float):
    """Predict using Ridge coefficients."""
    return X @ beta + intercept


def cross_validate_lambda(X: np.ndarray, y: np.ndarray, lambdas: list, k: int = 5):
    """
    K-fold cross-validation to select the best λ.
    Returns the λ with the lowest average MSE across folds.
    """
    n = len(y)
    indices = np.arange(n)
    fold_size = n // k
    best_lam = lambdas[0]
    best_mse = float('inf')

    for lam in lambdas:
        mse_folds = []
        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            beta, intercept = ridge_closed_form(X_tr, y_tr, lam)
            y_pred = ridge_predict(X_val, beta, intercept)
            mse = np.mean((y_val - y_pred) ** 2)
            mse_folds.append(mse)

        avg_mse = np.mean(mse_folds)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lam = lam

    return best_lam


@app.get("/")
def read_root():
    return {"status": "ok", "service": "stock-predictor-ml"}

@app.post("/train-and-predict")
def train_and_predict(request: PredictionRequest):
    try:
        # 1. Prepare Data
        df = pd.DataFrame([vars(d) for d in request.training_data])

        feature_cols = ['close', 'volume', 'sma_20', 'volatility', 'fed_funds', 'cpi']

        X_raw = df[feature_cols].values
        y_raw = df['close'].shift(-request.days_ahead).dropna().values
        X_raw = X_raw[:-request.days_ahead]  # Align X with y length

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

        # 4. Cross-validate to find best λ, then fit with closed-form solution
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        best_lam = cross_validate_lambda(X_train_scaled, y_train, lambdas, k=5)
        beta_train, intercept_train = ridge_closed_form(X_train_scaled, y_train, best_lam)

        # 5. Metrics (computed on held-out test set)
        # Out-of-sample R² for price CHANGE predicted by non-close features
        change_feature_cols = ['volume', 'sma_20', 'volatility', 'fed_funds', 'cpi']
        X_change_raw = df[change_feature_cols].values[:-request.days_ahead]
        current_prices_all = df['close'].values[:len(y_raw)]
        y_change_raw = y_raw - current_prices_all  # actual price change

        X_change_train, X_change_test = X_change_raw[:split_idx], X_change_raw[split_idx:]
        y_change_train, y_change_test = y_change_raw[:split_idx], y_change_raw[split_idx:]

        change_scaler = StandardScaler()
        X_change_train_scaled = change_scaler.fit_transform(X_change_train)
        X_change_test_scaled = change_scaler.transform(X_change_test)

        best_lam_change = cross_validate_lambda(X_change_train_scaled, y_change_train, lambdas, k=5)
        beta_change, intercept_change = ridge_closed_form(X_change_train_scaled, y_change_train, best_lam_change)

        y_change_pred_test = ridge_predict(X_change_test_scaled, beta_change, intercept_change)
        ss_res = np.sum((y_change_test - y_change_pred_test) ** 2)
        ss_tot = np.sum((y_change_test - np.mean(y_change_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Hit Rate (Directional Accuracy) on test set
        y_pred_test = ridge_predict(X_test_scaled, beta_train, intercept_train)
        current_prices_test = df['close'].values[split_idx:split_idx + len(y_test)]
        actual_dir_test = np.sign(y_test - current_prices_test)
        pred_dir_test = np.sign(y_pred_test - current_prices_test)
        non_flat = actual_dir_test != 0
        if non_flat.sum() > 0:
            hit_rate = float(np.mean(actual_dir_test[non_flat] == pred_dir_test[non_flat]))
        else:
            hit_rate = 0.0

        # 6. Re-fit on ALL data for the final prediction
        X_all_scaled = scaler.fit_transform(X_raw)
        best_lam_final = cross_validate_lambda(X_all_scaled, y_raw, lambdas, k=5)
        beta_final, intercept_final = ridge_closed_form(X_all_scaled, y_raw, best_lam_final)

        # 7. Predict
        current_feats = np.array([[
            request.current_features['close'],
            request.current_features['volume'],
            request.current_features['sma_20'],
            request.current_features['volatility'],
            request.current_features['fed_funds'],
            request.current_features['cpi']
        ]])

        current_scaled = scaler.transform(current_feats)
        prediction = ridge_predict(current_scaled, beta_final, intercept_final)[0]

        # 8. Feature Importance (Absolute coefficients as percentage)
        importance = np.abs(beta_final)
        total_importance = importance.sum()
        importance_pct = (importance / total_importance) * 100 if total_importance > 0 else importance

        feature_importance_dict = {
            "Price": float(importance_pct[0]),
            "Volume": float(importance_pct[1]),
            "SMA_20": float(importance_pct[2]),
            "Volatility": float(importance_pct[3]),
            "Fed Funds": float(importance_pct[4]),
            "CPI": float(importance_pct[5])
        }

        return {
            "predicted_price": float(prediction),
            "r_squared": float(r_squared),
            "hit_rate": float(hit_rate),
            "feature_importance": feature_importance_dict,
            "lambda_selected": float(best_lam_final)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

