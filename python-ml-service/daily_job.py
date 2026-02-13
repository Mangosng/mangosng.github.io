import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from supabase import create_client, Client
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Supabase Setup
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

supabase: Client = create_client(url, key)

# Constants
TABLE_NAME = "predictions"
BATCH_DELAY = 12 # seconds, to stay under 5 calls/min limit of free Polygon tier if used, though we use yfinance here which is more lenient but good to be safe/polite
# Actually yfinance uses Yahoo API which has rate limits too. 12s is very safe.

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers (replace . with - for yahoo, e.g. BF.B -> BF-B)
        tickers = [t.replace('.', '-') for t in tickers]
        logging.info(f"Fetched {len(tickers)} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        # Fallback to a small list for testing if wiki fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

def fetch_history(ticker):
    """Fetch last ~100 days of data for a ticker."""
    try:
        # Fetch 6 months to be safe for 20-day MA and lags
        hist = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if hist.empty:
            return None
        
        # Flatten MultiIndex columns if present (yfinance update)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        hist = hist.rename(columns={
            "Close": "close", 
            "Volume": "volume", 
            "Open": "open", 
            "High": "high", 
            "Low": "low"
        })
        
        # Ensure index is datetime
        hist.index = pd.to_datetime(hist.index)
        return hist
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def compute_features(df):
    """Compute features needed for the model."""
    df = df.copy()
    
    # 1. Technical Indicators
    # SMA 20
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Volatility (20-day std dev of log returns)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(window=20).std()
    
    # Momentum (Lagged Returns)
    df['ret_1d'] = df['log_ret'] # Current day's return is the input for "1D Momentum"
    df['ret_5d'] = df['log_ret'].rolling(window=5).sum()
    df['ret_10d'] = df['log_ret'].rolling(window=10).sum()
    df['ret_20d'] = df['log_ret'].rolling(window=20).sum()
    
    # Macro data (Fed Funds, CPI) - Fetching this daily is hard without API.
    # For now, we will use static/placeholder values or fetch from FRED if key exists.
    # To keep it robust for this batch job, we'll use recent values.
    # In a real prod env, we'd fetch these.
    # FEATURE HACK: Use fixed recent values to avoid FRED API dependency failure
    df['fed_funds'] = 5.33 
    df['cpi'] = 308.0

    return df

def predict_next_day(ticker, df):
    """
    Predict the log return for the NEXT day.
    Uses the coefficients from main.py (hardcoded or loaded).
    For simplicity/robustness in this script, we'll use the coefficients we derived/know.
    Ideally, this script imports the trained model.
    """
    # Load model?
    # Importing 'main' might trigger FastAPI app run.
    # We should extract the model training logic or save the model.
    # For now, let's TRAIN the model on the fly for each stock? 
    # No, that's too slow and fits on single stock data (overfitting).
    
    # We need the "Global" model.
    # Since we don't have a model registry, we'll use the Strategy:
    # "Train on the fly on the specific stock data?"
    # No, the main.py trains on the data passed in the request.
    # The main.py logic is: User sends data -> Train -> Predict.
    # So yes, we TRAIN on the specific stock's history!
    # Our API is designed to be "Train-and-Predict".
    
    # So we should call our own API?
    # Or replicate the logic here.
    # Replicating logic is safer for a batch job (no network dependency on own API).
    
    # Logic from main.py:
    # 1. Train/Test split (80/20)
    # 2. Standardize
    # 3. RidgeCV
    # 4. Predict
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    
    # Prepare Data
    # Target: Log return next day
    # Features: lags, volume, volatility...
    
    df['target'] = np.log(df['close'].shift(-1) / df['close']) # 1 day ahead target
    
    features = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'volume', 'volatility', 'fed_funds', 'cpi']
    
    data = df.dropna().copy()
    
    if len(data) < 50:
        return None
        
    X = data[features].values
    y = data['target'].values
    
    # We want to predict for "Tomorrow".
    # So we train on all available history (up to today-1 target).
    # The last row of 'df' has features for 'Today', but target is NaN (Tomorrow unknown).
    
    # Training Data: All rows where we have target
    X_train = X
    y_train = y
    
    # Prediction Input: The very last row (Today)
    # Re-compute features for the last row (which might have been dropped due to NaN target)
    last_row = df.iloc[-1]
    # Check if last_row has valid features (it should, as lags look back)
    # Only target is missing.
    if np.isnan(last_row['ret_20d']):
         return None
         
    current_features = last_row[features].values.reshape(1, -1)
    
    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    model.fit(X_scaled, y_train)
    
    # Predict
    curr_scaled = scaler.transform(current_features)
    pred_log_ret = model.predict(curr_scaled)[0]
    
    return pred_log_ret

def run():
    tickers = get_sp500_tickers()
    logging.info(f"Starting batch prediction for {len(tickers)} stocks...")
    
    today = datetime.now().strftime('%Y-%m-%d')
    predictions_made = 0
    
    for ticker in tickers:
        try:
            # Rate Limit
            time.sleep(BATCH_DELAY)
            
            # Fetch
            df = fetch_history(ticker)
            if df is None:
                continue
                
            # Process
            df = compute_features(df)
            
            # Predict
            pred_log_ret = predict_next_day(ticker, df)
            if pred_log_ret is None:
                continue
                
            # Store
            # Predicted Direction: +1 if > 0, -1 if < 0
            direction = 1 if pred_log_ret > 0 else -1
            
            data = {
                "ticker": ticker,
                "predicted_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'), # Next day
                # Actually, if we run after market close, we confirm the close of TODAY.
                # And we predict for TOMORROW.
                "predicted_log_return": float(pred_log_ret),
                "predicted_direction": direction,
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into Supabase
            supabase.table(TABLE_NAME).insert(data).execute()
            predictions_made += 1
            logging.info(f"Predicted {ticker}: {pred_log_ret:.5f}")
            
        except Exception as e:
            logging.error(f"Failed to process {ticker}: {e}")
            
    logging.info(f"Batch complete. Made {predictions_made} predictions.")

if __name__ == "__main__":
    run()
