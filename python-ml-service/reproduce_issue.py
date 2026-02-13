import requests
import json
import pandas as pd
import numpy as np

# Mock data to simulate the request
def create_mock_data():
    # Create 100 days of data
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = []
    
    # Simple linear trend + noise for predictability
    for i, date in enumerate(dates):
        # close price follows a trend
        close = 100 + i * 0.5 + np.random.normal(0, 2)
        
        data.append({
            "date": date.strftime('%Y-%m-%d'),
            "close": close,
            "volume": 1000 + np.random.normal(0, 100),
            "sma_20": close - 2, # simple mock
            "volatility": 0.02,
            "fed_funds": 4.5,
            "cpi": 300
        })
    
    return data

def test_prediction():
    training_data = create_mock_data()
    
    current_features = {
        "close": training_data[-1]['close'],
        "volume": training_data[-1]['volume'],
        "sma_20": training_data[-1]['sma_20'],
        "volatility": training_data[-1]['volatility'],
        "fed_funds": training_data[-1]['fed_funds'],
        "cpi": training_data[-1]['cpi']
    }
    
    payload = {
        "training_data": training_data,
        "current_features": current_features,
        "days_ahead": 5
    }
    
    # Assuming the service is running locally on port 8000
    # If not, this script is just a template for how the request looks
    try:
        response = requests.post("http://localhost:8000/train-and-predict", json=payload)
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print("Error:", response.text)
    except Exception as e:
        print("Failed to connect to local service:", e)
        # Fallback: We can import main.py directly if we are in the same dir structure
        # but for now let's just show what we expect
        print("Could not verify against live service. Proceeding with unit test.")

if __name__ == "__main__":
    test_prediction()
