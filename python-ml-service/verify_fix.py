"""Quick verification that metrics are in valid ranges after the fix."""
import pandas as pd
from main import TrainingData, PredictionRequest, train_and_predict

# Scenario 1: Perfectly linear uptrend
dates = pd.date_range(start='2023-01-01', periods=100)
td = [TrainingData(date=d.strftime('%Y-%m-%d'), close=100+i, volume=1000,
      sma_20=100+i-2, volatility=0.01, fed_funds=4.5, cpi=300)
      for i, d in enumerate(dates)]
cf = {'close': 200, 'volume': 1000, 'sma_20': 198,
      'volatility': 0.01, 'fed_funds': 4.5, 'cpi': 300}
r = train_and_predict(PredictionRequest(training_data=td, current_features=cf, days_ahead=1))

print(f"=== Linear Uptrend (days_ahead=1) ===")
print(f"  R²:       {r['r_squared']:.4f}  (should be reasonable, ~0.9+ for perfect trend)")
print(f"  Hit Rate: {r['hit_rate']:.4f}  (MUST be 0.0 - 1.0)")
print(f"  Price:    {r['predicted_price']:.2f}")
assert 0.0 <= r['hit_rate'] <= 1.0, f"Hit rate out of range: {r['hit_rate']}"
assert -1.0 <= r['r_squared'] <= 1.0, f"R² out of range: {r['r_squared']}"

# Scenario 2: More realistic with noise, days_ahead=79 (like the user's case)
import numpy as np
np.random.seed(42)
td2 = [TrainingData(date=d.strftime('%Y-%m-%d'), close=100 + i*0.5 + np.random.normal(0, 3),
       volume=1000 + np.random.normal(0, 200), sma_20=100 + i*0.5 - 2,
       volatility=0.02, fed_funds=4.5, cpi=300)
       for i, d in enumerate(pd.date_range(start='2023-01-01', periods=500))]
cf2 = {'close': td2[-1].close, 'volume': td2[-1].volume, 'sma_20': td2[-1].sma_20,
       'volatility': 0.02, 'fed_funds': 4.5, 'cpi': 300}
r2 = train_and_predict(PredictionRequest(training_data=td2, current_features=cf2, days_ahead=79))

print(f"\n=== Noisy Trend (days_ahead=79, like user's MSFT case) ===")
print(f"  R²:       {r2['r_squared']:.4f}")
print(f"  Hit Rate: {r2['hit_rate']:.4f}")
print(f"  Price:    {r2['predicted_price']:.2f}")
assert 0.0 <= r2['hit_rate'] <= 1.0, f"Hit rate out of range: {r2['hit_rate']}"
assert -1.0 <= r2['r_squared'] <= 1.0, f"R² out of range: {r2['r_squared']}"

print("\n✓ All assertions passed. Metrics are in valid ranges.")
