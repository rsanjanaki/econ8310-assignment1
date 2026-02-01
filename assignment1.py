import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train_data = pd.read_csv('assignment_data_train.csv')
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
train_data.set_index('Timestamp', inplace=True)


y_train = train_data['trips']

model = ExponentialSmoothing(
    y_train,
    seasonal_periods=168, 
    trend='add',
    seasonal='add',
    initialization_method='estimated'
)

modelFit = model.fit(optimized=True)

test_data = pd.read_csv('assignment_data_test.csv')

pred = modelFit.forecast(steps=744)

pred = np.array(pred)

test_trips_actual = pd.read_csv('tests/testData.csv')


rmse = np.sqrt(sum([(pred[i]-test_trips_actual['trips'][i])**2 for i in range(len(pred))])) / 744

print(f"RMSE: {rmse:.2f}")
print(f"\nAccuracy Levels:")
print(f"  Level 1 (< 220): {'✓ PASS' if rmse < 220 else '✗ FAIL'}")
print(f"  Level 2 (< 185): {'✓ PASS' if rmse < 185 else '✗ FAIL'}")
print(f"  Level 3 (< 171): {'✓ PASS' if rmse < 171 else '✗ FAIL'}")
print(f"\nPrediction Statistics:")
print(f"  Min predicted trips: {pred.min():.0f}")
print(f"  Max predicted trips: {pred.max():.0f}")
print(f"  Mean predicted trips: {pred.mean():.0f}")
print(f"  Total predictions: {len(pred)}")



