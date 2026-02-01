import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

if os.path.exists('assignment_data_train.csv'):
    train_data = pd.read_csv('assignment_data_train.csv')
else:
    train_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')

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

if os.path.exists('assignment_data_test.csv'):
    test_data = pd.read_csv('assignment_data_test.csv')
else:
    test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

pred = modelFit.forecast(steps=744)

pred = np.array(pred)

