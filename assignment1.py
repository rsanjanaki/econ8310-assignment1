import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os


train_path = "assignment_data_train.csv"
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"

if os.path.exists(train_path):
    train_data = pd.read_csv(train_path)
else:
    train_data = pd.read_csv(train_url)


train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
train_data = train_data.sort_values('Timestamp').set_index('Timestamp')


y_train = train_data['trips']


y_train.index.freq = y_train.index.inferred_freq


model = ExponentialSmoothing(
    y_train,
    seasonal_periods=168,        
    trend='add',                  
    seasonal='add',               
    initialization_method='estimated'
)


modelFit = model.fit(optimized=True)


pred = modelFit.forecast(steps=744)


pred = np.array(pred)
