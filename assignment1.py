import pandas as pd
import numpy as np
import os
from prophet import Prophet


train_path = "assignment_data_train.csv"
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"

if os.path.exists(train_path):
    train_data = pd.read_csv(train_path)
else:
    train_data = pd.read_csv(train_url)


train_data["Timestamp"] = pd.to_datetime(train_data["Timestamp"])
train_data = train_data.sort_values("Timestamp")


prophet_train = train_data.rename(columns={"Timestamp": "ds", "trips": "y"})[["ds", "y"]]


model = Prophet(
    yearly_seasonality=False,      
    weekly_seasonality=True,       
    daily_seasonality=True,        
    seasonality_mode="additive"   


model.add_seasonality(name="hourly", period=24, fourier_order=10)


modelFit = model.fit(prophet_train)


future = model.make_future_dataframe(periods=744, freq="H")


forecast = modelFit.predict(future)


pred = forecast["yhat"].tail(744).to_numpy()


pred = np.maximum(pred, 0)
