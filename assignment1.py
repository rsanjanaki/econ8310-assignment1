import pandas as pd
import numpy as np
import os
from pygam import LinearGAM, s, f


train_path = "assignment_data_train.csv"
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"

if os.path.exists(train_path):
    train_data = pd.read_csv(train_path)
else:
    train_data = pd.read_csv(train_url)


train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
train_data = train_data.sort_values('Timestamp')


x_train = train_data[['year', 'month', 'day', 'hour']].values
y_train = train_data['trips'].values


model = LinearGAM(s(0) + f(1) + f(2) + s(3))



modelFit = model.gridsearch(x_train, y_train)


test_path = "assignment_data_test.csv"
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

if os.path.exists(test_path):
    test_data = pd.read_csv(test_path)
else:
    test_data = pd.read_csv(test_url)


test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])
test_data = test_data.sort_values('Timestamp')
x_test = test_data[['year', 'month', 'day', 'hour']].values


pred = modelFit.predict(x_test)


pred = np.maximum(pred, 0)
