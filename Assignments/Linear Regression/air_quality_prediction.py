import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATASET
train_dataset = pd.read_csv("/home/ankur/PycharmProjects/Data/Air Quality Prediction/Train.csv")
test_dataset = pd.read_csv("/home/ankur/PycharmProjects/Data/Air Quality Prediction/Test.csv")

X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.iloc[:, :].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(y_pred.reshape(len(y_pred), 1))