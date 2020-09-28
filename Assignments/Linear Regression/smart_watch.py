# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATASET
X_train = pd.read_csv("/home/ankur/PycharmProjects/Data/Linear_regression_smartWatch/Linear_X_Train.csv")
y_train = pd.read_csv("/home/ankur/PycharmProjects/Data/Linear_regression_smartWatch/Linear_Y_Train.csv")
X_test = pd.read_csv("/home/ankur/PycharmProjects/Data/Linear_regression_smartWatch/Linear_X_Test.csv")

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values

plt.scatter(X_train, y_train, color='red')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()