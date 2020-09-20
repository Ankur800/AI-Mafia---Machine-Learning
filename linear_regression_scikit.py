# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# data preparation / data loading
dfx = pd.read_csv("/home/ankur/PycharmProjects/Data/linearX.csv")
dfy = pd.read_csv("/home/ankur/PycharmProjects/Data/linearY.csv")

# convert values in numpy arrays
x = dfx.values
y = dfy.values

# Linear Regression class's object
model = LinearRegression()

# Training
model.fit(x, y)     # fitting data in object

# Prediction
output = model.predict(x)

# print(model.score(x, y))      # tells the accuracy of model

# Visualize
plt.scatter(x, y, label='data')
plt.plot(x, output, color='black', label='prediction')
plt.legend()
plt.show()

# bias = model.intercept_
# coeff = model.coef_
# print(bias)
# print(coeff)