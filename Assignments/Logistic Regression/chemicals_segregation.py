import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from mpl_toolkits import mplot3d

X_dataset = pd.read_csv("/home/ankur/PycharmProjects/Data/Logistic Regression/Logistic_X_Train.csv")
y_dataset = pd.read_csv("/home/ankur/PycharmProjects/Data/Logistic Regression/Logistic_Y_Train.csv")
X_dataset_test = pd.read_csv("/home/ankur/PycharmProjects/Data/Logistic Regression/Logistic_X_Test.csv")

X_train = X_dataset.values
y_train = y_dataset.iloc[:, -1].values
X_test = X_dataset_test.values

xt = X_train[:, 1:2].reshape((-1, ))
yt = X_train[:, 2:3].reshape((-1, ))
zt = X_train[:, :1].reshape((-1, ))

# DATA VISUALIZATION
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(xt, yt, zt,  cmap='hsv')
plt.show()

# TRAINING
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(xt, yt, zt,  cmap='hsv')

zt = (-(classifier.coef_[0][1]*xt + classifier.coef_[0][0]*yt + classifier.intercept_[0]) / classifier.coef_[0][2])

ax.plot3D(xt, yt, zt, color='k', alpha=0.2)
plt.show()

y_pred = classifier.predict(X_test)
print(y_pred)

print(classifier.coef_)