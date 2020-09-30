# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# IMPORTING DATASET
dfx = pd.read_csv("/home/ankur/PycharmProjects/Data/Diabetes Classification/Diabetes_XTrain.csv")
dfy = pd.read_csv("/home/ankur/PycharmProjects/Data/Diabetes Classification/Diabetes_YTrain.csv")
dfx_test = pd.read_csv("/home/ankur/PycharmProjects/Data/Diabetes Classification/Diabetes_Xtest.csv")

X_train = dfx.values
y_train = dfy.iloc[:, -1].values
X_test = dfx_test.values

dfx[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dfx[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# PLOTTING BAR CHART
left = [1, 2, 3, 4, 5, 6, 7, 8]

max_height = int(dfx[['Glucose']].size)
preg = int(max_height - dfx[['Pregnancies']].isnull().sum())
gl = int(max_height - dfx[['Glucose']].isnull().sum())
bp = int(max_height - dfx[['BloodPressure']].isnull().sum())
st = int(max_height - dfx[['SkinThickness']].isnull().sum())
ins = int(max_height - dfx[['Insulin']].isnull().sum())
bmi = int(max_height - dfx[['BMI']].isnull().sum())
dpf = max_height
age = max_height

height = [preg, gl, bp, st, ins, bmi, dpf, age]

tick_labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']

plt.bar(left, height, tick_label=tick_labels, width=0.6, color=['red', 'blue'])
plt.show()

# Training data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)   # minkowski is Euclidean distance
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)