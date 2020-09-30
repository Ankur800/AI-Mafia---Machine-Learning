# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# IMPORTING DATASET
dataset = pd.read_csv("/home/ankur/PycharmProjects/Data/Digit Recognition/train.csv")
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# train test split
split = int(0.8 * X.shape[0])

X_train = X[:split, :]
y_train = y[:split]

X_test = X[split:, :]
y_test = y[split:]

# VISUALIZATION OF SOME SAMPLES
def drawImage(sample):
    # reshape that shape
    img = sample.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

drawImage(X_train[0])
print(y_train[0])

def distance(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))

# KNN Algo
def knn(x, y, query_point, k=5):
    # pick K nearest neighbours

    vals = []

    # for every point in the x
    for i in range(x.shape[0]):
        # compute distance
        d = distance(query_point, x[i])
        vals.append((d, y[i]))

    # Step 2: Sort the array and find k nearest points
    vals = sorted(vals)

    vals = vals[:k]

    # Majority votes

    vals = np.array(vals)

    new_values = np.unique(vals[:, 1], return_counts=True)

    # index of the maximum count
    index = new_values[1].argmax()

    # map this index with my data
    pred = new_values[0][index]

    return pred

# make predictions over test images
pred = knn(X_train, y_train, X_test[25])
print(pred)

drawImage(X_test[25])

def findAccuracy():
    correctResults = 0
    for i in range(X_test.shape[0]):
        tempPred = knn(X_train, y_train, X_test[i])
        if tempPred == y_test[i]:
            correctResults += 1

    accuracy = correctResults / len(X_test)
    return accuracy

print(findAccuracy())