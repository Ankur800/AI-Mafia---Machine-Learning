# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# IMPORTING DATASET
dfx = pd.read_csv("/home/ankur/PycharmProjects/Data/KNN/xdata.csv")
dfy = pd.read_csv("/home/ankur/PycharmProjects/Data/KNN/ydata.csv")
# Let's convert it into numpy array
x = dfx.iloc[:, 1:].values
y = dfy.iloc[:, -1].values

# PLOTTING DATASET
# let's take some query point
query = np.array([2, 3])
plt.scatter(query[0], query[1], color='red')
# dataset points
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# We want to find out the prediction for query point
# Start with KNN code
# STEP 1: To find K nearest neighbours based on distance
# distance formula between two numpy arrays - Euclidean distance
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

    print(vals)

    # Step 2: Sort the array and find k nearest points
    vals = sorted(vals)

    vals = vals[:k]

    # Majority votes

    vals = np.array(vals)

    new_values = np.unique(vals[:, 1], return_counts=True)

    print(new_values)

    # index of the maximum count
    index = new_values[1].argmax()

    # map this index with my data
    pred = new_values[0][index]

    return pred


print(knn(x, y, [2, 2]))