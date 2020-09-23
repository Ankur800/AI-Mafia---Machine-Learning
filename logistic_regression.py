import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# **********************
#    DATA PREPARATION  *
# **********************

mean_01 = np.array([1, 0.5])
cov_01 = np.array([[1, 0.1], [0.1, 1.2]])

mean_02 = np.array([4, 5])
cov_02 = np.array([[1.21, 0.1], [0.1, 1.3]])


# Normal Distribution
dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)

print(dist_01.shape)
print(dist_02.shape)

plt.figure(0)
plt.scatter(dist_01[:, 0], dist_01[:, 1], label='Class 0')
plt.scatter(dist_02[:, 0], dist_02[:, 1], color='r', marker='^', label='Class 1')
plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# **************************************************
#  SPLITTING DATA IN TRAINING AND TESTING DATASET  *
# **************************************************
data = np.zeros((1000, 3))
# print(data.shape)

data[:500, :2] = dist_01
data[500:, :2] = dist_02
data[500:, -1] = 1.0

np.random.shuffle(data)
print(data[:10])

split = int(0.8*data.shape[0])

X_train = data[:split, :-1]
X_test = data[split:, :-1]

Y_train = data[:split, -1]
Y_test = data[split:, -1]

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

# ***********************
#  LOGISTIC REGRESSION  *
# ***********************

def hypothesis(x, w, b):    # w-weight matrix [ a ]; b-bias(for intercept)
                                            # [ b ]
    hx = np.dot(x, w) + b       # dot product
    return sigmoid(hx)

def sigmoid(hx):
    return 1.0/(1.0 + np.exp(-1.0 * hx))

def errorFunction(x, y, w, b):       # error function
    error = 0.0
    for i in range(x.shape[0]):
        hx = hypothesis(x[i], w, b)
        error += y[i] * np.log2(hx) + (1-y[i]) * np.log2(1 - hx)
    return error/x.shape[0]         # mean absolute error

def getGradient(x, y, w, b):

    grad_b = 0.0                    # gradient for bias
    grad_w = np.zeros(w.shape)      # gradient for weight matrix
    for i in range(x.shape[0]):
        hx = hypothesis(x[i], w, b)
        grad_b += (y[i] - hx)
        grad_w += (y[i] - hx) * x[i]
    return [grad_w/x.shape[0], grad_b/x.shape[0]]

def gradientAscent(x, y, w, b, learning_rate=0.01):
    error = errorFunction(x, y, w, b)
    [grad_w, grad_b] = getGradient(x, y, w, b)
    w += learning_rate * grad_w
    b += learning_rate * grad_b
    return error, w, b

w = 2 * np.random.random((X_train.shape[1], ))
b = 5 * np.random.random()

loss = []

for i in range(1000):
    l, w, b = gradientAscent(X_train, Y_train, w, b, learning_rate=0.1)
    loss.append(l)

plt.plot(loss)
plt.show()

print(w, b)

plt.figure(0)
plt.scatter(dist_01[:, 0], dist_01[:, 1], label='Class 0')
plt.scatter(dist_02[:, 0], dist_02[:, 1], color='r', marker='^', label='Class 1')
plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.xlabel('x1')
plt.ylabel('x2')

x = np.linspace(-4, 8, 10)
y = -(w[0] * x + b) / w[1]

plt.plot(x, y, color='k')
plt.legend()
plt.show()