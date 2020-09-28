import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the training data
dfx = pd.read_csv("/home/ankur/PycharmProjects/Data/linearX.csv")
dfy = pd.read_csv("/home/ankur/PycharmProjects/Data/linearY.csv")

dfx = dfx.values
dfy = dfy.values

x = dfx.reshape((-1,))
y = dfy.reshape((-1,))

x = (x - x.mean()) / x.std()        # Shifting points to origin, i.e. fixing a range for points
y = y
plt.scatter(x, y)
plt.show()


# Gradient Descent Algorithm
def hypothesis(x, theta):
    return theta[0] + theta[1] * x


def errorFun(x, y, theta):
    error = 0
    for i in range(x.shape[0]):
        hx = hypothesis(x[i], theta)
        error += (hx - y[i])**2
    return error


def gradient(x, y, theta):
    gred = np.zeros((2,))
    for i in range(x.shape[0]):
        hx = hypothesis(x[i], theta)
        gred[0] += (hx - y[i])
        gred[1] += (hx - y[i]) * x[i]
    return gred


def gradientDescent(x, y, learning_rate=0.001):

    # random theta
    theta = np.array(([-2.0, 0.0]))

    max_iteration = 100
    itr = 0

    error_list = []
    theta_list = []

    while itr <= max_iteration:
        grad = gradient(x, y, theta)
        err = errorFun(x, y, theta)
        error_list.append(err)
        theta_list.append(theta)
        theta[0] -= learning_rate * grad[0]
        theta[1] -= learning_rate * grad[1]

        itr += 1

    return theta, error_list, theta_list


final_theta, error_list, theta_list = gradientDescent(x, y)
plt.plot(error_list)
plt.show()

print(final_theta)

# plot the line for testing data
xTest = np.linspace(-2, 6, 10)
print(xTest)

plt.scatter(x, y, label='Training Data')
plt.plot(xTest, hypothesis(xTest, final_theta), color='red', label='prediction')
plt.legend()
plt.show()