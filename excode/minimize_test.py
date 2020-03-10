import numpy as np
from matplotlib import pyplot as plt

from datasets import dataset_1
from gradient_descent import gradient_descent
from linear_cost import linear_cost
from linear_cost_derivate import linear_cost_derivate

# Training data
(X, y) = dataset_1
m, n = X.shape

theta_0 = np.random.rand(n, 1)
theta, costs, gradient_norms = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost,
    linear_cost_derivate,
    alpha=0.000000001,
    treshold=0.001,
    max_iter=10000
)
print("COSTS: ")
print(costs)
print("GN: ")
print(gradient_norms)

print ('THETA:', theta)

# Plot training data
# plt.scatter(X[:, 1], y)

# plt.plot(X[:, 1], np.matmul(X, theta), color='red')

plt.plot(np.arange(len(costs)), costs)

plt.show()

# # X => (11, 2)
# Xtraining = X[0:5, :]
# ytraining = y[0:5, :]
# Xcv = X[5:8, :]
# ycv = y[5:8, :]
# Xtest = X[8:, :]
# ytest = y[8:, :]

# # Model deduction
# theta = np.linalg.lstsq(Xtraining, ytraining)[0]

# print("Jtraining:", linear_cost(theta, Xtraining, ytraining))
# print("Jcv:", linear_cost(theta, Xcv, ycv))







