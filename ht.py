import numpy as np

from module import *
from matplotlib import pyplot as plt
data = np.loadtxt('Admission_Predict.csv',delimiter=",", skiprows=1)

#training set
t_set = data[:240,]
#cross validation set
cv_set = data[240:320,] 
#test set
test = data[320:,]
#X,y = t_set[:,1:len(t_set[0])-1], t_set[:,len(t_set[0])-1:len(t_set[0])],
#X,y = t_set[:,1], t_set[:,len(t_set[0])-1:len(t_set[0])]

# X, y = np.vstack((np.ones(len(t_set)), t_set[:,1])).T, t_set[:,len(t_set[0])-1:len(t_set[0])]

y = t_set[:,len(t_set[0])-1:len(t_set[0])]

X = np.vstack((
    np.ones(len(t_set)),
    t_set[:,1],
    t_set[:,1] **2 /100
)).T

# print(X)
x_t = X[:,1]
m, n = X.shape

theta_0 = np.random.rand(n, 1)
theta, costs, gradient_norms = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost_regular,
    linear_cost_derivate_regular,
    alpha=0.000001,
    threshold=0.01,
    max_iter= 100000,
    lamda=10
)

y_predic = np.matmul(X, theta)
#r^2
r2 = (((y-y.mean())**2).sum()-((np.matmul(X, theta)-y)**2).sum())/((y-y.mean())**2).sum()
print("R^2: ", r2)



## graficas 
plt.scatter(X[:, 1], y)
plt.scatter(X[:, 1], y_predic, color='red')
plt.show()

# plt.plot(np.arange(len(costs)), costs)

# plt.show()