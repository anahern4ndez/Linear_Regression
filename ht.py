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
X, y = np.vstack((np.ones(len(t_set)), t_set[:,6], t_set[:,6]**2)).T, t_set[:,len(t_set[0])-1:len(t_set[0])]

x_t = X[:,1]
m, n = X.shape

theta_0 = np.random.rand(n, 1)

theta = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost,
    linear_cost_derivate,
    alpha=0.000001,
    threshold=0.000001,
    max_iter=100000,
    lamda=2
)

plt.scatter(x_t, y)

#plt.plot(X[:, 1], np.matmul(X, theta), color='red')
plt.plot(x_t, (x_t*theta[1] + theta[0]), color='red')

plt.show()