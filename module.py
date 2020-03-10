import numpy as np

def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        alpha=0.01,
        threshold=0.0001,
        max_iter=10000,
        lamda=0):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta, lamda)) > threshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta, lamda)
        i += 1
        costs.append(cost(X, y, theta, lamda))
        gradient_norms.append(cost_derivate(X, y, theta, lamda))
    return theta, costs, gradient_norms

# def gradient_descent(
#         X,
#         y,
#         theta_0,
#         cost,
#         cost_derivate,
#         alpha=0.00001,
#         threshold=0.0001,
#         max_iter=10000,
#         lamda = 0):
#     theta, i = theta_0, 0
#     while np.linalg.norm(cost_derivate(X, y, theta, lamda)) > threshold and threshold and i < max_iter:
#         #print(theta, cost_derivate(X, y, theta, lamda))
#         print(np.linalg.norm(cost_derivate(X, y, theta, lamda)))
#         theta -= alpha * cost_derivate(X, y, theta, lamda)
#         i += 1
#         # if(i > max_iter-1):
#         #     print("iter")
#     return theta


def linear_cost(x, y, theta, lamda=0):
    m, _ = x.shape
    h = np.matmul(x, theta)
    sq = (h - y)**2
    res =  sq.sum() / (2*m)
    return res

def linear_cost_derivate(x, y ,theta, lamda=0):
    h = np.matmul(x, theta)
    m,_ = x.shape
    return np.matmul((h-y).T, x).T / m

def linear_cost_regular(x, y, theta, lamda):
    m, _ = x.shape
    h = np.matmul(x, theta)
    sq = (h - y)**2
    res =  sq.sum() / (2*m)
    regularization = (lamda/2*m)*((theta**2).sum())
    return res + regularization

def linear_cost_derivate_regular(x, y ,theta, lamda):
    h = np.matmul(x, theta)
    m,_ = x.shape
    regularization = (lamda/m)*((theta).sum())
    return (np.matmul((h-y).T, x).T / m) + regularization