import numpy as np

# def gradient_descent(
#         X,
#         y,
#         theta0,
#         cost,
#         cost_derivate,
#         alpha, threshold, maxIteraciones):
#     theta, i = theta0, 0
#     gradiente = cost_derivate
#     costs =[]
#     gradient_norm = []
#     while(np.linalg.norm(gradiente(X, y, theta)) > threshold and i < maxIteraciones):
#         theta0 -= alpha * gradiente(X, y, theta)
#         i+=1
#         costs.append(cost(X, y, theta))
#         gradient_norm.append(gradiente(X, y, theta))
#     return theta

def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        alpha=0.01,
        threshold=0.0001,
        max_iter=10000,
        lamda = 0):
    theta, i = theta_0, 0
    while np.linalg.norm(cost_derivate(X, y, theta, lamda)) > threshold and i < max_iter:
        #print(theta, cost_derivate(X, y, theta, lamda))
        theta -= alpha * cost_derivate(X, y, theta, lamda)
        i += 1
    return theta


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