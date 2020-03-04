import numpy as np

def gradient_descent(maxIteraciones, alpha, theta0, cost, cost_derivate, threshold, x,y):
    theta, i = theta0, 0
    gradiente = cost_derivate
    costs =[]
    gradient_norm = []
    while(np.linalg.norm(gradiente(x, y, theta)) > threshold and i < maxIteraciones):
        theta0 -= alpha * gradiente(x, y, theta)
        i+=1
        costs.append(cost(x, y, theta))
        gradient_norm.append(gradiente(x, y, theta))
    return theta

def linear_cost(x, y, theta):
    m, _ = x.shape
    h = np.matmul(x, theta)
    sq = (h - y)**2
    res =  sq.sum() / (2*m)
    return res

def linear_cost_derivate(x, y ,theta):
    h = np.matmul(x, theta)
    m,_ = x.shape
    return np.matmul((h-y).T, x).T / m

def linear_cost_regular(x, y, theta, lamda):
    m, _ = x.shape
    h = np.matmul(x, theta)
    sq = (h - y)**2
    res =  sq.sum() / (2*m)
    regularization = (lamda/2*m)((theta**2).sum())
    return res + regularization

def linear_cost_derivate_regular(x, y ,theta, lamda):
    h = np.matmul(x, theta)
    m,_ = x.shape
    regularization = (lamda/m)((theta).sum())
    return (np.matmul((h-y).T, x).T / m) + regularization