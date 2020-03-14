import numpy as np
import random 
from module import *
from matplotlib import pyplot as plt
data = np.loadtxt('Admission_Predict.csv',delimiter=",", skiprows=1)

#normalizacion de data 
norm_t_set = np.array([np.zeros(len(data))])

for arr in data.T:
    norm_tuple = []
    for index in range(len(arr)):
        norm_tuple.append((arr[index] - arr.min())/ (arr.max()- arr.min()))
    norm_t_set = np.vstack((norm_t_set, np.array(norm_tuple)))
data = norm_t_set[1:,]
data = data.T
data = data[:,1:]

# #training set
# t_set = data[:240,1:]
# #cross validation set
# cv_set = data[240:320,1:] 
# #test set
# test = data[320:,1:]

#slice de data random 
# Seleccionar conjunto de training y test

t_set = np.zeros(8).reshape((1,8))
cv_set = np.zeros(8).reshape((1,8))
test = np.zeros(8).reshape((1,8))
func = [t_set, cv_set, test]
perc = [240, 80, 80]
t_rand = []
for x in range(3):
    while(len(func[x]) < perc[x]+1):
        r = random.randint(0,399)
        if r not in t_rand: 
            t_rand.append(r)
            func[x] = np.concatenate((func[x], data[r].reshape((1,8))))
            
t_set, cv_set, test = func[0][1:,], func[1][1:,], func[2][1:,]

y = t_set[:,len(t_set[0])-1:len(t_set[0])]

X = np.vstack((
    np.ones(len(t_set)),
    t_set[:,:7].T,
)).T

m, n = X.shape

#Después de múltiples generaciones random, se encontró que theta_0 que brindaba los mejores resultados, era:
# theta_0 = np.random.rand(n, 1)
theta_0 = np.array([[-0.06706054],
 [ 0.21327747],
 [-0.19349977],
 [ 0.10454866],
 [-0.06015129],
 [ 0.28915204],
 [ 0.27315269],
 [-0.00179598]])

print("THETA_0: ", theta_0)

theta = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost_regular,
    linear_cost_derivate_regular,
    alpha=0.0001,
    threshold=0.0001,
    max_iter= 300000,
    lamda=0
)

y_predic = np.matmul(X, theta)
#r^2
r2 = (((y-y.mean())**2).sum()-((np.matmul(X, theta)-y)**2).sum())/((y-y.mean())**2).sum()
print("R^2: ", r2)
## graficas 
plt.scatter(X[:, 1], y)
plt.scatter(X[:, 1], y_predic, color='red')

plt.show()



# ### CROSS VALIDATION 

# cv_x = np.vstack((
#     np.ones(len(cv_set)),
#     cv_set[:,:7].T
# )).T


# cv_y = cv_set[:,len(cv_set[0])-1:len(cv_set[0])]

# # plt.scatter(cv_x[:, 1], cv_y)
# # plt.scatter(cv_x[:, 1], np.matmul(cv_x, theta), color='red')
# # plt.show()

# theta, costs, gradient_norms = gradient_descent(
#     cv_x,
#     cv_y,
#     theta_0,
#     linear_cost_regular,
#     linear_cost_derivate_regular,
#     alpha=0.0001,
#     threshold=0.0001,
#     max_iter= 300000,
#     lamda=0
# )

# #Cálculo de coeficiente de regresión (R^2)
# r2 = (((cv_y-cv_y.mean())**2).sum()-((np.matmul(cv_x, theta)-cv_y)**2).sum())/((cv_y-cv_y.mean())**2).sum()
# print("CV R^2: ", r2)

# #Error Cuadrático Medio

# ecm = (1/len(cv_set))*((np.matmul(cv_x, theta)-cv_y)**2).sum()
# print("CV ECM: " , ecm)

# #modificacion de lambda para mejores resultados 
# theta, costs, gradient_norms = gradient_descent(
#     X,
#     y,
#     theta_0,
#     linear_cost_regular,
#     linear_cost_derivate_regular,
#     alpha=0.0001,
#     threshold=0.0001,
#     max_iter=30000,
#     lamda=3.5)

# #Cálculo de coeficiente de regresión (R^2)
# r2 = (((cv_y-cv_y.mean())**2).sum()-((np.matmul(cv_x, theta)-cv_y)**2).sum())/((cv_y-cv_y.mean())**2).sum()
# print("CV R^2 lambda 4: ", r2)

# #Error Cuadrático Medio

# ecm = (1/len(cv_set))*((np.matmul(cv_x, theta)-cv_y)**2).sum()
# print("CV ECM lambda 4: " , ecm)