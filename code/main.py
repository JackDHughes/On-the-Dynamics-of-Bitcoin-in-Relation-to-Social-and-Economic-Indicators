import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import time

x = pd.read_csv("data/filtered/finalCombined/trainingData.csv", header=None, names=["gTrends", "twitter", "euros", "yen", "sp500"]).iloc[:, :].values
y = pd.read_csv("data/filtered/finalCombined/bitcoin.csv").iloc[:,1].values
x = (x - x.mean()) / x.std()
x = np.hstack((np.ones((len(x), 1)), x))

theta = np.matrix([0, 0, 0, 0, 0,0])
#print(x)


def predictY(theta, x):
  return theta*x.T

def cost(theta, x, y):
  return np.sum(np.square(predictY(theta, x) - y.T))

def scaledCost(theta,x,y):
  return cost(theta, x, y) / len(y)

def newTheta(theta, x, y, alpha):
  return theta - (alpha/x.shape[0])*sum(predictY(theta, x) - y.T)*x

def gradient(theta, x, y, alpha):
  currentTheta = theta
  costs = [cost(theta, x, y)]
  iters = 0
  while np.abs(cost(currentTheta, x, y) - cost(newTheta(currentTheta, x, y, alpha), x, y)) > 10000:
    currentTheta = newTheta(currentTheta, x, y, alpha)
    costs.append(cost(currentTheta, x, y))
    iters+=1
  return [currentTheta, costs, iters]

costInitial = cost(theta,x,y)
print("Initial Cost: " + str(costInitial))

# start = time.time()
# model = gradient(theta, x, y, 0.1)
# theta = model[0]
# end = time.time()

# costFinal = cost(theta, x, y)
# print("Ran " + str(model[2]) + " iterations")
# print("Finished training in " + str(end-start) + " seconds")
# print("Final Cost: " + str(costFinal))
# print("Total cost decreased by " + str(100 * (costInitial - costFinal) / costInitial) + "%")


# plt.plot(y)
# plt.plot(predictY(theta, x).T)
# plt.show()

# plt.plot(model[1])
# plt.show()
# print(theta)
print(x.shape)
print(y.shape)
print(theta.shape)

#[[ 4.92853217e+05  1.43360325e+06  6.26848306e+02 -2.08803383e+05 -2.25269992e+05  1.10590662e+05]] for all


