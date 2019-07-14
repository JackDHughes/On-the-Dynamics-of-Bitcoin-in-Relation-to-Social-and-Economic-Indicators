import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


google = pd.read_csv("data/google.csv")
print(np.matrix(google))


x = np.matrix([[1, 1, 4, 9], [1, 2, 2, 4], [1, 3, 9, 4], [1, 2.5, 3, 9]])
y = np.matrix([[3], [4], [1], [2]])
theta = np.matrix([0, 0, 0, 0])


def predictY(theta, x):
  return theta*x.T

def cost(theta, x, y):
  return np.sum(np.square(predictY(theta, x) - y.T))

def newTheta(theta, x, y, alpha):
  return theta - (alpha/x.shape[0])*sum(predictY(theta, x) - y.T)*x

def gradient(theta, x, y, alpha):
  currentTheta = theta
  while np.abs(cost(currentTheta, x, y) - cost(newTheta(currentTheta, x, y, alpha), x, y)) > 0.000001:
    currentTheta = newTheta(currentTheta, x, y, alpha)
  return currentTheta

theta = gradient(theta, x, y, 0.01)
plt.plot(x[:,1], y, "o")
plt.plot(x[:,1], predictY(theta, x).T)
plt.plot(x[:,2], y, "o")
plt.plot(x[:,2], predictY(theta, x).T)
plt.plot(x[:,3], y, "o")
plt.plot(x[:,3], predictY(theta, x).T)
plt.title("Thetas: " + str(theta) + ", \nCost = " + str(cost(theta, x, y)))
#plt.show()


#get number of tweets per day




