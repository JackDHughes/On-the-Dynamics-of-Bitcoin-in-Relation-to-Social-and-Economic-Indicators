import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data = pd.read_csv("data/bitcoinPrices.csv")
#tweets = pd.read_csv("data/twitterData.csv") 
#print(tweets.values)
#print(data)
#print(tweets)


x = np.matrix([[1, 1], [1, 2], [1, 3]])
y = np.matrix([[1], [2], [3]])
theta = np.matrix([0, 0])


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
plt.title("Thetas: " + str(theta) + ", \ncost = " + str(cost(theta, x, y)))
plt.show()


#get number of tweets per day




