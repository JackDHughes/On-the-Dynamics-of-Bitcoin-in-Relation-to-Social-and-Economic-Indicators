import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime

data = pd.read_csv("data/filtered/finalCombined/trainingData.csv", header=None, names=["gTrends", "twitter", "euros", "yen", "sp500"]).values
btc = pd.read_csv("data/filtered/finalCombined/bitcoin.csv").iloc[:,1].values
alpha = 0.1



trainX = []
testX = []

trainY = []
testY = []

for i in range(len(btc)):
  if i % 2 == 0:
    trainX.append(data[i])
    trainY.append(btc[i])
  else:
    testX.append(data[i])
    testY.append(btc[i])

def norm(x):
  return np.hstack((np.ones((len(x), 1)), (x - x.mean()) / x.std()))

def predictY(theta, x):
  return theta*x.T

def cost(theta, x, y):
  return (1/(2*len(y))) * np.sum(np.square(predictY(theta, x) - y.T))

def newTheta(theta, x, y, alpha):
  return theta - (alpha/x.shape[0])*sum(predictY(theta, x) - y.T)*x

def gradient(theta, x, y, alpha):
  currentTheta = theta
  costs = [cost(theta, x, y)]
  iters = 0
  while np.abs(cost(currentTheta, x, y) - cost(newTheta(currentTheta, x, y, alpha), x, y)) > 0.001:
    currentTheta = newTheta(currentTheta, x, y, alpha)
    costs.append(cost(currentTheta, x, y))
    iters+=1
  return [currentTheta, costs, iters]

def r2(theta, x, y):
  calcY = predictY(theta, x).T
  avg = y.mean()
  seLine = 0
  seMean = 0
  for i in range(len(y)):
    seLine += (y[i,0] - calcY[i,0])**2
    seMean += (calcY[i,0] - avg)**2
  return 1 - seLine/seMean



trainX = norm(np.matrix(trainX))
testX = norm(np.matrix(testX))
trainY = np.matrix(trainY).T
testY = np.matrix(testY).T

theta = np.matrix([0, 0, 0, 0, 0,0])

def runModel(theta, trainX, testX, trainY, testY, alpha, label):
  costInitial = cost(theta,trainX,trainY)

  print("------" + label + " FEATURES-----")

  print("Initial Cost: " + str(costInitial))
  start = time.time()
  model = gradient(theta, trainX, trainY, alpha)
  theta = model[0]
  end = time.time()

  costFinal = cost(theta, trainX, trainY)
  print("Ran " + str(model[2]) + " iterations | alpha = " + str(alpha))
  print("Finished training in " + str(end-start) + " seconds")
  print("Final Cost: " + str(costFinal))
  print("Total cost decreased by " + str(100 * (costInitial - costFinal) / costInitial) + "%")
  print("R^2 (Training): " + str(r2(theta, trainX, trainY)))
  print("R^2 (Testing): " + str(r2(theta, testX, testY)))

  # training data
  plt.plot(range(len(trainX)), predictY(theta, trainX).T, label="Prediction")
  plt.plot(range(len(trainX)), trainY, label="Actual")
  plt.title("Training Data")
  plt.xlabel("Time")
  plt.ylabel("Bitcoin Price")
  plt.savefig(label + "TrainingData" + str(datetime.datetime.now()) + ".png")
  plt.legend(loc=2)
  plt.show()

  plt.plot(model[1])
  plt.title("Gradient Descent")
  plt.xlabel("Iterations")
  plt.ylabel("J(θ)")
  plt.savefig(label + "COST" + str(datetime.datetime.now()) + ".png")
  plt.show()

  # test data
  plt.plot(range(len(testX)), predictY(theta, testX).T, label="Prediction")
  plt.plot(range(len(testX)), testY, label="Actual")
  plt.savefig(label + "TestingData" + str(datetime.datetime.now()) + ".png")
  plt.title("Test Data")
  plt.xlabel("Time")
  plt.ylabel("Bitcoin Price")
  plt.legend(loc=2)
  plt.show()
  print(theta)

  pd.DataFrame(theta).to_csv(label + "thetas" + str(datetime.datetime.now()) + ".csv")
  print("")

runModel(theta, trainX, testX, trainY, testY, alpha, "ALL")
runModel(np.matrix([0, 0, 0]), trainX[:,0:3], testX[:,0:3], trainY, testY, alpha, "SOCIAL")
runModel(np.matrix([0, 0, 0, 0]), np.hstack((np.ones((len(trainX), 1)), trainX[:,3:6])), np.hstack((np.ones((len(testX), 1)), testX[:,3:6])), trainY, testY, alpha, "ECON")


socialTest = testX[:,0:3]
econTest = np.hstack((np.ones((len(testX), 1)), testX[:,3:6]))

econThetas = pd.read_csv("ECONthetas.csv").values[:,1:5]
socialThetas = pd.read_csv("SOCIALthetas.csv").values[:,1:4]
allThetas = pd.read_csv("ALLthetas.csv").values[:,1:8]

plt.scatter(np.array(predictY(allThetas, testX)), np.array(testY).T, marker=".")
plt.title("All Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.scatter(np.array(predictY(socialThetas, socialTest)), np.array(testY), marker=".")
plt.title("Social Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.scatter(np.array(predictY(econThetas, econTest)), np.array(testY), marker=".")
plt.title("Economic Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Econ Cost: " + str(cost(econThetas, np.hstack((np.ones((len(testX), 1)), testX[:,3:6])), testY)))
print("Social Cost: " + str(cost(socialThetas, testX[:,0:3], testY)))
print("Total Cost: " + str(cost(allThetas, testX, testY)))

print("Econ R2: " + str(r2(econThetas, econTest, testY)))
print("Social R2: " + str(r2(socialThetas, socialTest, testY)))
print("Total R2: " + str(r2(allThetas, testX, testY)))