import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '~/Desktop/SERA/Discussions/ipython-notebooks-master/data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=["Size", "Beadroom", "Price"])
data2.head()

data2 = (data2 - data2.mean())/ data2.std()
data2.head()

#add ones colum
data2.insert(0, 'Ones', 0)

#set x(training data) and Y (target variable)
cols = data2.shape[1]
x2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]
theta2 = np.matrix(np.array([0,0,0]))

#Convert to matrices and initialize theta
x2 = np.matrix(x2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x2, y2)

prediction = model.predict(x2)
prediction.shape

def computeCost(y, y_pred):
    inner = np.power((y_pred-y),2)
    return np.sum(inner)/2*len(y)

computeCost(y2, prediction)

model.get_params()

model.score(x2, y2)


#LOGISTIC REGRESSION

from scipy.io import loadmat

path = '~/Desktop/Data/ex3data1.txt'
data = loadmat(path)
data

data['x'].shape
x = data['x']
first_image = x[0]
first_image.shape
first_image = np.reshape(20,20)
first_image.shape

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cost(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(x * theta.T)))
    reg = (learningRate / 2 * len(x)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(x)) + reg

def gradient(theta, x, y, learningRate):
    theta = np.martix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(x * theta.T) - y
    
    grad = ((x.T * error) / len(x)).T + ((learningrate / len(x)) * theta)
    
    #intercept gradient is not regularized
    grad[0,0] = np.sum(np.multiply(error, x[:,0])) / len(x)
    
    return np.array(grad).ravel()

    from scipy.optimize import minimize

def one_vs_all(x, y, num_labels, learning_rate):
    rows = x.shape[0]
    params = x.shape[1]
    #k x (n+1) array for the parmameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    #insert a comum of ones at the beginning for the intercept term
    x = npinsert(x, 0, values=np.ones(rows), axis=1)
    
    #labels are 1-indexed instread of 0-indexed
    for i in range(1, num_labels + 1):
      theta = np.zeros(params + 1)
      y_i = np.array([1 if label == i else 0 for label in y])
      y_i = np.reshape(y_i, (rows,1))
        
      #minimize the objective function
      fmin = minimize(fun=cost, x0=theta, args=(x, y_i, learning_rate), method='TNC', jac=gradient)
      all_theta[i-1, :] = fmin.x
        
      return all_theta

rows = data['x'].shape[0]
params = data['x'].shape[1]

x = np.insert(data['x'], 0, values=npones(rows), axis=1)
theta = np.zeros(params + 1)
all_theta = one_vs_all(data['x'], data['y'], 10, 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1)


def predict_all(x, all_theta):
  rows = x.shape[0]
  params = x.shape[1]
  num_labels = all_theta.shape[0]
    
  x = np.insert(x, 0, values=np.ones(rows), axis=1)
  x = np.matrix(x)
  all_theta = np.matrix(all_theta)
    
    #compute the class probability for each class on each triaing instance
  h = sigmoid(x * all_theta.T)
    
    #Create array of the index with the max probability
  h_argmax = np.argmax(h, axis=1)
    
    #because our array was zero-indexed we need to add one for the true label prediction
  h_argmax = h_argmax + 1
    
  return h_argmax


y_pred = predict_all(data['x'], all_theta)
correct = [1 if a==b else 0 for(a,b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))


