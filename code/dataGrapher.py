import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

y = pd.read_csv("data/filtered/finalCombined/trainingData.csv").iloc[:, 0].values
plt.plot(y)
plt.title("Bitcoin's Popularity on Google")
plt.xlabel("Time")
plt.ylabel("Scaled Number of Google Searches for \"Bitcoin\"")
plt.savefig("results/google.png")
plt.show()

x = pd.read_csv("data/filtered/finalCombined/trainingData.csv").iloc[:, 1].values
plt.plot(x)
plt.title("Volume of Tweets Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Tweets Mentioning \"Bitcoin\"")
plt.savefig("results/tweets.png")
plt.show()

u = pd.read_csv("data/filtered/finalCombined/trainingData.csv").iloc[:, 2].values
plt.plot(u)
plt.title("Euro to US Dollar Exchange Rate")
plt.xlabel("Time")
plt.ylabel("Conversion From Euros to US Dollars")
plt.savefig("results/Euro_USD.png")
plt.show()

t = pd.read_csv("data/filtered/finalCombined/trainingData.csv").iloc[:, 3].values
plt.plot(t)
plt.title("Japanese Yen to US Dollar Exchange Rate")
plt.xlabel("Time")
plt.ylabel("Conversion From Japanese Yen to US Dollars")
plt.savefig("results/Yen_USD.png")
plt.show()

u = pd.read_csv("data/filtered/finalCombined/trainingData.csv").iloc[:, 4].values
plt.plot(u)
plt.title("S&P 500 Index Values Over Time")
plt.xlabel("Time")
plt.ylabel("S&P 500 Index Value")
plt.savefig("results/SP500.png")
plt.show()

u = pd.read_csv("data/filtered/finalCombined/bitcoin.csv").iloc[:, 1].values
plt.plot(u)
plt.title("Bitcoin's Value Over Time")
plt.xlabel("Time")
plt.ylabel("Value of Bitcoin")
plt.savefig("results/bitcoin.png")
plt.show()