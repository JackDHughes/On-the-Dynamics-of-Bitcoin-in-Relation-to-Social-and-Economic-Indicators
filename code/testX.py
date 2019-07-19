import pandas as pd
import numpy as np
x = pd.read_csv("data/filtered/finalCombined/trainingData.csv", header=None, names=["gTrends", "twitter", "euros", "yen", "sp500"]).iloc[0:2, :]
print(x.head())