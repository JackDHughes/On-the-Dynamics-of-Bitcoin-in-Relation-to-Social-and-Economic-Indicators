import pandas as pd
import numpy as np

google = pd.read_csv("data/unfiltered/google.csv", header=None, names=["week", "popularity"]).drop([0,1], axis=0)

#for datapoint in google.iterrows():

values = []

def repeateValues(arr, times):
  output = []
  for val in arr:
    for i in range(times):
      output.append(val)
  return output


for i in range(len(google)):
  values.append([google.iloc[i,0], google.iloc[i,1]])

pd.DataFrame(repeateValues(values, 7)).to_csv("data/filtered/google.csv")

print(google.head())
