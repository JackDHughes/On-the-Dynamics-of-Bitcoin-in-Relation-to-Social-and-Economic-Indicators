import pandas as pd
import numpy as np
from datetime import datetime, timedelta

google = pd.read_csv("data/unfiltered/gTrends.csv", header=None, names=["week", "popularity"]).drop([0,1], axis=0)

#for datapoint in google.iterrows():

values = []

def repeateValues(arr, times):
  output = []
  for val in arr:
    for i in range(times):
      output.append(val)
  return output
#print(datetime.strptime(war_start, '%Y-%m-%d'))

for i in range(len(google)):
  values.append([google.iloc[i,0], google.iloc[i,1]])

filtered = pd.DataFrame(repeateValues(values, 7))

for week in range(len(filtered)):
  for day in range(7):
    filtered[0].


print(filtered.head())
print(datetime.strptime(filtered[0].values[0], '%Y-%m-%d') - timedelta(days=1))
#.to_csv("data/filtered/google.csv")

