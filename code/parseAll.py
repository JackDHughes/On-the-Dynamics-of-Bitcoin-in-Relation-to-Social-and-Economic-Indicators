import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bitcoin = dict(pd.read_csv("data/filtered/bitcoin.csv", header=None).values)
gTrends = dict(pd.read_csv("data/filtered/google.csv", header=None).values)
twitter = dict(pd.read_csv("data/filtered/tweets.csv", header=None).drop([0,1], axis=0).values)
euros = dict(pd.read_csv("data/filtered/EUR_USD.csv", header=None).drop([0,1], axis=0).values)
sp500 = dict(pd.read_csv("data/filtered/SP500.csv", header=None).drop([0,1], axis=0).drop([2, 3, 4, 5, 6], axis=1).values)
yen = dict(pd.read_csv("data/filtered/YEN_USD.csv", header=None).drop([0,1], axis=0).values)
#print(bitcoin)


removeDates = []
for date in bitcoin:
  if date not in gTrends and date not in removeDates:
    removeDates.append(date)
  if date not in twitter and date not in removeDates:
    removeDates.append(date)
  if date not in euros and date not in removeDates:
    removeDates.append(date)
  if date not in sp500 and date not in removeDates:
    removeDates.append(date)
  if date not in yen and date not in removeDates:
    removeDates.append(date)
print(removeDates)