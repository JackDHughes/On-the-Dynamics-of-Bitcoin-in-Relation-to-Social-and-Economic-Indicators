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
<<<<<<< HEAD



for date in removeDates:
  if date in bitcoin:
    bitcoin.pop(date)
  if date in gTrends:
    gTrends.pop(date)
  if date in twitter:
    twitter.pop(date)
  if date in euros:
    euros.pop(date)
  if date in sp500:
    sp500.pop(date)
  if date in yen:
    yen.pop(date)


finaldata = []
gTrends = list(gTrends.values())
twitter = list(twitter.values())
euros = list(euros.values())
yen = list(yen.values())
sp500 = list(sp500.values())


for i in range(len(bitcoin)):
  finaldata.append([gTrends[i], twitter[i], euros[i], yen[i], sp500[i]])

print(len(twitter))
print((len(gTrends)))
print(len(euros))
print(len(sp500))
print(len(yen))
print(len(bitcoin))
# pd.DataFrame(twitter).to_csv("data/filtered/finalCombined/twitter.csv")
pd.DataFrame(finaldata).to_csv("data/filtered/finalCombined/trainingData.csv")

pd.DataFrame(bitcoin.values()).to_csv("data/filtered/finalCombined/bitcoin.csv")
print(len(bitcoin))
print(len(finaldata))
=======
print(removeDates)
>>>>>>> db38dfd9d5ddc2773d791c10d2c2607eb5973c3d
