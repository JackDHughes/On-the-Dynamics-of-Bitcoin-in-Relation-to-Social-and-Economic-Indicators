import pandas as pd 
import numpy as np 

tweets = pd.read_csv("data/unfiltered/tweets.csv")

values = []
for i in range(len(tweets)):
    values.append([tweets.iloc[i, 0], tweets.iloc[i, 1]])

pd.DataFrame(values).to_csv("data/filtered/tweets.csv")

print(tweets.head())