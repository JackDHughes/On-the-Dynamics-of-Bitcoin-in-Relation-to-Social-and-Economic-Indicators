import pandas as pd
import numpy as np

bitcoin = pd.read_csv("data/unfiltered/bitcoin.csv")

values = []
for i in range(len(bitcoin)):
  values.append([bitcoin.iloc[i, 0], int(str(bitcoin.iloc[i,2]).replace(",", ""))])


pd.DataFrame(values).to_csv("data/filtered/bitcoin.csv")

print(bitcoin.head())