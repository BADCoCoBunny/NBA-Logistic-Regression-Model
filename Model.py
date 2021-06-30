import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def binaryConverter(): # It converts True/False values to 1/0
    arr = df["W/L"].values
    sub_arr = df["W/L"].values

    for x, i in enumerate(arr):
        if i == "W":
            sub_arr[x] = 1
        else:
            sub_arr[x] = 0

df = pd.read_csv("NBA 2017-2018 Data.csv") # Read the csv archive and safe the information on the df variable that is a DataFrame type
binaryConverter()
#plt.scatter(df["PTS"], df["3PM"])

#X = df[["TEAM", "MATCHUP", "PTS",
#"3PM", "OREB", "DREB", "REB", "AST",
#"STL", "BLK"]].values

X = df[["MIN", "PTS", "3PM", "OREB",
    "DREB", "REB", "AST", "STL",
    "BLK"]].values
y = df["W/L"].values
y=y.astype('int')


model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(y_pred[:5])
print(y[:5])


print(model.score(X, y))
print("accuracy:", accuracy_score(y, y_pred))
print("precision:", precision_score(y, y_pred))
print("recall:", recall_score(y, y_pred))
print("f1 score:", f1_score(y, y_pred))
"""
TEAM: Team name
MATCHUP: Team vs who?
W/L: Winner or Losser
"""