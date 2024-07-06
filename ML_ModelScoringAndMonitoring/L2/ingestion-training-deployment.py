import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import os



trainingdata = pd.read_csv("sales.csv")

X=trainingdata.loc[:,['timeperiod']].values.reshape(-1, 1)
y=trainingdata['sales'].values.reshape(-1, 1)

a = LinearRegression()            
model = a.fit(X, y)



############Pushing to Production###################

pickle.dump(model, open('./production/' + "LinearRegression.pkl", 'wb'))






