import pandas as pd
import pickle
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import ast
import numpy as np

with open('l3final.pkl', 'rb') as file:
    model = pickle.load(file)


data = pd.read_csv('testdatafinal.csv')
X = data['timeperiod'].values.reshape(-1,1)
#print teh X dimnetions


y = data['sales'].values.reshape(-1,1)

predicted = model.predict(X)

msqscore = mean_squared_error(predicted,y)


with open('l3finalscores.txt','r') as f:
    previosmsq = ast.literal_eval(f.read())

print("lower than min:",msqscore < min(previosmsq))

print("parametric significance outlier pass ?",msqscore < np.mean(previosmsq)-2*np.std(previosmsq),np.std(previosmsq)) 

iqr = np.quantile(previosmsq, 0.75)-np.quantile(previosmsq, 0.25)
print("non-parametric outlier pass?", msqscore < np.quantile(previosmsq, 0.25)-iqr*1.5,iqr)


