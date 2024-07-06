"""
This script is used to score 
the model using the test data and write 
the result to the latestscore.txt file.
it use the trained model to predict the test data
and calculate the f1 score of the model

"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_model_path']) + "/"
test_data_path = os.path.join(config['test_data_path']) + "/"


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(dataset_csv_path + 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(test_data_path + 'testdata.csv', 'r') as f:
        data = pd.read_csv(f)
    X = data.drop(['exited', 'corporation'], axis=1)
    y = data['exited']
    y_pred = model.predict(X)
    f1score = metrics.f1_score(y, y_pred)
    f1score = round(f1score, 2)
    print(f1score)
    with open(dataset_csv_path + 'latestscore.txt', 'a') as f:
        f.write(str(f1score))

