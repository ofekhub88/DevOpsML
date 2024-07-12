"""
This script trains a logistic regression model on 
the data and saves the model to the output folder.
it split the data into training and testing data
drop the columns 'exited' and 'corporation' from the data
and use the logistic regression model to fit the training data
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

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


#################Function for training the model
def train_model():

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    with open(dataset_csv_path + "/" + "finaldata.csv", "r") as f:
        data = pd.read_csv(f)
    X = data.drop(["exited", "corporation"], axis=1)
    y = data["exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model.fit(X_train, y_train)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path + "/" + "trainedmodel.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()