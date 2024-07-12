import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import json
import numpy as np
import seaborn as sns
import os
import diagnostics as diag
import matplotlib.pyplot as plt

with open("version", "r") as f:
    version = json.load(f)["Latest"]
version = str(version)
###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

output_folder_path = os.path.join(config["output_folder_path"]) + "/"
output_model_path = os.path.join(config["output_model_path"]) + "/"


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    with open(output_folder_path + "finaldata.csv", "r") as f:
        data = pd.read_csv(f)
    Y = data["exited"]
    y_predict = diag.model_predictions(output_folder_path + "finaldata.csv")
    confusion_matrix = metrics.confusion_matrix(Y, y_predict)
    # plot the confusion matrix

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.savefig(output_model_path + f"confusionmatrix{version}.png")


if __name__ == "__main__":
    score_model()
