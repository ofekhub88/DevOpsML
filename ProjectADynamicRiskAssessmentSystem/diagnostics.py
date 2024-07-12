"""
This script is used to get the model predictions, 
summary statistics, 
missing data, 
execution time, 
and outdated packages.
"""

import pandas as pd
import numpy as np
import timeit
import os
import json
from training import train_model
from ingestion import merge_multiple_dataframe
import pickle
import config as cfg

with open("version", "r") as f:
    version = f.read()
##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

LOGGER = cfg.get_logger()
dataset_csv_path = os.path.join(config["output_folder_path"]) + "/"
test_data_path = os.path.join(config["test_data_path"]) + "/"
output_folder_path = os.path.join(config["output_folder_path"]) + "/"
prod_deployment_path = os.path.join(config["prod_deployment_path"]) + "/"
output_model_path = os.path.join(config["output_model_path"]) + "/"


##################Function to get model predictions
def model_predictions(dataset=dataset_csv_path + "finaldata.csv"):
    LOGGER.info("Getting model predictions")
    # read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path + "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)
    with open(dataset, "r") as f:
        data = pd.read_csv(f)
    X = data.drop(["exited", "corporation"], axis=1)
    y_pred = model.predict(X)
    LOGGER.info(f"Model predictions are ready {y_pred}")
    return y_pred


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    LOGGER.info("Getting summary statistics")
    with open(output_folder_path + "finaldata.csv", "r") as f:
        data = pd.read_csv(f)
    X = data.drop("corporation", axis=1)

    col_stat = {}
    for col in X.columns:
        col_data = X[col]
        col_stat[col] = [
            {"mean": col_data.mean()},
            {"median": col_data.median()},
            {"std": col_data.std()},
        ]
    LOGGER.info(f"Summary statistics are ready {col_stat}")
    return col_stat


########################### Missing Fata
def missing_data():
    # calculate missing data
    LOGGER.info("Getting missing data")
    with open(output_folder_path + "finaldata.csv", "r") as f:
        data = pd.read_csv(f)
    X = data.drop("corporation", axis=1)
    missing_data = (X.isnull().sum() / len(X)) * 100
    LOGGER.info(f"Missing data are ready {missing_data}")
    return missing_data.to_json()


##################Function to get timings
def execution_time():
    # calculate timing for ingestion and training
    LOGGER.info("Getting execution time")

    my_timing = {}
    start_time = timeit.default_timer()
    merge_multiple_dataframe()
    my_timing["ingestion"] = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    train_model()
    my_timing["training"] = timeit.default_timer() - start_time
    LOGGER.info(f"Execution time is ready {my_timing}")
    return my_timing


##################Function to check dependencies
def outdated_packages_list():
    # check for outdated packages
    LOGGER.info("Checking for outdated packages")
    os.system("pip list --outdated > outdated.txt")
    with open("outdated.txt", "r") as f:
        outdated_packages = f.read()
    # read the file requirements.txt
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    # make a json from the requirements packagane name is key and versio is values
    requirements = requirements.split("\n")
    requirements_dict = {}
    for requirement in requirements:
        if requirement:
            package = requirement.split("==")
            requirements_dict[package[0]] = package[1]
    # make the same in outdated.txt
    outdated_packages = outdated_packages.split("\n")
    outdated_packages_dict = {}
    for outdated_package in outdated_packages:
        if outdated_package:
            package = outdated_package.split()
            outdated_packages_dict[package[0]] = package[2]
    outdated_packages = []
    for package in requirements_dict:
        if package in outdated_packages_dict:
            outdated_packages.append(
                {package: [requirements_dict[package], outdated_packages_dict[package]]}
            )
        else:
            outdated_packages.append(
                {package: [requirements_dict[package], "not installed"]}
            )
    LOGGER.info(f"Outdated packages are ready {outdated_packages}")
    return outdated_packages


if __name__ == "__main__":
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
