"""
This file contains the function to deploy 
the model into the production environment.
copy the latest pickle file, 
the latestscore.txt value, 
and the ingestfiles.txt file into the deployment directory
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
import config as cfg
import shutil

LOGGER = cfg.get_logger()



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path'])  + "/"
prod_deployment_path = os.path.join(config['prod_deployment_path'])  + "/"
output_model_path = os.path.join(config['output_model_path'])  + "/"



####################function for deployment
def store_model_into_pickle(model):
    LOGGER.info('Storing the model into pickle file')
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(output_model_path + model, prod_deployment_path + 'trainedmodel.pkl')
    shutil.copy(output_model_path + 'latestscore.txt', prod_deployment_path + 'latestscore.txt')
    shutil.copy(output_folder_path + 'ingestedfiles.txt', prod_deployment_path + 'ingestedfiles.txt')
    LOGGER.info('Model stored successfully')