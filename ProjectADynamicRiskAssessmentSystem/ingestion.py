"""
This script performs data ingestion by merging multiple CSV files from a specified input folder
and writing the merged data to a CSV file in the specified output folder.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    global input_folder_path , output_folder_path
    input_folder_path += "/"
    output_folder_path += "/"

    #list all csv file in input_folder_path 
    files = os.listdir(input_folder_path)
    files = [file for file in files if file.endswith('.csv')]

    #read all csv files and merge them
    df = pd.DataFrame()
    csv_files = ""
    for file in files:
        csv_files += file + "/n "
        df = pd.concat([df, pd.read_csv(input_folder_path + file)], ignore_index=True)
        
    with open(output_folder_path + 'ingestedfiles.txt', 'w') as f:
        f.write(csv_files)
    # remove duplicate records 
    df = df.drop_duplicates()
    # write the merged data to a csv file
    df.to_csv(output_folder_path + 'finaldata.csv', index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()