"""
This file contains the parameters used in the project.

"""

import os
import logging

os.environ["QT_QPA_PLATFORM"] = "offscreen"

log_file = "logs/results.log" # log file path

image_eda_path = "images/eda" # image path for plot stats etc ..

image_results_path = "images/results" # image path for model results stats

data_file_path = "./data/bank_data.csv" # input data file path

model_path = "models" # models path

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
# columns to convert to categorical mean encoding and target encoding 
cat_columns = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
# columns with numerical values
quant_columns = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]
# columns used by model
keep_cols = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]
# RandomForestClassifier  paramters
param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt"],
    "max_depth": [4, 5, 100],
    "criterion": ["gini", "entropy"],
}
