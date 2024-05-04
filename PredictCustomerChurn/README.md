# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project is about predicting customer churn for a bank. The data is from a bank and contains information about the customers. The goal is to predict whether a customer will churn or not. The project is divided into 4 parts:
Basicly the projects demonstare the best practices of software engineering and machine learning. The project is divided into 2 main parts:
1. Code Quality , which focus on writing clean and modular code.
2. add documentation to the code to make it easier to understand and maintain.
3. Testing , which focus on writing tests to ensure the code is working as expected.

## progrum flow 
   1. churn_library.py
      1.0 fisrt the config/params.py file is imported to get the configuration parameters
      1.1 load_data from data directory and and reurtn the data frame 
      1.2  endodeing some columns that are not numerical
      1.3 create images of input data statistics 
      1.3  feature engineering to create new features also to split the data into train and test
      1.4 train the model using logistic regression and random forest classifier
      1.5 save the model to models directory
      1.6 create images of model results: classification reports and feature importance plots
    2. churn_script_logging_and_tests.py
        2.0 fisrt the config/params.py file is imported to get the configuration parameters
        2.1 test the functions in churn_library.py using pytest
        2.2 log the results of the tests to logs/results.log file

## Running Files
  To run the project, follow the steps below:
    1. Clone the repository
    2. Install the required libraries using the requirements.txt file
    3. Run the churn_script_logging_and_tests.py file
   ```bash
      python churn_script_logging_and_tests.py
   ```
    4. Run the churn_library.py file
   ```bash 
   ipython churn_library.py
   ```

## Files and data description




```bash
├── Guide.ipynb   # Guide for the project
├── README.md
├── churn_library.py  # Library file main program file
├── churn_notebook.ipynb # Jupyter notebook for the project
├── churn_script_logging_and_tests.py # pytest file including testign of functions from churn_library.py
├── config   
│   ├── __init__.py
│   └── params.py  # Configuration files includes logging configuration and model configuration and paramters 
├── data
│   └── bank_data.csv # Data file 
├── images
│   ├── eda # EDA images containing histograms and bar plots of input data statistics
│   │   ├── Transaction_count_density.png
│   │   ├── Transaction_count_hist.png
│   │   ├── age_hist.png
│   │   ├── churn_hist.png
│   │   ├── education_level_bar.png
│   │   └── marital_status_bar.png
│   └── results # Results images containing classification reports and feature importance plots
│       ├── LogisticRegressionTestClassificationReport.png
│       ├── LogisticRegressionTrainClassificationReport.png
│       ├── RandomForestFeatureImportance.png
│       ├── RandomForestTestClassificationReport.png
│       └── RandomForestTrainClassificationReport.png
├── logs # Logs files generated when you run the churn_script_logging_and_tests.py or churn_library.py
│   └── results.log
├── models # Model files generated when you run the churn_library.py in training function 
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt # Required libraries for python 3.6
├── requirements_py3.8.txt # Required libraries for python 3.8
```

