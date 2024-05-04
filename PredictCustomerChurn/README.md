# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project is about predicting customer churn for a bank. The data is from a bank and contains information about the customers. The goal is to predict whether a customer will churn or not. The project is divided into 4 parts:
Basicly the projects demonstare the best practices of software engineering and machine learning. The project is divided into 2 main parts:
1. Coee Quality , which focus on writing clean and modular code.
2. Testing , which focus on writing tests to ensure the code is working as expected.
   
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
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── config
│   ├── __init__.py
│   └── params.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── Transaction_count_density.png
│   │   ├── Transaction_count_hist.png
│   │   ├── age_hist.png
│   │   ├── churn_hist.png
│   │   ├── education_level_bar.png
│   │   └── marital_status_bar.png
│   └── results
│       ├── LogisticRegressionTestClassificationReport.png
│       ├── LogisticRegressionTrainClassificationReport.png
│       ├── RandomForestFeatureImportance.png
│       ├── RandomForestTestClassificationReport.png
│       └── RandomForestTrainClassificationReport.png
├── logs
│   └── results.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt
├── requirements_py3.8.txt
```

