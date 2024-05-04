'''
  chrun_library.py is a library that performs the following functions:
    1. Import data from a csv file
    2. Perform EDA on the data
    3. Encode the categorical columns
    4. Perform feature engineering
    5. Train the models
    6. Store the models
    7. Store the classification reports as images
    8. Store the feature importance plots as images

'''

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import from directory config the module params as cfgp
from config import params as cfgp

sns.set()


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    # readin the file as csv and format it as pandas dataframe
    df = pd.read_csv(pth)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    df.head()
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    logging.info("Performing EDA")

    # genarete a histogram of the churn column for existing and non-existing
    # customers
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig(cfgp.image_eda_path + "/churn_hist.png")
    # generate a histogram of the age column
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig(cfgp.image_eda_path + "/age_hist.png")
    # generate a normalized stat of the income column
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(
        kind="bar", color=["blue", "green", "orange", "red", "purple"]
    )
    plt.savefig(cfgp.image_eda_path + "/marital_status_bar.png")
    # generate a normmlized stats of the education column
    plt.figure(figsize=(20, 10))
    df.Education_Level.value_counts("normalize").plot(
        kind="bar",
        color=[
            "blue",
            "green",
            "orange",
            "red",
            "purple",
            "brown",
            "pink"])
    plt.savefig(cfgp.image_eda_path + "/education_level_bar.png")
    # generate a histogram  of the total transaction count density
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(cfgp.image_eda_path + "/Transaction_count_density.png")
    # generate a heatmap of the correlation between the columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(cfgp.image_eda_path + "/correlation_heatmap.png")
    logging.info(
        "EDA complete. Figures saved to images folder %s", cfgp.image_eda_path)


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string Y columns name

    output:
            df: pandas dataframe with new columns for x values
    """
    logging.info("Start encodeding category columns")

    # for each category in the category list, create a new column with the
    # propotion of churn for each category
    for category in category_lst:
        category_group = df[[category, response]].groupby(category).mean()[
            response]
        df[f"{category}_{response}"] = df[category].map(category_group)

    logging.info("Encoding complete")
    # drop all the original category columns
    df.drop(category_lst, axis=1, inplace=True)
    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string Y columns name

    output:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    """
    logging.info("Start feature engineering")
    # create x and y values
    x_df = pd.DataFrame()
    y_df = pd.DataFrame()
    x_df[cfgp.keep_cols] = df[cfgp.keep_cols]  # extract the columns to keep
    y_df[response] = df[response]
    df = None

    logging.info(x_df.head())

    # split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_df, test_size=0.3, random_state=42
    )
    logging.info("Feature engineering complete")
    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # create a list of dictionaries containing the y values and the y predictions for each model
    # and the filename to save the classification report
    classification_list = [
        {
            "y": y_train,
            "y_preds": y_train_preds_rf,
            "filename": "RandomForestTrainClassificationReport.png",
        },
        {
            "y": y_test,
            "y_preds": y_test_preds_rf,
            "filename": "RandomForestTestClassificationReport.png",
        },
        {
            "y": y_train,
            "y_preds": y_train_preds_lr,
            "filename": "LogisticRegressionTrainClassificationReport.png",
        },
        {
            "y": y_test,
            "y_preds": y_test_preds_lr,
            "filename": "LogisticRegressionTestClassificationReport.png",
        },
    ]

    # for each item in the classification list, create a classification report
    # and save it as an image
    for item in classification_list:
        plt.figure(figsize=(6, 4))
        plt.text(
            0.01,
            0.05,
            str(classification_report(item["y"], item["y_preds"])),
            {"fontsize": 14},
        )
        plt.axis("off")
        plt.grid(False)
        plt.savefig(cfgp.image_results_path + "/" + item["filename"])


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            output_pth: path to store the figure

    output:
             None
    """
    # create a dataframe of the feature importances
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=x_data.columns,
        columns=["importance"])

    # sort the values in the dataframe
    feature_importances = feature_importances.sort_values(
        "importance", ascending=False)

    # create a bar plot of the feature importances
    plt.figure(figsize=(20, 10))
    sns.barplot(
        x=feature_importances.index,
        y=feature_importances["importance"],
        palette="viridis",
    )
    plt.xticks(rotation=90)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # Define the random forest classifier
    rfc = RandomForestClassifier(random_state=42)

    # Define the logistic regression model
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    # Define the grid search parameters for the random forest classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=cfgp.param_grid, cv=5)

    # Fit the random forest classifier , Start with the random forest
    # classifier Training
    logging.info("Training Random Forest Classifier")
    cv_rfc.fit(x_train, y_train)
    logging.info("Completed training of Logistic Regression Classifier")

    # Fit the logistic regression classifier , Start with the logistic
    # regression classifier Training
    logging.info("Training Logistic Regression Classifier")
    lrc.fit(x_train, y_train)
    logging.info("Completed training of Logistic Regression Classifier")

    # save the model to disk
    joblib.dump(cv_rfc.best_estimator_, cfgp.model_path + "/rfc_model.pkl")
    joblib.dump(lrc, cfgp.model_path + "/logistic_model.pkl")

    # get the predictions for the training and testing data for the random
    # forest and logistic regression models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # get the predictions for the training and testing data for the logistic
    # regression model
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # create a classification report for the training and testing data for the
    # random forest and logistic regression models
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # create a feature importance plot for the random forest classifier
    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train,
        cfgp.image_results_path + "/RandomForestFeatureImportance.png",
    )
    return


def main():
    logging.info("Starting Churn Library")
    churn_df = import_data(
        cfgp.data_file_path
    )  # perform eda stats and save to images folder
    EncodedDf = encoder_helper(
        churn_df, cfgp.cat_columns, "Churn"
    )  # perform encoding of the columns cat_columns
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        EncodedDf, "Churn"
    )  # perform feature engineering
    train_models(
        x_train, x_test, y_train, y_test
    )  # train the models and save them to disk

    logging.info("Churn Library completed")


if __name__ == "__main__":
    main()
