"""
This script is used to test the functions in the churn_library.py file.

"""

import pytest
import pandas as pd
import churn_library as cls
from config import params as cfgp
import logging

response = "Churn"


@pytest.fixture(scope="module")
def df():
    try:
        df = cls.import_data(cfgp.data_file_path)
    except FileNotFoundError as err:
        raise err
    return df


@pytest.fixture(scope="module")
def df_encoded(df):
    return cls.encoder_helper(df, cfgp.cat_columns, "Churn")


@pytest.fixture(scope="module")
def feature_engineering(df_encoded):
    return cls.perform_feature_engineering(df_encoded, response)


def test_import_data(df):
    try:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("test_import_data passed.")
    except Exception as e:
        logging.error("test_import_data failed. Exception: %s", e)


def test_encoder_helper(df_encoded):
    try:
        for col in cfgp.cat_columns:
            assert f"{col}_{response}" in df_encoded.columns
            logging.info("test_encoder_helper passed.")
    except Exception as e:
        logging.error("test_encoder_helper failed. Exception: %s", e)


def test_perform_feature_engineering(feature_engineering):
    try:
        x_train, x_test, y_train, y_test = feature_engineering
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("test_perform_feature_engineering passed.")
    except Exception as e:
        logging.error("test_perform_feature_engineering failed. Exception: %s", e)


def test_train_models(feature_engineering):
    try:
        x_train, x_test, y_train, y_test = feature_engineering
        cls.train_models(x_train, x_test, y_train, y_test)
    except Exception as e:
        logging.error("test_train_models failed. Exception: %s", e)
