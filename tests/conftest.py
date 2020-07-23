import pytest
from pyspark.ml.classification import LogisticRegression

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES, ADULT_TRAIN_DATA_PATH


@pytest.fixture
def preprocessor():
    return _get_preprocessor(TEST_DATA_PATH)


@pytest.fixture
def logistic_model(preprocessor):
    return _fit_logistic(preprocessor)


@pytest.fixture
def preprocessor_train_data():
    return _get_preprocessor(ADULT_TRAIN_DATA_PATH)


@pytest.fixture
def logistic_model_train_data(preprocessor_train_data):
    return _fit_logistic(preprocessor_train_data)


def _get_preprocessor(data_path: str):
    df = DataLoader().load_relative(path=data_path, columns=ADULT_COLUMN_NAMES)
    preprocessor = DataPreprocessor(train_df=df, test_df=df)
    return preprocessor


def _fit_logistic(preprocessor):
    preprocessor.prepare_to_model(target_col='income', to_strip=' .')
    lr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.2)
    fit_model = lr.fit(preprocessor.train_encoded_df)
    return fit_model
