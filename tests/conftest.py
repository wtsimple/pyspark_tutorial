import pytest
from pyspark.ml.classification import LogisticRegression

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES


@pytest.fixture
def preprocessor():
    df = DataLoader().load_relative(path=TEST_DATA_PATH, columns=ADULT_COLUMN_NAMES)
    return DataPreprocessor(train_df=df, test_df=df)


@pytest.fixture
def logistic_model(preprocessor):
    preprocessor.prepare_to_model(target_col='income', to_strip=' .')
    lr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.2)
    return lr.fit(preprocessor.train_encoded_df)
