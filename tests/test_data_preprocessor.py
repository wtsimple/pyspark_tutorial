"""The preprocessor is expected to clean and encode the data,
making it ready for modelling"""

import pytest

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES


@pytest.fixture
def preprocessor():
    df = DataLoader().load_relative(path=TEST_DATA_PATH, columns=ADULT_COLUMN_NAMES)
    return DataPreprocessor(train_df=df, test_df=df)


def test_preprocessor_get_factors(preprocessor):
    # Example 1
    factors_example = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                       'native_country', 'income']
    assert preprocessor.get_factors() == factors_example
    # Example 2 (rename last column)
    preprocessor.train_df = preprocessor.train_df.withColumnRenamed('income', 'income2')
    factors_example[-1] = "income2"
    assert preprocessor.get_factors() == factors_example


def test_preprocessor_get_numeric_columns(preprocessor):
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    assert preprocessor.get_numeric_columns() == numeric_cols
    preprocessor.train_df = preprocessor.train_df.withColumnRenamed('capital_loss', 'capital_super_loss')
    numeric_cols[-2] = 'capital_super_loss'
    assert preprocessor.get_numeric_columns() == numeric_cols


def test_data_preprocessor_explore_factors(preprocessor):
    pass  # preprocessor.explore_factors()
