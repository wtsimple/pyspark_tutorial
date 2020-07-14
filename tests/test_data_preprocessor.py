"""The preprocessor is expected to clean and encode the data,
making it ready for modelling"""

import pytest

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, COLUMN_NAMES


@pytest.fixture
def preprocessor():
    df = DataLoader().load_relative(path=TEST_DATA_PATH, columns=COLUMN_NAMES)
    return DataPreprocessor(train_df=df, test_df=df)


def test_data_preprocessor_explore_factors(preprocessor):
    pass
