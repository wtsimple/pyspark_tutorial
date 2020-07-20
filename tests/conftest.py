import pytest

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES


@pytest.fixture
def preprocessor():
    df = DataLoader().load_relative(path=TEST_DATA_PATH, columns=ADULT_COLUMN_NAMES)
    return DataPreprocessor(train_df=df, test_df=df)
