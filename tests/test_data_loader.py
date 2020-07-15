import pytest
from pyspark.sql import DataFrame

from data_loader import DataLoader
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES


@pytest.fixture
def data_loader():
    return DataLoader()


def test_data_loader_loads_data_frame(data_loader):
    df = data_loader.load_relative(path=TEST_DATA_PATH, columns=ADULT_COLUMN_NAMES)
    assert isinstance(df, DataFrame)
    assert df.columns == ADULT_COLUMN_NAMES
    # Check some values from the first row
    first_row = df.first()
    assert first_row.income == ' <=50K'
    assert first_row.age == 39
