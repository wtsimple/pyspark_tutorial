import pytest
from pyspark.sql import DataFrame

from data_loader import DataLoader

COLUMN_NAMES = ["age",
                "workclass",
                "fnlwgt",
                "education",
                "education_num",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
                "native_country",
                "income"]


@pytest.fixture
def data_loader():
    return DataLoader()


def test_data_loader_loads_data(data_loader):
    df = data_loader.load_relative(path="data/data_example_for_tests.csv", columns=COLUMN_NAMES)
    assert isinstance(df, DataFrame)
    assert df.columns == COLUMN_NAMES
    # Check some values from the first row
    first_row = df.first()
    assert first_row.income == ' <=50K'
    assert first_row.age == 39
