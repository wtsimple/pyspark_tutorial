"""The preprocessor is expected to clean and encode the data,
making it ready for modelling"""
import pandas
import pytest

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from settings import TEST_DATA_PATH, ADULT_COLUMN_NAMES
from spark_launcher import SparkLauncher


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
    renamed = preprocessor.train_df.withColumnRenamed('income', 'income2')
    preprocessor.train_df = renamed
    preprocessor.test_df = renamed
    factors_example[-1] = "income2"
    assert preprocessor.get_factors() == factors_example


def test_preprocessor_get_numeric_columns(preprocessor):
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    assert preprocessor.get_numeric_columns() == numeric_cols
    renamed = preprocessor.train_df.withColumnRenamed('capital_loss', 'capital_super_loss')
    preprocessor.train_df = renamed
    preprocessor.test_df = renamed
    numeric_cols[-2] = 'capital_super_loss'
    assert preprocessor.get_numeric_columns() == numeric_cols


def test_data_preprocessor_explore_factors():
    example_train = [["value1"], ["value2"], ["value1"], ["value3"]]
    example_test = [["value1"], ["value2"], ["value1"]]
    exp_df = _create_exploration_df(example_test, example_train, is_numeric=False)

    assert isinstance(exp_df, pandas.DataFrame)
    assert exp_df.shape == (3, 2)
    assert exp_df.in_train['value1'] == 1
    assert exp_df.in_test['value3'] == 0


def test_exploring_numeric_columns():
    example_train = [[99.], [25.], [-17.3], [81.47]]
    example_test = [[33.], [45.14], [1.]]
    exp_df = _create_exploration_df(example_test, example_train, is_numeric=True)
    assert exp_df.train['count'] == pytest.approx(4)
    assert exp_df.test['count'] == pytest.approx(3)
    assert exp_df.train['max'] == pytest.approx(99.0)
    assert exp_df.test['max'] == pytest.approx(45.14)


def _create_exploration_df(example_test, example_train, is_numeric=False):
    example_cols = ["column1"]
    test_df, train_df = _create_testing_dataframes(example_cols, example_test, example_train)
    prep = DataPreprocessor(train_df=train_df, test_df=test_df)
    if is_numeric:
        factor_exploration = prep.explore_numeric_columns()
    else:
        factor_exploration = prep.explore_factors()

    assert len(factor_exploration) == 1
    return factor_exploration["column1"]


def _create_testing_dataframes(example_cols, data_test, data_train):
    train_df = SparkLauncher().session.createDataFrame(data_train, example_cols)
    test_df = SparkLauncher().session.createDataFrame(data_test, example_cols)
    return test_df, train_df
