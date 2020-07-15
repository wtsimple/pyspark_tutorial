import pytest
from pyspark.sql import SparkSession

from spark_launcher import SparkLauncher


@pytest.fixture
def spark():
    return SparkLauncher()


def test_spark_session_is_ok(spark):
    assert isinstance(spark.session, SparkSession)


def test_bug_different_versions_error(spark):
    """
    Test intended to reproduce an error

    Exception: Python in worker has different version 2.7 than that in driver 3.6, PySpark cannot run with different
    minor versions.Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set
    """
    example_cols = ["column1"]
    example_data = [["value1"], ["value2"]]
    df = spark.session.createDataFrame(example_data, example_cols)
    df.show()
