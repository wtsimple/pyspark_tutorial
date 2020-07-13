import pytest
from pyspark.sql import SparkSession

from spark_launcher import SparkLauncher


@pytest.fixture
def spark():
    return SparkLauncher()


def test_spark_session_is_ok(spark):
    assert isinstance(spark.session, SparkSession)
