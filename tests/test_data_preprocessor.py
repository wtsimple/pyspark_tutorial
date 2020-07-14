"""The preprocessor is expected to clean and encode the data,
making it ready for modelling"""

import pytest

from data_preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


def test_data_preprocessor_x(preprocessor):
    pass
