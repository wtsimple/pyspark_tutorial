from typing import Dict

import pandas
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame


class ModelEvaluator(object):
    """To encapsulate the production of models metrics
    in particular intended to compare models"""


    def __init__(self, metrics=None):
        self.metrics = metrics if metrics else BinaryClassificationMetrics


    def compare(self, data_frames: Dict[str, DataFrame], models: list):
        return pandas.DataFrame()
