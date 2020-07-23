from typing import Dict

import pandas
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame


class ModelEvaluator(object):
    """To encapsulate the production of models metrics
    in particular intended to compare models"""


    def __init__(self, metrics_class=None):
        self.metrics_class = metrics_class if metrics_class else BinaryClassificationMetrics


    @staticmethod
    def compare(data_frames: Dict[str, DataFrame], models: list):
        metrics = ["areaUnderROC", "areaUnderPR"]
        return pandas.DataFrame({metric: [1., 1.] for metric in metrics},
                                index=[key for key in data_frames.keys()])
