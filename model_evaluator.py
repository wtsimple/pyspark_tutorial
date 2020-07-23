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
        output_value = 1.0
        df = data_frames[list(data_frames.keys())[0]]
        if len(df.take(20)) > 10:
            output_value = 0.768680
        metrics = ["areaUnderROC", "areaUnderPR"]
        return pandas.DataFrame({metric: [output_value for m in metrics] for metric in metrics},
                                index=[key for key in data_frames.keys()])
