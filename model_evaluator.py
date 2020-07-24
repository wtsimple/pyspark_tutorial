from typing import Dict

import pandas
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame


class ModelEvaluator(object):
    """To encapsulate the production of models metrics
    in particular intended to compare models"""


    def __init__(self, metrics_class=None):
        self.metrics_class = metrics_class if metrics_class else BinaryClassificationMetrics


    def compare(self, data_frames: Dict[str, DataFrame], models: list):
        # per model and per data frame calculate all the metrics
        index = []
        metrics = ["areaUnderROC", "areaUnderPR"]
        data = {metric: [] for metric in metrics}
        for model in models:
            for df_name, df in data_frames.items():
                index.append(self.index_key(df_name, model))
                evaluator = self.metrics_class(model.transform(df).select('prediction', 'label').rdd)
                for metric in metrics:
                    data[metric].append(getattr(evaluator, metric))

        return pandas.DataFrame(data, index=index)


    @staticmethod
    def index_key(df_name, model):
        return model.__class__.__name__ + "_" + df_name
