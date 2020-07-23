from typing import Dict

import pandas
import pytest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from model_evaluator import ModelEvaluator

"""Expected results with real data
    model           dataset   AUC_ROC  AUC_PR
0  logistic_class   train  0.768680  0.640418
1  logistic_class    test  0.765914  0.632130
"""

CLASSIFICATION_METRICS = ["areaUnderROC", "areaUnderPR"]


def test_model_evaluator_with_linear_regression(logistic_model, preprocessor):
    _check_evaluation(preprocessor=preprocessor, model=logistic_model,
                      metrics={"areaUnderROC": 1., "areaUnderPR": 1.})


def test_model_evaluator_with_linear_regression_and_full_train_data(logistic_model_train_data, preprocessor_train_data):
    _check_evaluation(preprocessor=preprocessor_train_data,
                      model=logistic_model_train_data, metrics={"areaUnderROC": 0.768680, "areaUnderPR": 0.640418})


def _check_evaluation(preprocessor, model, metrics: Dict[str, float]):
    metrics_class = BinaryClassificationMetrics
    evaluator = ModelEvaluator(metrics_class=metrics_class)
    dataframes_sets = [['train', 'test'], ['train1', 'test1']]
    for dataframes in dataframes_sets:
        comparison = evaluator.compare(
            data_frames={dataframe: preprocessor.train_encoded_df for dataframe in dataframes},
            models=[model])

        assert isinstance(comparison, pandas.DataFrame)

        for metric in metrics:
            assert metric in comparison
            for dataframe in dataframes:
                assert comparison[metric][dataframe] == pytest.approx(metrics[metric])
