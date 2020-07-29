from typing import Dict

import pandas
import pytest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from data_preprocessor import DataPreprocessor
from model_evaluator import ModelEvaluator
from models_container import ModelsContainer, ModelKinds

"""Expected results with real data
    model           dataset   AUC_ROC  AUC_PR
0  logistic_class   train  0.768680  0.640418
1  logistic_class    test  0.765914  0.632130
"""

CLASSIFICATION_METRICS = ["areaUnderROC", "areaUnderPR"]


def test_model_evaluator_with_linear_regression_and_tiny_dataset(logistic_model, preprocessor):
    _check_evaluation(preprocessor=preprocessor, model=logistic_model,
                      metrics={"areaUnderROC": 1., "areaUnderPR": 1.})


def test_model_evaluator_with_linear_regression_and_full_train_data(logistic_model_train_data, preprocessor_train_data):
    _check_evaluation(preprocessor=preprocessor_train_data,
                      model=logistic_model_train_data,
                      metrics={"areaUnderROC": 0.764655781, "areaUnderPR": 0.63384702449})


def test_several_classification_models_fitting(preprocessor_train_data):
    df = preprocessor_train_data.train_df.sample(0.1)
    preprocessor = DataPreprocessor(train_df=df, test_df=df)
    preprocessor.prepare_to_model(target_col='income', to_strip=' .')

    models = ModelsContainer()
    models.fit(preprocessor.train_encoded_df, kind=ModelKinds.CLASSIFICATION)
    expected_results = [
        {"model": models.logistic_class.fitted_model,
         "metrics": {"areaUnderROC": 0.770414, "areaUnderPR": 0.646093}, },
        {"model": models.random_forest_class.fitted_model,
         "metrics": {"areaUnderROC": 0.674751, "areaUnderPR": 0.664931}, },
        {"model": models.gbt_class.fitted_model,
         "metrics": {"areaUnderROC": 0.811643, "areaUnderPR": 0.746147}, },
        {"model": models.svm_class.fitted_model,
         "metrics": {"areaUnderROC": 0.750627, "areaUnderPR": 0.645328}, },
        {"model": models.naive_bayes_class.fitted_model,
         "metrics": {"areaUnderROC": 0.615000, "areaUnderPR": 0.504709}, },
    ]
    for result in expected_results:
        _check_evaluation(preprocessor=preprocessor, model=result["model"], metrics=result["metrics"])


def _check_evaluation(preprocessor, model, metrics: Dict[str, float]):
    evaluator = ModelEvaluator(metrics_class=BinaryClassificationMetrics)
    # The purpose of this parameter is to prove names can be arbitrary in the compare method
    dataframes_sets = [['train', 'test'], ['train1', 'test1']]
    for dataframes in dataframes_sets:
        comparison = evaluator.compare(
            data_frames={dataframe: preprocessor.train_encoded_df for dataframe in dataframes},
            models=[model])

        assert isinstance(comparison, pandas.DataFrame)

        for metric in metrics:
            assert metric in comparison
            for dataframe in dataframes:
                assert comparison[metric][evaluator.index_key(dataframe, model)] == pytest.approx(metrics[metric],
                                                                                                  abs=0.035)
