import pandas
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from model_evaluator import ModelEvaluator


def test_model_evaluator_with_linear_regression(logistic_model, preprocessor):
    metrics = BinaryClassificationMetrics
    evaluator = ModelEvaluator(metrics=metrics)
    comparison = evaluator.compare(
        data_frames={
            "train": preprocessor.train_encoded_df,
            "test": preprocessor.test_encoded_df,
        },
        models=[logistic_model])
    assert isinstance(comparison, pandas.DataFrame)
    assert "areaUnderROC" in comparison
    assert "areaUnderPR" in comparison
    assert comparison['areaUnderPR']['train']
