import pandas
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from model_evaluator import ModelEvaluator


def test_model_evaluator_with_linear_regression(logistic_model, preprocessor):
    metrics_class = BinaryClassificationMetrics
    evaluator = ModelEvaluator(metrics_class=metrics_class)

    dataframes_sets = [['train', 'test'], ['train1', 'test1']]
    for dataframes in dataframes_sets:
        comparison = evaluator.compare(
            data_frames={dataframe: preprocessor.train_encoded_df for dataframe in dataframes},
            models=[logistic_model])

        assert isinstance(comparison, pandas.DataFrame)

        metrics = ["areaUnderROC", "areaUnderPR"]
        for metric in metrics:
            assert metric in comparison
            for dataframe in dataframes:
                assert comparison[metric][dataframe] >= 0
                assert comparison[metric][dataframe] <= 1
