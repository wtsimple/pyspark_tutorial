from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

from suprevised_learning_base import DataPreprocessor, session, ModelTuner, ModelEvaluator


class RegressionModelEvaluator(ModelEvaluator):
    def __init__(self):
        super().__init__()
        self.metrics_class = RegressionMetrics


    def append_eval_row(self, model_type, dataset, metrics):
        if not self.whole_metrics:
            self.whole_metrics = {"model": [], "dataset": [], "Explained_Var": [],
                                  "Mean_Abs_Er": [], "Mean_Square_Er": [], "RMSE": [], "r2": []}

        self.whole_metrics['model'].append(model_type)
        self.whole_metrics['dataset'].append(dataset)
        self.whole_metrics['Explained_Var'].append(metrics.explainedVariance)
        self.whole_metrics['Mean_Abs_Er'].append(metrics.meanAbsoluteError)
        self.whole_metrics['Mean_Square_Er'].append(metrics.meanSquaredError)
        self.whole_metrics['RMSE'].append(metrics.rootMeanSquaredError)
        self.whole_metrics['r2'].append(metrics.r2)


if __name__ == "__main__":
    # Loading
    preprocessor = DataPreprocessor()
    tuner = ModelTuner(evaluator=RegressionEvaluator())
    evaluator = RegressionModelEvaluator()

    adult_df = session.read.csv("/home/armando/machine_learning/datasets/adult.data", header=True, inferSchema=True)
    adult_test = session.read.csv("/home/armando/machine_learning/datasets/adult.test", header=True, inferSchema=True)
    adult_df = preprocessor.clean_string_columns_data(adult_df)
    adult_test = preprocessor.clean_string_columns_data(adult_test)

    # Exploration
    # preprocessor.explore_factors(train_df=adult_df, test_df=adult_test)

    adult_class = preprocessor.fit(adult_df, target_col='age').transform(adult_df)
    adult_test_class = preprocessor.transform(adult_test)

    specs = {
        'linear_reg': {"regParam": [0], "elasticNetParam": [0]},
        # 'glr_reg': {"regParam": [0]},
        # 'decision_tree_reg': {},  # {'numTrees': [20], "maxDepth": [5], "subsamplingRate": [0.7]},
        # 'random_forest_reg': {'numTrees': [20], "maxDepth": [5], "subsamplingRate": [0.7]},
        # 'gbt_reg': {},
        # 'svm_class': {'regParam': [0], 'standardization': [True]},
    }
    models = tuner.get_crossvalidated_models(specs, adult_class)

    print(evaluator.evaluation(adult_class, adult_test_class, models=models))

    print('')
