from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from legacy.suprevised_learning_base import DataPreprocessor, session, ModelEvaluator
from legacy.suprevised_learning_base import ModelTuner


class ClassificationPreprocessor(DataPreprocessor):
    """Makes the dataframe ready for logistic modelling"""
    pass


class ClassificationModelEvaluator(ModelEvaluator):
    def __init__(self):
        super().__init__()
        self.metrics_class = BinaryClassificationMetrics


    def append_eval_row(self, model_type, dataset, metrics):
        if not self.whole_metrics:
            self.whole_metrics = {"model": [], "dataset": [], "AUC_ROC": [], "AUC_PR": []}

        self.whole_metrics['model'].append(model_type)
        self.whole_metrics['dataset'].append(dataset)
        self.whole_metrics['AUC_ROC'].append(metrics.areaUnderROC)
        self.whole_metrics['AUC_PR'].append(metrics.areaUnderPR)


if __name__ == "__main__":
    # Loading
    preprocessor = ClassificationPreprocessor()
    evaluator = ClassificationModelEvaluator()

    adult_df = session.read.csv("/home/armando/machine_learning/datasets/adult.data", header=True, inferSchema=True)
    adult_test = session.read.csv("/home/armando/machine_learning/datasets/adult.test", header=True, inferSchema=True)
    adult_df = preprocessor.clean_string_columns_data(adult_df)
    adult_test = preprocessor.clean_string_columns_data(adult_test)

    # Exploration
    # preprocessor.explore_factors(train_df=adult_df, test_df=adult_test)

    adult_class = preprocessor.fit(adult_df, target_col='income').transform(adult_df)
    adult_test_class = preprocessor.transform(adult_test)

    lr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.2)
    lr_model = lr.fit(adult_class)

    rfc = RandomForestClassifier(labelCol="label", featuresCol="features",
                                 numTrees=10, cacheNodeIds=True,
                                 subsamplingRate=0.7)
    rfc_model = rfc.fit(adult_class)

    features = evaluator.extract_feature_importance(rfc_model.featureImportances, adult_class, 'features')

    # Hyperparameter tuning

    tuner = ModelTuner(evaluator=BinaryClassificationEvaluator())
    specs = {
        'logistic_class': {"regParam": [0], "elasticNetParam": [0]},
        # 'random_forest_class': {'numTrees': [20], "maxDepth": [5], "subsamplingRate": [0.7]},
        'gbt_class': {},
        # 'perceptron_class': {},
        'svm_class': {'regParam': [0], 'standardization': [True]},
        # 'naive_bayes_class': {},
    }
    models = tuner.get_crossvalidated_models(specs, adult_class)

    print(evaluator.evaluation(adult_class, adult_test_class, models=models))

    print('')
