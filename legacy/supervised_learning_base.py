import os
import re
import sys

import pandas as pd
import pyspark.sql.functions as func
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, \
    RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import SparkSession

environment = ['PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON']
for var in environment:
    os.environ[var] = sys.executable

conf = SparkConf()
# conf.set('spark.jars', "/home/share/postgresql-42.2.12.jar")
# conf.set('spark.driver.extraClassPath', "/home/share/postgresql-42.2.12.jar")
session = SparkSession.builder.config(conf=conf).getOrCreate()
session.sparkContext.setLogLevel("ERROR")


class DataPreprocessor(object):
    """Makes the dataframe ready for logistic modelling"""


    def __init__(self):
        self.factor_cols = []
        self.numeric_cols = []
        self.indexed_vectors = []
        self.indexed_factors = []
        self.target_col = ''
        self.pipeline = Pipeline()
        self.encoder = OneHotEncoderEstimator()


    @staticmethod
    def clean_column_names(df):
        """Standardizes columns names"""
        for col_name in df.columns:
            new_col_name = re.sub(r"[^a-z0-9]+", "_", col_name.strip())
            new_col_name = new_col_name.strip("_")
            df = df.withColumnRenamed(col_name, new_col_name)
        return df


    def clean_string_columns_data(self, df, to_strip='. '):
        """Does some preprocessing on string columns"""
        out_df = self.clean_column_names(df)
        # strip string columns
        string_cols = [col for col, data_type in out_df.dtypes if data_type == 'string']
        strip_col = func.udf(lambda x: x.strip(to_strip))
        for col in string_cols:
            out_df = out_df.withColumn(col, strip_col(col))

        return out_df


    def fit(self, input_df, target_col='income'):
        print("Preprocessor Fit")
        self.target_col = target_col
        fitted_df = self.clean_string_columns_data(input_df)

        self.__create_columns_names(fitted_df)

        self.pipeline = Pipeline(
            stages=[StringIndexer(inputCol=factor, outputCol=factor + "_cat") for factor in self.factor_cols])

        self.pipeline = self.pipeline.fit(fitted_df)
        string_encoded = self.pipeline.transform(fitted_df)

        self.encoder = OneHotEncoderEstimator(inputCols=self.indexed_factors, outputCols=self.indexed_vectors)
        self.encoder = self.encoder.fit(string_encoded)

        return self


    def __create_columns_names(self, input_df):
        self.factor_cols = [col for col, data_type in input_df.dtypes if data_type == 'string']
        self.numeric_cols = [col for col, data_type in input_df.dtypes if data_type in ['double', 'int']
                             if col != self.target_col]
        self.indexed_factors = [factor + "_cat" for factor in self.factor_cols if factor != self.target_col]
        self.indexed_vectors = [fac + "_vec" for fac in self.indexed_factors]


    def transform(self, input_df):
        print("Preprocessor Transform")
        transformed_df = self.clean_string_columns_data(input_df)

        string_encoded = self.pipeline.transform(transformed_df)
        one_hot_encoded = self.encoder.transform(string_encoded)

        v_assembler = VectorAssembler(inputCols=self.numeric_cols + self.indexed_vectors, outputCol='features')
        ready_to_model_df = v_assembler.transform(one_hot_encoded)

        final_target_col = self.target_col + ("_cat" if self.target_col + "_cat" in ready_to_model_df.columns else "")
        ready_to_model_df = ready_to_model_df \
            .select('features', final_target_col) \
            .withColumnRenamed(final_target_col, 'label')
        # Make sure the label column is a double, this might cause errors otherwise
        ready_to_model_df = ready_to_model_df.withColumn('label', ready_to_model_df['label'].cast("double"))
        return ready_to_model_df


    @staticmethod
    def explore_factors(train_df, test_df):
        for column, dtype in train_df.dtypes:
            if dtype == 'string':
                train_df.groupBy(column).count().orderBy('count', ascending=False)
                test_c = test_df.groupBy(column).count() \
                    .orderBy('count', ascending=False).withColumn('test_count', func.col('count'))
                train_c = train_df.groupBy(column).count() \
                    .orderBy('count', ascending=False).withColumn('train_count', func.col('count'))
                train_c.join(test_c, [column], how="outer").select(column, 'train_count', 'test_count').show()


class ModelEvaluator(object):

    def __init__(self):
        self.metrics_class = BinaryClassificationMetrics
        self.whole_metrics = {}


    def evaluation(self, train_df, test_df=None, models: list = None):

        for model in models:
            model_type = getattr(model, "display_name", model.__class__.__name__)
            train_metrics = self.metrics_class(model.transform(train_df).select('prediction', 'label').rdd)
            self.append_eval_row(model_type, 'train', train_metrics)
            if test_df:
                test_metrics = self.metrics_class(model.transform(test_df).select('prediction', 'label').rdd)
                self.append_eval_row(model_type, 'test', test_metrics)

        # The random guessing only makes sense in classification
        if "Classification" in self.metrics_class.__class__.__name__:
            self.append_random_metrics(test_df, train_df)

        return pd.DataFrame(self.whole_metrics)


    def append_eval_row(self, model_type, dataset, metrics):
        """Needs to be overridden by children classes"""
        pass


    def append_random_metrics(self, test_df, train_df):
        train_random_metrics = self.random_metrics(train_df)
        self.append_eval_row("Random", 'train', train_random_metrics)
        if test_df:
            test_random_metrics = self.random_metrics(test_df)
            self.append_eval_row("Random", 'test', test_random_metrics)


    @staticmethod
    def random_metrics(train_df):
        train_random_metrics = BinaryClassificationMetrics(
            train_df.withColumn(
                'prediction', func.when(func.rand() > 0.5, 1.0).otherwise(0.0)).select('prediction', 'label').rdd
        )
        return train_random_metrics


    @staticmethod
    def extract_feature_importance(feature_imp, dataset, features_col='features'):
        list_extract = []
        for i in dataset.schema[features_col].metadata["ml_attr"]["attrs"]:
            list_extract = list_extract + dataset.schema[features_col].metadata["ml_attr"]["attrs"][i]

        var_list = pd.DataFrame(list_extract)
        var_list['score'] = var_list['idx'].apply(lambda x: feature_imp[x])

        return var_list.sort_values('score', ascending=False)


class ModelTuner(object):
    def __init__(self, evaluator=None):
        # Evaluator
        if evaluator is None:
            self.evaluator = BinaryClassificationEvaluator()
        else:
            self.evaluator = evaluator

        # Classification Models
        self.logistic_class = LogisticRegression(maxIter=20)
        self.random_forest_class = RandomForestClassifier(cacheNodeIds=True)
        self.gbt_class = GBTClassifier(cacheNodeIds=True)
        # For the Perceptron it's important to set up the layers
        # the first layer correspond to the amount of features and the last to the amount of classes
        self.perceptron_class = MultilayerPerceptronClassifier(layers=[100, 5, 2])
        self.svm_class = LinearSVC(maxIter=20)
        self.naive_bayes_class = NaiveBayes()
        # Regression Models
        self.linear_reg = LinearRegression(maxIter=20, regParam=0.1, elasticNetParam=0.1)
        self.glr_reg = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
        self.decision_tree_reg = DecisionTreeRegressor(cacheNodeIds=True)
        self.random_forest_reg = RandomForestRegressor(cacheNodeIds=True)
        self.gbt_reg = GBTRegressor(cacheNodeIds=True)


    def get_crossvalidated_models(self, specs: dict = None, train_df=None):
        tuned_models = []
        for model_name in specs:
            model_trainer = getattr(self, model_name)
            # Create the parameters grid
            param_grid = ParamGridBuilder()
            for param, values in specs[model_name].items():
                param_grid.addGrid(getattr(model_trainer, param), values)
            param_grid = param_grid.build()

            crossval = CrossValidator(
                estimator=model_trainer,
                estimatorParamMaps=param_grid,
                evaluator=self.evaluator,
                numFolds=4
            )
            print("Training", model_name)
            tuned_model = crossval.fit(train_df)
            tuned_model.display_name = model_name
            tuned_models.append(tuned_model)

        return tuned_models
