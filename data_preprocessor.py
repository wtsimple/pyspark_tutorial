from typing import List

import pyspark.sql.functions as func
from pyspark.sql import DataFrame


class DataPreprocessor(object):
    def __init__(self, train_df: DataFrame = None, test_df: DataFrame = None):
        assert train_df.schema == test_df.schema, "Train and test dataframes must have the same schema"
        self.train_df = train_df
        self.test_df = test_df


    def explore_factors(self):
        """Generates a dictionary of one pandas dataframe per column
        containing all the columns values
        """
        factor_exploration = {}
        for column in self.get_factors():
            join_count = self._get_join_count(column)
            pandas_df = join_count.toPandas()
            pandas_df = pandas_df.set_index(column)
            factor_exploration[column] = self._binarize_pandas(pandas_df, 'in_test', 'in_train')
        return factor_exploration


    def get_factors(self):
        return self._get_cols_by_types(types=['string'])


    def get_numeric_columns(self):
        return self._get_cols_by_types(types=['double', 'int'])


    def _get_join_count(self, column=''):
        """
        Intended for factor exploration, does a group by in the test and train df
        and joins the result by value, hence exposing things like missing values in each
        of the dataframes
        """
        test_c = self.test_df.groupBy(column).count() \
            .orderBy('count', ascending=False).withColumn('in_test', func.col('count'))
        train_c = self.train_df.groupBy(column).count() \
            .orderBy('count', ascending=False).withColumn('in_train', func.col('count'))
        join_count = train_c.join(test_c, [column], how="outer").select(column, 'in_train', 'in_test')
        return join_count


    def _get_cols_by_types(self, types: List[str] = None):
        """Returns a list of columns in the self dataframes given a list of their types

        examples of types are 'int', 'double', and 'string'
        """
        train_cols = [col for col, data_type in self.train_df.dtypes if data_type in types]
        test_cols = [col for col, data_type in self.test_df.dtypes if data_type in types]
        assert train_cols == test_cols, "Columns in train and test df should be the same"
        return train_cols


    @staticmethod
    def _binarize_pandas(pandas_df, *cols):
        """
        Turns a pandas dataframe with numeric columns into 1s and 0s

        threshold is simply value > 0
        """
        for col in cols:
            pandas_df[col] = (pandas_df[col] > 0).astype(int)
        return pandas_df
