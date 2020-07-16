from typing import List

import pyspark.sql.functions as func


class DataPreprocessor(object):
    def __init__(self, train_df=None, test_df=None):
        self.train_df = train_df
        self.test_df = test_df


    def explore_factors(self):
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
        test_c = self.test_df.groupBy(column).count() \
            .orderBy('count', ascending=False).withColumn('in_test', func.col('count'))
        train_c = self.train_df.groupBy(column).count() \
            .orderBy('count', ascending=False).withColumn('in_train', func.col('count'))
        join_count = train_c.join(test_c, [column], how="outer").select(column, 'in_train', 'in_test')
        return join_count


    def _get_cols_by_types(self, types: List[str] = None):
        train_cols = [col for col, data_type in self.train_df.dtypes if data_type in types]
        test_cols = [col for col, data_type in self.test_df.dtypes if data_type in types]
        assert train_cols == test_cols, "Columns in train and test df should be the same"
        return train_cols


    @staticmethod
    def _binarize_pandas(df, *cols):
        for col in cols:
            df[col] = (df[col] > 0).astype(int)
        return df
