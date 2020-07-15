from typing import List


class DataPreprocessor(object):
    def __init__(self, train_df=None, test_df=None):
        self.train_df = train_df
        self.test_df = test_df


    def get_factors(self):
        return self._get_cols_by_types(types=['string'])


    def get_numeric_columns(self):
        return self._get_cols_by_types(types=['double', 'int'])


    def _get_cols_by_types(self, types: List[str] = None):
        return [col for col, data_type in self.train_df.dtypes if data_type in types]
