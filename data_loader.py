import inspect
import os

from spark_launcher import SparkLauncher


class DataLoader(object):
    def __init__(self):
        self.spark = SparkLauncher()


    def load_relative(self, path='', columns=None):
        absolute_path = self._get_absolute_path(path)
        df = self.spark.session.read.csv(absolute_path, header=False, inferSchema=True)
        if columns:
            df = self._rename_columns(columns, df)
        return df


    @staticmethod
    def _rename_columns(columns, df):
        for new_col, old_col in zip(columns, df.columns):
            df = df.withColumnRenamed(old_col, new_col)
        return df


    def _get_absolute_path(self, relative_path=''):
        current_file = inspect.getfile(self.__class__)
        directory = os.path.dirname(current_file)
        file_name = os.path.join(
            directory, relative_path)
        return file_name
