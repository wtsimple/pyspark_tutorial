import os
import sys

from pyspark.sql import SparkSession


class SparkLauncher(object):
    def __init__(self):
        # Needed to prevent different worker and driver python versions error
        for var in ['PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON']:
            os.environ[var] = sys.executable

        self.session = SparkSession.builder.getOrCreate()
