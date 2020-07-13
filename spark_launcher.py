from pyspark.sql import SparkSession


class SparkLauncher(object):
    def __init__(self):
        self.session = SparkSession.builder.getOrCreate()