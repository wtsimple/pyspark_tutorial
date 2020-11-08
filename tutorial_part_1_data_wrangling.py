# Before you get started remember to do
# pip install pyspark
# (is recommended to do it on a virtual environment)
import math
import os
import sys
from time import time

import pyspark.sql.functions as funcs
import pyspark.sql.types as types
from pyspark.sql import SparkSession

# without these environment variables Spark might complain
# the python version at your driver is different from the one
# at your workers
environment = ['PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON']
for var in environment:
    os.environ[var] = sys.executable

if __name__ == "__main__":
    print('--- Starting the session ---- ')
    session = SparkSession.builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")

    print('--- RDD example creation ---- ')
    rdd = session.sparkContext.parallelize([1, 2, 3])
    # gets the first two values
    print(rdd.take(num=2))
    # gets the length of the rdd
    print(rdd.count())
    # gets the full rdd. WATCH OUT, THIS COULD MAKE THE DRIVER COLLAPSE IF THE RDD IS BIG
    print(rdd.collect())

    print('--- Dataframe example creation --- ')
    df = session.createDataFrame([[1, 2, 3], [4, 5, 6]],
                                 ['column1', 'column2', 'column3'])
    # you can use all the RDD methods shown before, but also
    df.show(3)

    print(' --- Immutability ----')
    # lets reuse our RDD
    my_rdd = rdd
    # when you execute map, you don't transform my_rdd,
    # instead you create a new RDD
    my_rdd.map(lambda x: x * 100)
    print(my_rdd.take(3))
    # The my_rdd object is immutable, but you can point the 'my_rdd' name
    # to the newly create, transformed RDD
    my_rdd = my_rdd.map(lambda x: x * 100)
    print(my_rdd.take(3))

    print('---- Transformations and actions -----')
    # create a bigger RDD
    big_rdd = session.sparkContext.parallelize(range(1, 10 ** 7))
    # perform a transformation
    time_before = time()
    big_rdd = big_rdd.map(lambda x: math.log(x) + math.sqrt(x) + x ** 2)
    time_after = time()
    print("Transformations are lazy-loaded so they look instant. Time: ",
          time_after - time_before)

    # do a seemingly innocent action, just take the first value
    time_before = time()
    print(big_rdd.take(1))
    time_after = time()
    print(
        "When you perform an action is when the transformations "
        "really happen. Time: ", time_after - time_before
    )

    print('---- Dataframe transformations -----')

    # create a function
    def multiply_by_ten(number):
        return number * 10.0

    # use the function to create an UDF
    multiply_udf = funcs.udf(multiply_by_ten, types.DoubleType())
    # this will create a new DF with an added column 'multiplied', build by applying the UDF to 'column1'
    transformed_df = df.withColumn('multiplied', multiply_udf('column1'))
    transformed_df.show()

    def take_log_in_all_columns(row: types.Row):
        old_row = row.asDict()
        new_row = {f'log({column_name})': math.log(value)
                   for column_name, value in old_row.items()}
        return types.Row(**new_row)

    logarithmic_dataframe = df.rdd.map(take_log_in_all_columns).toDF()
    logarithmic_dataframe.show()

    print('---- Running SQL queries ----')
    # you create a view from your dataframe
    df.createOrReplaceTempView("table1")
    # call queries over that view, you'll get a new dataframe as result
    df2 = session.sql("SELECT column1 AS f1, column2 as f2 from table1")
    df2.show()

    print('---- Direct Column operations -------')
    df3 = df.withColumn('derived_column',
                        df['column1'] + df['column2'] * df['column3'])
    df3.show()

    print('--- Aggregations and quick statistics -------')
    # the dataframe doesn't have headers that's why we need the column names
    ADULT_COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
    ]
    # I downloaded the adult.data file from
    # https://archive.ics.uci.edu/ml/datasets/adult
    #  to my data folder and renamed it to adult.data.csv
    csv_df = session.read.csv(
        'data/adult.data.csv', header=False, inferSchema=True
    )
    # we'll set the column names one by one on this loop
    for new_col, old_col in zip(ADULT_COLUMN_NAMES, csv_df.columns):
        csv_df = csv_df.withColumnRenamed(old_col, new_col)

    # quick descriptive statistics
    csv_df.describe().show()
    # get average work hours per age
    work_hours_df = csv_df.groupBy('age') \
        .agg(funcs.avg('hours_per_week'), funcs.stddev_samp('hours_per_week')) \
        .sort('age')
    work_hours_df.show(100)

    print('---- The End :) -----')
