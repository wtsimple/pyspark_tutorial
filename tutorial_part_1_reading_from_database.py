from pyspark.sql import SparkSession

if __name__ == '__main__':
    print('--- Database connection ----')

    session = SparkSession.builder.config(
        'spark.jars', 'bin/postgresql-42.2.16.jar'
    ).config(
        'spark.driver.extraClassPath', 'bin/postgresql-42.2.16.jar'
    ).getOrCreate()

    url = f"jdbc:postgresql://your_host_ip:5432/your_database"
    properties = {'user': 'your_user', 'password': 'your_password'}
    # read from a table into a dataframe
    df = session.read.jdbc(
        url=url, table='your_table_name', properties=properties
    )
    # you can build a transformed dataframe
    transformed_df = df.withColumn('new_column', 'old_column')
    # then you can save it to another table (or the same)
    # modes according to documentation
    # :param mode: specifies the behavior of the save operation
    #  when data already exists.
    #
    #     * ``append``: Append contents of this :class:`DataFrame` to existing data.
    #     * ``overwrite``: Overwrite existing data.
    #     * ``ignore``: Silently ignore this operation if data already exists.
    #     * ``error`` or ``errorifexists`` (default case): Throw an exception if data already \
    #         exists.
    transformed_df.write.jdbc(
        url=url, table='new_table', mode='append', properties=properties
    )
