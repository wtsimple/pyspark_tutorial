from pyspark.ml.classification import LogisticRegression


def test_fit_logistic_regression(preprocessor):
    preprocessor.prepare_to_model(target_col='income', to_strip=' .')
    lr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.2)
    lr_model = lr.fit(preprocessor.train_encoded_df)
