from enum import Enum

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, \
    LinearSVC, NaiveBayes
from pyspark.sql import DataFrame

from spark_launcher import SparkLauncher


class ModelKinds(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Model(object):
    def __init__(self, model, name='', kind=ModelKinds.CLASSIFICATION):
        self.model = model
        self.name = name
        self.kind = kind
        self.fitted_model = None


class ModelsContainer(object):
    model_path = "pyspark.ml"


    def __init__(self):
        self.spark = SparkLauncher()
        # Models
        self.logistic_class = LogisticRegression(maxIter=20)
        self.random_forest_class = RandomForestClassifier(cacheNodeIds=True)
        self.gbt_class = GBTClassifier(cacheNodeIds=True)
        self.svm_class = LinearSVC(maxIter=20)
        self.naive_bayes_class = NaiveBayes()

        self._wrap_models()


    @property
    def classification(self):
        """Returns the classification models"""
        return self._get_models_of_kind(kind=ModelKinds.CLASSIFICATION)


    def fit(self, data: DataFrame, kind="*"):
        """Loops though all models of some kind and generates fitted models"""
        if kind == "*":
            models = self._all_models_dict.values()
        else:
            models = self._get_models_of_kind(kind)

        for model in models:
            model.fitted_model = model.model.fit(data)


    @property
    def fitted_models(self):
        return [model.fitted_model for model in self._all_models_dict.values()]


    def _wrap_models(self):
        """Wraps the pyspark model in our own Model class that
        provides some metadata and perhaps extra functionality"""
        for name, obj in self._all_models_dict.items():
            wrapped = Model(model=obj, name=name)
            setattr(self, name, wrapped)


    @property
    def _all_models_dict(self):
        return {name: obj for name, obj in self.__dict__.items()
                if self.model_path in str(obj.__class__) or isinstance(obj, Model)}


    def _get_models_of_kind(self, kind):
        return [obj for name, obj in self.__dict__.items()
                if getattr(obj, "kind", None) == kind]
