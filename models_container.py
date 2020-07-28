from enum import Enum

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, \
    LinearSVC, NaiveBayes

from spark_launcher import SparkLauncher


class ModelKinds(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Model(object):
    def __init__(self, model, name='', kind=ModelKinds.CLASSIFICATION):
        self.model = model
        self.name = name
        self.kind = kind


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
        return [obj for name, obj in self.__dict__.items()
                if getattr(obj, "kind", None) == ModelKinds.CLASSIFICATION]


    def _wrap_models(self):
        for name, obj in self.__dict__.items():
            if self.model_path in str(obj.__class__):
                wrapped = Model(model=obj, name=name)
                setattr(self, name, wrapped)
