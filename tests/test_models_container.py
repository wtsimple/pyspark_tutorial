from models_container import ModelsContainer


def test_example_models_init():
    ModelsContainer()


def test_get_classification_models():
    names = [model.name for model in ModelsContainer().classification]
    assert set(names) == {"logistic_class",
                          "random_forest_class",
                          "gbt_class",
                          "svm_class",
                          "naive_bayes_class"}
