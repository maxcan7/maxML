from sklearn.preprocessing import StandardScaler

from maxML.utils import get_estimator_fn


def test_get_estimator_fn():
    """
    Test that the get_estimator_fn function can parse a string and load as a function.
    """
    sklearn_module_str = "sklearn.preprocessing.StandardScaler"
    sklearn_module = get_estimator_fn(sklearn_module_str)
    assert sklearn_module == StandardScaler
