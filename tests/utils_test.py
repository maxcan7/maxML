import pytest
from sklearn.preprocessing import StandardScaler

from maxML.utils import get_estimator_fn


def test_get_estimator_fn():
    """
    Test that the get_estimator_fn function can parse a string and load as a function.
    """
    sklearn_module_str = "sklearn.preprocessing.StandardScaler"
    sklearn_module = get_estimator_fn(sklearn_module_str)
    assert sklearn_module == StandardScaler


@pytest.mark.parametrize(
    "module_str, expected_error",
    [
        pytest.param(
            "sklearn.preprocessing.NonExistentClass",
            AttributeError,
            id="invalid_attribute",
        ),
        pytest.param(
            "nonexistent_module.SomeClass",
            ModuleNotFoundError,
            id="invalid_module",
        ),
    ],
)
def test_get_estimator_fn_invalid(module_str: str, expected_error: type) -> None:
    """
    Test that get_estimator_fn raises the appropriate error for invalid module strings.
    """
    with pytest.raises(expected_error):
        get_estimator_fn(module_str)
