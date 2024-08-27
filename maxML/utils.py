import importlib
from collections.abc import Callable

from sklearn.base import BaseEstimator


def get_estimator_fn(module_str: str) -> Callable[..., BaseEstimator]:
    """
    Parses the module_str, imports the module, and returns the function.
    """
    module_name = ".".join(module_str.split(".")[:-1])
    module_obj = importlib.import_module(module_name)
    function_name = module_str.split(".")[-1]
    return getattr(module_obj, function_name)
