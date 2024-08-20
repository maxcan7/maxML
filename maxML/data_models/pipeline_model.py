import importlib

from pydantic import BaseModel
from pydantic import field_validator
from sklearn.base import BaseEstimator


class PipelineModel(BaseModel):
    sklearn_model: str
    input_path: str
    target: str
    metrics: list[str]

    @field_validator("sklearn_model", mode="after")
    def retrieve_model_type(sklearn_model: str) -> BaseEstimator:
        """
        Given that BaseEstimator is not supported by pydantic-core, convert
        the sklearn_model str to an sklearn model Estimator through pydantic
        field validation after parsing.
        """
        module_name = ".".join(sklearn_model.split(".")[:-1])
        module_obj = importlib.import_module(module_name)
        function_name = sklearn_model.split(".")[-1]
        return getattr(module_obj, function_name)()
