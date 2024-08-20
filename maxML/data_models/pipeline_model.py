import importlib

from pydantic import BaseModel
from pydantic import field_validator
from sklearn.base import BaseEstimator


class PipelineModel(BaseModel):
    model_type: BaseEstimator
    input_path: str
    target: str
    metrics: dict[str, list[str]]

    @field_validator(mode="before")
    def retrieve_model_type(model_type: str) -> BaseEstimator:
        module_name = ".".join(model_type.split(".")[:-1])
        module_obj = importlib.import_module(module_name)
        function_name = model_type.split(".")[-1]
        return getattr(module_obj, function_name)()
