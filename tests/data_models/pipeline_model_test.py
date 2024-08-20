import typing as T

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from maxML.data_models.pipeline_model import PipelineModel


def test_retrieve_model_type(logistic_model_config: dict[str, T.Any]):
    pipeline_model = PipelineModel(**logistic_model_config)
    assert isinstance(pipeline_model.sklearn_model, BaseEstimator)
    assert isinstance(pipeline_model.sklearn_model, LogisticRegression)
