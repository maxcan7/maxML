import re
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from maxML.evaluators import do_evaluation
from maxML.evaluators import evaluate
from maxML.evaluators import EvaluatorConfig
from maxML.evaluators import get_evaluator_fn
from maxML.evaluators import LinearEvaluator
from maxML.evaluators import LogisticEvaluator


ModelMixinType = type[ClassifierMixin | RegressorMixin]
EvaluatorType = type[LogisticEvaluator | LinearEvaluator]
y_testType = Union[pd.Series, np.ndarray]
X_testType = Union[pd.DataFrame, np.ndarray]
data_fnType = Callable[..., tuple[X_testType, y_testType]]
predictionsType = type[np.ndarray]
EvaluatorDataType = tuple[y_testType, X_testType, predictionsType, Pipeline]


def get_evaluator_type(evaluator: EvaluatorType) -> str:
    """
    Extracts the evaluator type name from the string of the evaluator class.
    """
    pattern = r"<class 'maxML\.evaluators\.(\w+)'>"
    match = re.search(pattern, str(evaluator))
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract evaluator type from {evaluator}")


def create_model(model_cls: ModelMixinType, **kwargs) -> ModelMixinType:
    """
    Factory to create model instances with arbitrary keyword arguments.
    """
    return model_cls(**kwargs)


def get_evaluator_data(
    data_fn: data_fnType,
    model_cls: ModelMixinType,
    **model_kwargs: Optional[dict[str, Any]],
) -> EvaluatorDataType:
    """
    Helper function to create sample data and pipelines.
    """
    X, y = data_fn(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = create_model(model_cls, **model_kwargs)
    pipeline = Pipeline([("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return y_test, X_test, predictions, pipeline


# Unit tests
@pytest.mark.parametrize(
    "evaluator, data_fn, model_cls, expected_keys, model_kwargs",
    [
        pytest.param(
            LogisticEvaluator,
            make_classification,
            LogisticRegression,
            ["accuracy", "report", "roc_auc"],
            {"random_state": 42},
            id="LogisticEvaluator",
        ),
        pytest.param(
            LinearEvaluator,
            make_regression,
            LinearRegression,
            ["mse", "rmse", "r2"],
            {},
            id="LinearEvaluator",
        ),
    ],
)
def test_evaluators(
    evaluator: EvaluatorType,
    data_fn: data_fnType,
    model_cls: ModelMixinType,
    expected_keys: list[str],
    model_kwargs: Optional[dict[str, Any]],
) -> None:
    if not model_kwargs:
        model_kwargs = {}
    y_test, X_test, predictions, pipeline = get_evaluator_data(
        data_fn, model_cls, **model_kwargs
    )
    results = evaluator.run(y_test, X_test, predictions, pipeline)
    assert isinstance(results, dict)
    for key in expected_keys:
        assert key in results


@pytest.mark.parametrize(
    "evaluator",
    [
        pytest.param(LogisticEvaluator, id="LogisticEvaluator"),
        pytest.param(LinearEvaluator, id="LinearEvaluator"),
    ],
)
def test_get_evaluator_fn(evaluator: EvaluatorType) -> None:
    evaluator_type = get_evaluator_type(evaluator)
    config = EvaluatorConfig(type=evaluator_type)
    evaluator_fn = get_evaluator_fn(config)
    assert evaluator_fn == evaluator


def test_get_evaluator_fn_missing_type() -> None:
    config = EvaluatorConfig()  # Missing type field
    with pytest.raises(KeyError) as excinfo:
        get_evaluator_fn(config)
    assert "evaluator_config is missing type field." in str(excinfo.value)


@pytest.mark.parametrize(
    "evaluator, data_fn, model_cls, model_kwargs",
    [
        pytest.param(
            LogisticEvaluator,
            make_classification,
            LogisticRegression,
            {"random_state": 42},
            id="LogisticEvaluator",
        ),
        pytest.param(
            LinearEvaluator,
            make_regression,
            LinearRegression,
            {},
            id="LinearEvaluator",
        ),
    ],
)
def test_evaluate(
    evaluator: EvaluatorType,
    data_fn: data_fnType,
    model_cls: ModelMixinType,
    model_kwargs: Optional[dict[str, Any]],
) -> None:
    if not model_kwargs:
        model_kwargs = {}
    y_test, X_test, preds, pipeline = get_evaluator_data(
        data_fn, model_cls, **model_kwargs
    )
    evaluator_type = get_evaluator_type(evaluator)
    evaluator_configs = [EvaluatorConfig(type=evaluator_type)]
    results = evaluate(evaluator_configs, y_test, X_test, preds, pipeline)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)


def test_do_evaluation() -> None:
    evaluator_configs: list[EvaluatorConfig] = [
        EvaluatorConfig(type="LogisticEvaluator")
    ]
    assert do_evaluation(evaluator_configs) is True

    evaluator_configs = [EvaluatorConfig()]  # Missing type field
    assert do_evaluation(evaluator_configs) is False
