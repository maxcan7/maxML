from typing import Any
from typing import Optional
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from maxML.config_schemas import EvaluatorConfig


class Evaluator(Protocol):
    @staticmethod
    def run(
        y_test: pd.DataFrame,
        x_test: pd.DataFrame,
        predictions: np.ndarray,
        pipeline: Optional[Pipeline] = None,
    ) -> dict[str, Any]:
        """TODO"""
        ...


class LogisticEvaluator:
    @staticmethod
    def run(
        y_test: pd.DataFrame,
        X_test: pd.DataFrame,
        predictions: np.ndarray,
        pipeline: Pipeline,
    ) -> dict:
        """TODO"""
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        # TODO: Convert to pydantic or dataclass.
        return {
            "accuracy": accuracy,
            "report": report,
            "roc_auc": roc_auc,
        }


class LinearEvaluator:
    @staticmethod
    def run(
        y_test: pd.DataFrame,
        X_test: pd.DataFrame,
        predictions: np.ndarray,
        pipeline: Optional[Pipeline] = None,
    ) -> dict:
        """TODO"""
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        # TODO: Convert to pydantic or dataclass.
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }


EVALUATORS = {
    "LogisticEvaluator": LogisticEvaluator,
    "LinearEvaluator": LinearEvaluator,
}


def get_evaluator_fn(evaluator_config: EvaluatorConfig) -> Evaluator:
    """
    Return Evaluator module as defined in evaluator field in config.
    """
    if not evaluator_config.evaluator:
        raise KeyError("evaluator_config is missing evaluator field.")
    evaluator_type = evaluator_config.evaluator
    return EVALUATORS[evaluator_type]  # type: ignore


def evaluate(
    evaluator_configs: list[EvaluatorConfig],
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
    predictions: np.ndarray,
    pipeline: Optional[Pipeline] = None,
) -> list[dict[str, Any]]:
    """TODO"""
    evaluations = []
    for evaluator_config in evaluator_configs:
        evaluator = get_evaluator_fn(evaluator_config)
        evaluations.append(evaluator.run(y_test, X_test, predictions, pipeline))
    return evaluations


def do_evaluation(evaluator_configs: list[EvaluatorConfig]) -> bool:
    """
    This is used to determine whether or not to do evaluation.
    """
    return any([evaluator_configs[0].evaluator])
