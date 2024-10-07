import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from maxML.config_schemas import load_config
from maxML.config_schemas import ModelConfig
from maxML.config_schemas import PipelineConfig
from maxML.evaluators import do_evaluation
from maxML.evaluators import evaluate
from maxML.preprocessors import compose_preprocessor
from maxML.preprocessors import do_preprocessing
from maxML.utils import get_estimator_fn


"""
Runner script for an end-to-end sklearn model pipeline using maxML.
"""


def get_y(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Subset dataframe to just target column."""
    return df[target]


def get_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Subset dataframe removing target column."""
    return df.drop(target, axis=1)


def create_model_pipeline(
    model_config: ModelConfig, preprocessor: ColumnTransformer | FeatureUnion = None
) -> Pipeline:
    """
    Creates a scikit-learn Pipeline for a machine learning model.
    The pipeline optionally includes a preprocessing step and always includes
    a model step.
    """
    model_args = model_config.args or {}
    model_module = get_estimator_fn(model_config.module)
    model = model_module(**model_args)
    steps = [("model", model)]
    if preprocessor:
        steps.insert(0, ("preprocessor", preprocessor))
    return Pipeline(steps)


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from input_path as pandas DataFrame."""
    return pd.read_csv(Path.cwd() / input_path)


def run(pipeline_config_path: str) -> None:
    """
    Run sklearn pipelines on dataset with preprocessors steps.
    """
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    df = load_data(pipeline_config.input_path)

    preprocessor = None
    if do_preprocessing(pipeline_config.preprocessors):
        preprocessor = compose_preprocessor(pipeline_config.preprocessors)

    pipeline = create_model_pipeline(
        model_config=pipeline_config.model, preprocessor=preprocessor
    )

    X = get_X(df=df, target=pipeline_config.model.target)
    y = get_y(df=df, target=pipeline_config.model.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **pipeline_config.train_test_split
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"{predictions=}")

    if do_evaluation(pipeline_config.evaluators):
        evaluations = evaluate(
            pipeline_config.evaluators, y_test, X_test, predictions, pipeline
        )
        print(f"{evaluations=}")

    # TODO: Add write (either dump evaluations or integrate with e.g. MLFlow)


if __name__ == "__main__":
    try:
        pipeline_config_path = sys.argv[1]
    except IndexError:
        raise
    run(pipeline_config_path)
