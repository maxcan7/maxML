import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.evaluators import do_evaluation
from maxML.evaluators import evaluate
from maxML.preprocessors import compose_preprocessor
from maxML.preprocessors import do_preprocessing


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
    model: BaseEstimator, preprocessor: ColumnTransformer = None
) -> Pipeline:
    """TODO: Configure"""
    if not preprocessor:
        return Pipeline([("model", model)])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from input_path as pandas DataFrame."""
    return pd.read_csv(Path.cwd() / input_path)


def run(pipeline_config_path: str) -> None:
    """
    Run sklearn pipelines on dataset with preprocessors steps.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    df = load_data(pipeline_config.input_path)

    preprocessor = None
    if do_preprocessing(pipeline_config.preprocessors):
        preprocessor = compose_preprocessor(pipeline_config.preprocessors)

    pipeline = create_model_pipeline(
        model=pipeline_config.sklearn_model, preprocessor=preprocessor
    )

    X = get_X(df=df, target=pipeline_config.target)
    y = get_y(df=df, target=pipeline_config.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # TODO: Configure
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(predictions)

    if do_evaluation(pipeline_config.evaluators):
        evaluations = evaluate(
            pipeline_config.evaluators, y_test, X_test, predictions, pipeline
        )
        print(evaluations)

    # TODO: Add write (either dump evaluations or integrate with e.g. MLFlow)


if __name__ == "__main__":
    try:
        pipeline_config_path = sys.argv[1]
    except IndexError:
        raise
    run(pipeline_config_path)
