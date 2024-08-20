import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from maxML.config_schemas import PipelineConfig


"""
This demo_pipeline is a more conventional approach to running an sklearn model
pipeline end-to-end.
"""


def evaluate_logistic(
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
    pipeline: Pipeline,
    predictions: np.ndarray,
) -> dict:
    """TODO: configure"""
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    # TODO: Convert to pydantic or dataclass.
    return {
        "accuracy": accuracy,
        "report": report,
        "roc_auc": roc_auc,
    }


def evaluate_linear(
    y_test: pd.DataFrame, X_test: pd.DataFrame, predictions: np.ndarray
) -> dict:
    """TODO: configure"""
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    # TODO: Convert to pydantic or dataclass.
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def get_y(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Subset dataframe to just target column."""
    return df[target]


def get_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Subset dataframe removing target column."""
    return df.drop(target, axis=1)


def create_model_pipeline(
    model: BaseEstimator, preprocessor: ColumnTransformer
) -> Pipeline:
    """TODO: Configure"""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def create_preprocessor() -> ColumnTransformer:
    """TODO: Configure"""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_columns = ["Age", "Income", "Years_of_Experience"]
    numeric = ("numeric", numeric_pipeline, numeric_columns)

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    nominal_columns = ["Gender", "City"]
    nominal = ("nominal", nominal_pipeline, nominal_columns)

    ed_categories = ["High School", "Bachelor's", "Master's" "PhD"]
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                # NOTE: Unkown Value is 4 bc HS -> PhD == 1 -> 3.
                OrdinalEncoder(
                    categories=[ed_categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=4,
                ),
            ),
        ]
    )
    ordinal = ("ordinal", ordinal_pipeline, ["Education"])

    return ColumnTransformer(transformers=[numeric, nominal, ordinal])


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from input_path as pandas DataFrame."""
    return pd.read_csv(Path.cwd() / input_path)


def main(pipeline_config_path: str) -> None:
    """
    Run sklearn pipelines on dataset with preprocessing steps.
    """
    pipeline_config = PipelineConfig(
        **yaml.safe_load(open(Path.cwd() / pipeline_config_path))
    )
    df = load_data(pipeline_config.input_path)
    preprocessor = create_preprocessor()

    pipeline = create_model_pipeline(
        model=pipeline_config.sklearn_model, preprocessor=preprocessor
    )

    X = get_X(df=df, target=pipeline_config.target)
    y = get_y(df=df, target=pipeline_config.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # TODO: Abstract out evaluate logic.
    # linear_metrics = evaluate_linear(
    #     y_test=y_test, X_test=X_test, predictions=linear_predictions
    # )
    # print("Linear Regression:")
    # print(f"  MSE: {linear_metrics['mse']:.2f}")
    # print(f"  RMSE: {linear_metrics['rmse']:.2f}")
    # print(f"  R-squared: {linear_metrics['r2']:.2f}")

    # TODO: Abstract out evaluate logic.
    logistic_metrics = evaluate_logistic(
        y_test=y_test,
        X_test=X_test,
        pipeline=pipeline,
        predictions=predictions,
    )
    print("\nLogistic Regression:")
    print(f"  Accuracy: {logistic_metrics['accuracy']:.2f}")
    print(f"  Classification Report:\n{logistic_metrics['report']}")
    print(f"  ROC AUC: {logistic_metrics['roc_auc']:.2f}")


if __name__ == "__main__":
    try:
        pipeline_config_path = sys.argv[1]
    except IndexError:
        raise
    main(pipeline_config_path)
