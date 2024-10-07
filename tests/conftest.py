import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.pipeline import load_data
from maxML.preprocessors import compose_preprocessor
from maxML.preprocessors import Preprocessor


def columntransformer_fixture() -> ColumnTransformer:
    """Hard-coded ColumnTransformer fixture."""
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

    ed_categories = ["High School", "Bachelor's", "Master's", "PhD"]
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ed_categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )
    ordinal = ("ordinal", ordinal_pipeline, ["Education"])
    return ColumnTransformer(transformers=[numeric, nominal, ordinal])


def featureunion_fixture() -> FeatureUnion:
    """
    Hard-coded FeatureUnion fixture which takes a ColumnTransformer as one of its
    transformers in order to test both FeatureUnion and the ability of maxML
    preprocessors to handle two Preprocessors.
    """
    columntransformer = ("columntransformer", columntransformer_fixture())
    numeric_featureunion_pipeline = Pipeline(
        [
            ("robust_scaler", RobustScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
        ]
    )
    numeric_featureunion_columns = ["Age", "Income", "Years_of_Experience"]
    numeric_featureunion = (
        "numeric",
        numeric_featureunion_pipeline,
        numeric_featureunion_columns,
    )
    return FeatureUnion(transformer_list=[columntransformer, numeric_featureunion])


@pytest.fixture
def test_preprocessor(test_config: PipelineConfig) -> Preprocessor:
    return compose_preprocessor(test_config.preprocessors)


@pytest.fixture
def test_data(test_config: PipelineConfig) -> pd.DataFrame:
    return load_data(test_config.input_path)


@pytest.fixture
def test_config() -> PipelineConfig:
    return load_config("tests/test_configs/columntransformer_logistic.yaml")
