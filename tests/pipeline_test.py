import pandas as pd

from maxML.config_schemas import PipelineConfig
from maxML.pipeline import create_model_pipeline
from maxML.pipeline import get_X
from maxML.pipeline import get_y
from maxML.preprocessors import Preprocessor


def test_get_X(test_config: PipelineConfig, test_data: pd.DataFrame) -> None:
    X = get_X(df=test_data, target=test_config.model.target)
    assert test_config.model.target not in X.columns
    pd.testing.assert_frame_equal(
        X, test_data.loc[:, test_data.columns != test_config.model.target]
    )


def test_get_y(test_config: PipelineConfig, test_data: pd.DataFrame) -> None:
    y = get_y(df=test_data, target=test_config.model.target)
    assert y.name == test_config.model.target
    pd.testing.assert_series_equal(y, test_data[test_config.model.target])


def test_create_model_pipeline(
    test_config: PipelineConfig, test_preprocessor: Preprocessor
) -> None:
    pipeline = create_model_pipeline(
        model_config=test_config.model, preprocessor=test_preprocessor
    )
    assert pipeline


def test_load_data(test_data: pd.DataFrame) -> None:
    """test_data pytest fixture loaded with load_data."""
    assert isinstance(test_data, pd.DataFrame)
