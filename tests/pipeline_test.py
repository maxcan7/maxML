import pandas as pd
import pytest

from maxML.config_schemas import PipelineConfig
from maxML.pipeline import create_model_pipeline
from maxML.pipeline import get_X
from maxML.pipeline import get_y
from maxML.pipeline import run
from maxML.preprocessors import Preprocessor


def test_get_X(test_config: PipelineConfig, test_data: pd.DataFrame) -> None:
    """Test that get_X drops the target column from the DataFrame."""
    X = get_X(df=test_data, target=test_config.model.target)
    assert test_config.model.target not in X.columns
    pd.testing.assert_frame_equal(
        X, test_data.loc[:, test_data.columns != test_config.model.target]
    )


def test_get_X_with_feature_columns(test_data: pd.DataFrame) -> None:
    """
    Test that feature_columns restricts the returned DataFrame to specified columns.
    """
    feature_columns = ["Age", "Income"]
    X = get_X(df=test_data, target="Purchased", feature_columns=feature_columns)
    assert list(X.columns) == feature_columns
    assert "Purchased" not in X.columns


def test_get_y(test_config: PipelineConfig, test_data: pd.DataFrame) -> None:
    """Test that get_y returns only the target column as a Series."""
    y = get_y(df=test_data, target=test_config.model.target)
    assert y.name == test_config.model.target
    pd.testing.assert_series_equal(y, test_data[test_config.model.target])


@pytest.mark.parametrize(
    "include_preprocessor, expected_steps",
    [
        pytest.param(False, ["model"], id="no_preprocessor"),
        pytest.param(True, ["preprocessor", "model"], id="with_preprocessor"),
    ],
)
def test_create_model_pipeline(
    test_config: PipelineConfig,
    test_preprocessor: Preprocessor,
    include_preprocessor: bool,
    expected_steps: list[str],
) -> None:
    """Test that the pipeline has the correct steps with and without a preprocessor."""
    preprocessor = test_preprocessor if include_preprocessor else None
    pipeline = create_model_pipeline(
        model_config=test_config.model, preprocessor=preprocessor
    )
    assert [name for name, _ in pipeline.steps] == expected_steps


def test_load_data(test_data: pd.DataFrame) -> None:
    """test_data pytest fixture loaded with load_data."""
    assert isinstance(test_data, pd.DataFrame)


def test_dropna(test_data: pd.DataFrame) -> None:
    """Test that dropna removes rows with NaN values from the dataset."""
    assert test_data.isnull().any().any()
    df_clean = test_data.dropna()
    assert not df_clean.isnull().any().any()
    assert len(df_clean) < len(test_data)


@pytest.mark.parametrize(
    "config_path",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            id="columntransformer_logistic",
        ),
        pytest.param(
            "tests/test_configs/columntransformer_poly_logistic.yaml",
            id="columntransformer_poly",
        ),
        pytest.param("tests/test_configs/no_evaluators.yaml", id="no_evaluators"),
        pytest.param("tests/test_configs/no_preprocessors.yaml", id="no_preprocessors"),
    ],
)
def test_run(config_path: str) -> None:
    """Integration test: verify run() completes without error for each config."""
    run(config_path)
