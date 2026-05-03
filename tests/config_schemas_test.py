import pytest

from maxML.config_schemas import EvaluatorConfig
from maxML.config_schemas import ModelConfig
from maxML.config_schemas import PipelineConfig
from maxML.config_schemas import PreprocessorConfig
from maxML.config_schemas import load_config

_BASE_CONFIG_DICT: dict = {
    "input_path": "test.csv",
    "train_test_split": {"test_size": 0.2},
    "model": {"module": "sklearn.linear_model.LogisticRegression", "target": "y"},
    "preprocessors": [],
    "evaluators": [],
}


@pytest.mark.parametrize(
    "config_path",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml", id="full_configs"
        ),
        pytest.param("tests/test_configs/no_preprocessors.yaml", id="no_preprocessors"),
        pytest.param("tests/test_configs/no_evaluators.yaml", id="no_evaluators"),
    ],
)
def test_valid_config(config_path: str):
    """Test that a valid configuration loads correctly."""
    config: PipelineConfig = load_config(config_path)
    assert config.input_path == "tests/test_data/gemini_sample_data.csv"
    assert isinstance(config.preprocessors[0], PreprocessorConfig)
    assert isinstance(config.model, ModelConfig)
    assert config.model.args["random_state"] == 42  # type: ignore
    assert isinstance(config.evaluators[0], EvaluatorConfig)


@pytest.mark.parametrize(
    "config_path, expected_dropna, expected_feature_columns",
    [
        pytest.param(
            "tests/test_configs/no_preprocessors.yaml",
            True,
            ["Age", "Income", "Years_of_Experience"],
            id="dropna_and_feature_columns",
        ),
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            False,
            None,
            id="default_dropna_and_feature_columns",
        ),
    ],
)
def test_optional_config_fields(
    config_path: str,
    expected_dropna: bool,
    expected_feature_columns: list[str] | None,
):
    """Test that dropna and feature_columns are parsed correctly."""
    config: PipelineConfig = load_config(config_path)
    assert config.dropna == expected_dropna
    assert config.model.feature_columns == expected_feature_columns


@pytest.mark.parametrize(
    "override, field_name",
    [
        pytest.param(
            {"preprocessors": [{"type": "InvalidPreprocessor", "pipelines": []}]},
            "preprocessors",
            id="invalid_preprocessor_type",
        ),
        pytest.param(
            {"preprocessors": [{"pipelines": []}]},
            "preprocessors",
            id="missing_preprocessor_type_key",
        ),
        pytest.param(
            {"evaluators": [{"type": "InvalidEvaluator"}]},
            "evaluators",
            id="invalid_evaluator_type",
        ),
        pytest.param(
            {"evaluators": [{"metrics": ["accuracy_score"]}]},
            "evaluators",
            id="missing_evaluator_type_key",
        ),
    ],
)
def test_invalid_config(override: dict, field_name: str):
    """Test that invalid or missing type keys raise a KeyError."""
    with pytest.raises(KeyError):
        PipelineConfig(**{**_BASE_CONFIG_DICT, **override})
