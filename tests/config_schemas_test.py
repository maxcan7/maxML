import pytest

from maxML.config_schemas import EvaluatorConfig
from maxML.config_schemas import load_config
from maxML.config_schemas import ModelConfig
from maxML.config_schemas import PipelineConfig
from maxML.config_schemas import PreprocessorConfig


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
    assert config.input_path == "data/gemini_sample_data.csv"
    assert isinstance(config.preprocessors[0], PreprocessorConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.evaluators[0], EvaluatorConfig)


# TODO:
# - Add more cases.
# - Test specific configs i.e. PreprocessorConfig, ModelConfig, EvaluatorConfig.abs
