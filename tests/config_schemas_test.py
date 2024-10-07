import pytest
from pydantic import ValidationError

from maxML.config_schemas import EvaluatorConfig
from maxML.config_schemas import EVALUATORS
from maxML.config_schemas import load_config
from maxML.config_schemas import ModelConfig
from maxML.config_schemas import PipelineConfig
from maxML.config_schemas import PreprocessorConfig
from maxML.config_schemas import PREPROCESSORS


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


# @pytest.mark.parametrize(
#     "config_type",
#     [
#         pytest.param("preprocessors", id="preprocessors"),
#         pytest.param("evaluators", id="preprocessors"),
#     ]
# )
# def test_missing_type_key(config_type: str):
#     """Test that missing 'type' key in preprocessors or evaluators raises error."""
#     config_dict = load_config_dict("tests/test_configs/columntransformer_logistic.yaml")
#     config_dict["preprocessors"][0] = {}
#     with pytest.raises(KeyError) as exc_info:
#         load_config("tests/test_configs/columntransformer_logistic.yaml")
#     assert f"{config_type} dict must contain a type key." in str(exc_info.value)


# @pytest.mark.parametrize(
#     "config_type, valid_list",
#     [
#         pytest.param("preprocessors", PREPROCESSORS, id="preprocessors"),
#         pytest.param("evaluators", EVALUATORS, id="evaluators"),
#     ],
# )
# def test_invalid_type_value(config_type: str, valid_list: list[str]):
#     """Test that invalid 'type' value in preprocessors or evaluators raises error."""
#     config_dict = load_config("tests/test_configs/columntransformer_logistic.yaml")
#     config_dict[config_type][0]["type"] = "InvalidPreprocessor"
#     with pytest.raises(KeyError) as exc_info:
#         PipelineConfig(**config_dict)
#     assert f"Invalid type: {config_type}" in str(exc_info.value)
#     assert f"Must be one of {valid_list}" in str(exc_info.value)


# @pytest.mark.parametrize(
#     "config_type",
#     [
#         pytest.param("preprocessors", id="preprocessors"),
#         pytest.param("evaluators", id="evaluators"),
#     ],
# )
# def test_empty_config_lists(config_type: str):
#     """Test that empty preprocessors or evaluators lists are handled correctly."""
#     config_dict = load_config("tests/test_configs/columntransformer_logistic.yaml")
#     config_dict[config_type] = []
#     config = PipelineConfig(**config_dict)
#     assert len(getattr(config, config_type)) == 1
#     assert isinstance(getattr(config, config_type)[0], PreprocessorConfig)


# def test_unknown_config_type():
#     """Test that an unknown config type raises ValueError."""
#     config_dict = load_config("tests/test_configs/columntransformer_logistic.yaml")
#     config_dict["unknown_field"] = [{"type": "something"}]

#     with pytest.raises(ValueError) as exc_info:
#         PipelineConfig(**config_dict)
#     assert "Unknown config type: unknown_field" in str(exc_info.value)
