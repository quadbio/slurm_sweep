import pytest

from slurm_sweep.utils import ConfigValidator


def test_load_valid_config(valid_config_file):
    validator = ConfigValidator(config_path=str(valid_config_file))
    validator.load_config()
    assert "wandb" in validator.config
    assert "general" in validator.config


def test_load_invalid_config(invalid_config_file):
    validator = ConfigValidator(config_path=str(invalid_config_file))
    validator.load_config()
    with pytest.raises(ValueError, match="The `wandb` block must include `program`, `method`, and `parameters` keys."):
        validator.validate()


def test_missing_config_file():
    validator = ConfigValidator(config_path="non_existent_file.yaml")
    with pytest.raises(FileNotFoundError, match="Configuration file 'non_existent_file.yaml' not found."):
        validator.load_config()


def test_unexpected_blocks(unexpected_block_config_file):
    validator = ConfigValidator(config_path=str(unexpected_block_config_file))
    validator.load_config()
    with pytest.warns(UserWarning, match="The configuration file contains unexpected blocks: {'extra_block'}"):
        validator.validate()


def test_default_slurm_block(valid_config_file):
    validator = ConfigValidator(config_path=str(valid_config_file))
    validator.load_config()
    validator.validate()
    assert "slurm" in validator.config
    assert validator.config["slurm"] == {}
