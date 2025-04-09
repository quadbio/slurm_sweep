import pytest
import yaml


@pytest.fixture
def valid_config_file(tmp_path):
    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
        },
    }
    file_path = tmp_path / "valid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def invalid_config_file(tmp_path):
    config = {
        "wandb": {
            "method": "grid",
        },
        "general": {
            "entity": "test_entity",
        },
    }
    file_path = tmp_path / "invalid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def unexpected_block_config_file(tmp_path):
    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
        },
        "extra_block": {"key": "value"},
    }
    file_path = tmp_path / "unexpected_block_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path
