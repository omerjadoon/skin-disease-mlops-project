"""
Validation tests for dataset and model configurations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class TestConfigValidation:
    """Validate project configuration files under configs/."""

    def test_configs_exist(self):
        """Ensure all required YAML configs exist in the configs directory."""
        required_configs = ["api.yaml", "data.yaml", "model.yaml", "train.yaml"]
        for config_name in required_configs:
            config_path = CONFIG_DIR / config_name
            assert config_path.exists(), f"Configuration file {config_name} is missing."

    def test_configs_are_valid_yaml(self):
        """Verify that all configuration files can be parsed as valid YAML."""
        for config_path in CONFIG_DIR.glob("*.yaml"):
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                assert isinstance(cfg, dict), f"{config_path.name} is not a valid YAML dictionary."
            except Exception as e:
                pytest.fail(f"Failed to parse {config_path.name}: {e}")

    def test_data_config_values(self):
        """Validate key configurations in data.yaml."""
        data_cfg_path = CONFIG_DIR / "data.yaml"
        with open(data_cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Mode and classes
        assert cfg.get("dataset_mode") in ["folder", "csv"], (
            "dataset_mode must be 'folder' or 'csv'"
        )
        assert cfg.get("class_names") == ["mild", "moderate", "severe"], (
            "class_names must be ['mild', 'moderate', 'severe']"
        )

        # Splitting parameters
        val_split = cfg.get("val_split", 0)
        test_split = cfg.get("test_split", 0)
        assert 0.0 <= val_split <= 1.0, "val_split must be in [0.0, 1.0]"
        assert 0.0 <= test_split <= 1.0, "test_split must be in [0.0, 1.0]"
        assert val_split + test_split < 1.0, "Sum of splits must be less than 1.0"

        # Normalization
        normalize = cfg.get("normalize", {})
        assert "mean" in normalize and "std" in normalize, "normalize must contain mean and std"
        assert len(normalize["mean"]) == 3, "mean must have 3 channels"
        assert len(normalize["std"]) == 3, "std must have 3 channels"

        # Image settings
        assert cfg.get("image_size", 0) > 0, "image_size must be a positive integer"
        assert cfg.get("image_channels", 0) == 3, "image_channels must be 3"

    def test_model_config_values(self):
        """Validate key configurations in model.yaml."""
        model_cfg_path = CONFIG_DIR / "model.yaml"
        with open(model_cfg_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg.get("backbone") in [
            "resnet18",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b2",
        ], f"Unsupported backbone: {cfg.get('backbone')}"
        assert cfg.get("num_classes") == 3, "num_classes must be 3"
        assert cfg.get("class_names") == ["mild", "moderate", "severe"], (
            "class_names must be ['mild', 'moderate', 'severe']"
        )

        # Image properties
        input_size = cfg.get("input_size", [])
        assert len(input_size) == 2, "input_size must be [height, width]"
        assert input_size[0] > 0 and input_size[1] > 0, "input_size dimensions must be positive"

        # Dropout
        dropout = cfg.get("dropout", -1.0)
        assert 0.0 <= dropout <= 1.0, "dropout must be in [0.0, 1.0]"

    def test_train_config_values(self):
        """Validate key configurations in train.yaml."""
        train_cfg_path = CONFIG_DIR / "train.yaml"
        with open(train_cfg_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg.get("max_epochs", 0) > 0, "max_epochs must be a positive integer"
        assert cfg.get("learning_rate", 0.0) > 0.0, "learning_rate must be positive"
        assert cfg.get("batch_size", 0) > 0, "batch_size must be a positive integer"

        assert cfg.get("optimizer") in ["adam", "sgd", "adamw"], "Unsupported optimizer"
        assert cfg.get("scheduler") in ["cosine", "step", "none"], "Unsupported scheduler"

    def test_api_config_values(self):
        """Validate key configurations in api.yaml."""
        api_cfg_path = CONFIG_DIR / "api.yaml"
        with open(api_cfg_path) as f:
            cfg = yaml.safe_load(f)

        assert 1 <= cfg.get("port", 0) <= 65535, "Port must be in range [1, 65535]"
        assert cfg.get("class_names") == ["mild", "moderate", "severe"], (
            "class_names must be ['mild', 'moderate', 'severe']"
        )
        assert "database" in cfg, "database config is required"
        assert "host" in cfg["database"], "database host is required"
        assert "port" in cfg["database"], "database port is required"
