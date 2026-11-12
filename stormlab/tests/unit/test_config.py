"""Loading and validating the shipped ``config.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest

from stormlab.config import ConfigError, default_config_path, load_config


def test_the_shipped_configuration_loads() -> None:
    config = load_config()

    assert config.source == default_config_path()
    assert config.variant == "default"
    assert config.frost_zone_depth == 0.8
    assert config.hydraulics.v_min == 0.7


def test_a_caller_may_request_the_configuration_by_name() -> None:
    assert load_config(variant="default").variant == "default"


def test_a_request_for_another_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="requested"):
        load_config(variant="other")


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(default_config_path().read_text(encoding="utf-8") + "\nmystery: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown config keys"):
        load_config(path)


def test_a_frost_zone_outside_the_admissible_range_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    text = default_config_path().read_text(encoding="utf-8").replace("frost_zone_depth: 0.8", "frost_zone_depth: 5.0")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="frost_zone_depth"):
        load_config(path)
