from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Dictionary with recursive attribute access and serialization."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls({key: cls.wrap(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.wrap(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: unwrap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return unwrap(self)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    default_path = Path(__file__).with_name("default_config.yaml")
    data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    if path is not None and Path(path).resolve() != default_path.resolve():
        custom = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _deep_merge(data, custom)
    if overrides:
        data = _deep_merge(data, overrides)
    return Config.wrap(data)


def apply_dotted_overrides(config: Config, values: list[str]) -> Config:
    data = config.to_dict()
    for expression in values:
        if "=" not in expression:
            raise ValueError(f"Override must have KEY=VALUE form: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        cursor = data
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return Config.wrap(data)
