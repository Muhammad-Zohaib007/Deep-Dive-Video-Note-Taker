"""Configuration management for Deep-Dive Video Note Taker.

Loads configuration from:
1. config.default.yaml (bundled defaults)
2. ~/.notetaker/config.yaml (user overrides)
3. CLI flags / environment variables (highest priority)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def _get_default_config_path() -> Path:
    """Locate the bundled config.default.yaml.

    Works both when running from source and when installed as a package.
    """
    # First: try relative to this file (works in source tree)
    source_path = Path(__file__).parent.parent.parent / "config.default.yaml"
    if source_path.exists():
        return source_path

    # Second: try importlib.resources (works when installed as package)
    try:
        import importlib.resources as pkg_resources
        # Try the top-level package directory
        ref = pkg_resources.files("notetaker").joinpath("../../config.default.yaml")
        if hasattr(ref, "_path") and Path(str(ref)).exists():
            return Path(str(ref))
    except Exception:
        pass

    # Fallback: return the source path (will be handled gracefully by _load_yaml)
    return source_path


_DEFAULT_CONFIG_PATH = _get_default_config_path()
_USER_CONFIG_DIR = Path.home() / ".notetaker"
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_paths(config: dict) -> dict:
    """Expand ~ in path values."""
    path_keys = {"data_dir", "output_dir", "persist_directory", "log_dir"}
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = _expand_paths(value)
        elif isinstance(value, str) and key in path_keys:
            config[key] = str(Path(value).expanduser())
    return config


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class AppConfig:
    """Application configuration singleton."""

    _instance: AppConfig | None = None
    _config: dict[str, Any]

    def __new__(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
        return cls._instance

    def load(self, overrides: dict[str, Any] | None = None) -> None:
        """Load configuration from defaults, user file, and overrides."""
        # 1. Bundled defaults
        defaults = _load_yaml(_DEFAULT_CONFIG_PATH)

        # 2. User config
        user_config = _load_yaml(_USER_CONFIG_PATH)

        # 3. Merge: defaults < user < overrides
        merged = _deep_merge(defaults, user_config)
        if overrides:
            merged = _deep_merge(merged, overrides)

        # Expand ~ paths
        self._config = _expand_paths(merged)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Ensure data directories exist
        self._ensure_dirs()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to the config.

        Supported env vars:
            NOTETAKER_OLLAMA_BASE_URL -> ollama.base_url
            NOTETAKER_OLLAMA_MODEL    -> ollama.model
            NOTETAKER_WHISPER_MODEL   -> whisper.model
            NOTETAKER_DATA_DIR        -> data_dir
            NOTETAKER_OUTPUT_DIR      -> output_dir
        """
        env_mappings = {
            "NOTETAKER_OLLAMA_BASE_URL": "ollama.base_url",
            "NOTETAKER_OLLAMA_MODEL": "ollama.model",
            "NOTETAKER_WHISPER_MODEL": "whisper.model",
            "NOTETAKER_DATA_DIR": "data_dir",
            "NOTETAKER_OUTPUT_DIR": "output_dir",
        }
        for env_var, dotted_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self.set(dotted_key, value)

    def _ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.data_dir / "videos",
            self.data_dir / "models",
            Path(self.get("chroma.persist_directory", str(self.data_dir / "chroma"))),
            Path(self.get("logging.log_dir", str(self.data_dir / "logs"))),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def init_user_config(self) -> None:
        """Copy default config to user directory if not present."""
        if not _USER_CONFIG_PATH.exists():
            _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            if _DEFAULT_CONFIG_PATH.exists():
                shutil.copy2(_DEFAULT_CONFIG_PATH, _USER_CONFIG_PATH)

    @property
    def data_dir(self) -> Path:
        return Path(self._config.get("data_dir", str(Path.home() / ".notetaker")))

    @property
    def output_dir(self) -> Path:
        return Path(self._config.get("output_dir", "./outputs"))

    @property
    def max_duration(self) -> int:
        return int(self._config.get("max_video_duration_seconds", 900))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dot notation (e.g., 'ollama.model')."""
        keys = dotted_key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a config value using dot notation."""
        keys = dotted_key.split(".")
        target = self._config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        """Return the full config as a dictionary."""
        return self._config.copy()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None


def get_config(overrides: dict[str, Any] | None = None) -> AppConfig:
    """Get or initialize the application config."""
    config = AppConfig()
    if not config._config:
        config.load(overrides)
    return config
