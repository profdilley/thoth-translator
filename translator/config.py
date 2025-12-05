"""
Configuration management for THOTH translation tool.

This module handles loading, saving, and accessing configuration settings
from YAML files and environment variables. It provides sensible defaults
and validation for all configuration options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ColumnDefaults:
    """Default settings for column selection behavior."""

    auto_select_foreign_text: bool = True
    skip_numeric: bool = True
    skip_dates: bool = True
    skip_english: bool = True
    skip_empty: bool = True


@dataclass
class PerformanceSettings:
    """Performance-related configuration options."""

    batch_size: int = 32
    max_workers: int = 4
    show_progress: bool = True


@dataclass
class DetectionSettings:
    """Language detection configuration options."""

    confidence_threshold: float = 0.5
    fallback_language: str = "eng_Latn"


@dataclass
class ModelPaths:
    """Paths to translation and detection models."""

    nllb_path: str = "models/nllb-200-distilled-600M"
    lid_path: str = "models/lid218e.bin"
    argos_path: str = "models/argos"


@dataclass
class Config:
    """
    Main configuration class for THOTH.

    Loads configuration from YAML file with fallback to defaults.
    Supports environment variable overrides for key settings.

    Attributes:
        default_engine: Default translation engine ('nllb' or 'argos')
        column_defaults: Column selection behavior settings
        performance: Performance-related settings
        detection: Language detection settings
        models: Paths to model files
        column_overrides: Per-column language overrides

    Example:
        config = Config.load("config.yaml")
        print(config.default_engine)  # 'nllb'
        print(config.performance.batch_size)  # 32
    """

    default_engine: str = "nllb"
    column_defaults: ColumnDefaults = field(default_factory=ColumnDefaults)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    models: ModelPaths = field(default_factory=ModelPaths)
    column_overrides: dict[str, str] = field(default_factory=dict)

    # Internal: path to the loaded config file
    _config_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML file.

        Searches for config in the following order:
        1. Explicitly provided path
        2. THOTH_CONFIG environment variable
        3. config.yaml in current directory
        4. config.yaml in script directory

        Args:
            config_path: Optional explicit path to config file

        Returns:
            Config object with loaded settings
        """
        # Determine config file location
        if config_path:
            path = Path(config_path)
        elif os.environ.get("THOTH_CONFIG"):
            path = Path(os.environ["THOTH_CONFIG"])
        else:
            # Try current directory first
            path = Path("config.yaml")
            if not path.exists():
                # Try script directory
                script_dir = Path(__file__).parent.parent
                path = script_dir / "config.yaml"

        # Load from file if it exists
        config_dict: dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        # Build config object
        config = cls._from_dict(config_dict)
        config._config_path = path if path.exists() else None

        # Apply environment variable overrides
        config._apply_env_overrides()

        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Create Config from dictionary (parsed YAML).

        Args:
            data: Dictionary of configuration values

        Returns:
            Config object with values from dictionary
        """
        # Parse nested configurations
        column_defaults = ColumnDefaults(
            **data.get("column_defaults", {})
        ) if data.get("column_defaults") else ColumnDefaults()

        performance = PerformanceSettings(
            **data.get("performance", {})
        ) if data.get("performance") else PerformanceSettings()

        detection = DetectionSettings(
            **data.get("detection", {})
        ) if data.get("detection") else DetectionSettings()

        models = ModelPaths(
            **data.get("models", {})
        ) if data.get("models") else ModelPaths()

        return cls(
            default_engine=data.get("default_engine", "nllb"),
            column_defaults=column_defaults,
            performance=performance,
            detection=detection,
            models=models,
            column_overrides=data.get("column_overrides", {}),
        )

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Engine override
        if os.environ.get("THOTH_ENGINE"):
            self.default_engine = os.environ["THOTH_ENGINE"]

        # Batch size override
        if os.environ.get("THOTH_BATCH_SIZE"):
            try:
                self.performance.batch_size = int(os.environ["THOTH_BATCH_SIZE"])
            except ValueError:
                pass

        # Model path overrides
        if os.environ.get("THOTH_NLLB_PATH"):
            self.models.nllb_path = os.environ["THOTH_NLLB_PATH"]
        if os.environ.get("THOTH_LID_PATH"):
            self.models.lid_path = os.environ["THOTH_LID_PATH"]
        if os.environ.get("THOTH_ARGOS_PATH"):
            self.models.argos_path = os.environ["THOTH_ARGOS_PATH"]

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save to (defaults to original load path)
        """
        save_path = Path(path) if path else self._config_path
        if not save_path:
            save_path = Path("config.yaml")

        config_dict = self.to_dict()

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for YAML serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "default_engine": self.default_engine,
            "column_defaults": {
                "auto_select_foreign_text": self.column_defaults.auto_select_foreign_text,
                "skip_numeric": self.column_defaults.skip_numeric,
                "skip_dates": self.column_defaults.skip_dates,
                "skip_english": self.column_defaults.skip_english,
                "skip_empty": self.column_defaults.skip_empty,
            },
            "performance": {
                "batch_size": self.performance.batch_size,
                "max_workers": self.performance.max_workers,
                "show_progress": self.performance.show_progress,
            },
            "detection": {
                "confidence_threshold": self.detection.confidence_threshold,
                "fallback_language": self.detection.fallback_language,
            },
            "models": {
                "nllb_path": self.models.nllb_path,
                "lid_path": self.models.lid_path,
                "argos_path": self.models.argos_path,
            },
            "column_overrides": self.column_overrides,
        }

    def get_model_dir(self) -> Path:
        """
        Get the base directory for models.

        Returns:
            Path to models directory
        """
        # Try to find models relative to config or script
        if self._config_path:
            base = self._config_path.parent
        else:
            base = Path(__file__).parent.parent

        return base / "models"

    def get_nllb_path(self) -> Path:
        """Get full path to NLLB model."""
        path = Path(self.models.nllb_path)
        if path.is_absolute():
            return path
        if self._config_path:
            return self._config_path.parent / path
        return Path(__file__).parent.parent / path

    def get_lid_path(self) -> Path:
        """Get full path to fastText LID model."""
        path = Path(self.models.lid_path)
        if path.is_absolute():
            return path
        if self._config_path:
            return self._config_path.parent / path
        return Path(__file__).parent.parent / path

    def get_argos_path(self) -> Path:
        """Get full path to Argos models directory."""
        path = Path(self.models.argos_path)
        if path.is_absolute():
            return path
        if self._config_path:
            return self._config_path.parent / path
        return Path(__file__).parent.parent / path

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate engine
        if self.default_engine not in ("nllb", "argos"):
            errors.append(
                f"Invalid engine '{self.default_engine}'. Must be 'nllb' or 'argos'."
            )

        # Validate batch size
        if self.performance.batch_size < 1:
            errors.append("Batch size must be at least 1.")
        if self.performance.batch_size > 256:
            errors.append("Batch size should not exceed 256 for memory efficiency.")

        # Validate max workers
        if self.performance.max_workers < 1:
            errors.append("Max workers must be at least 1.")
        if self.performance.max_workers > 32:
            errors.append("Max workers should not exceed 32.")

        # Validate confidence threshold
        if not (0.0 <= self.detection.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0.")

        return errors

    def check_models_exist(self) -> dict[str, bool]:
        """
        Check which required models exist on disk.

        Returns:
            Dictionary mapping model name to existence status
        """
        return {
            "nllb": self.get_nllb_path().exists(),
            "lid": self.get_lid_path().exists(),
            "argos": self.get_argos_path().exists(),
        }

    def get_missing_models(self) -> list[str]:
        """
        Get list of models that need to be downloaded.

        Returns:
            List of missing model names
        """
        status = self.check_models_exist()
        missing = [name for name, exists in status.items() if not exists]

        # For NLLB engine, we only need NLLB and LID
        # For Argos engine, we need Argos and LID
        if self.default_engine == "nllb" and "argos" in missing:
            missing.remove("argos")
        if self.default_engine == "argos" and "nllb" in missing:
            missing.remove("nllb")

        return missing


def get_default_config() -> Config:
    """
    Get default configuration without loading from file.

    Returns:
        Config with all default values
    """
    return Config()


def generate_config_template() -> str:
    """
    Generate a YAML configuration template with comments.

    Returns:
        YAML string with configuration template
    """
    return '''# THOTH Configuration File
# Translator for Hybrid Offline Text Handling

# Default translation engine: "nllb" or "argos"
# NLLB-200: Better language coverage (200 languages), recommended for most use cases
# Argos: May have better quality for some Western European languages
default_engine: nllb

# Default behavior for column selection
column_defaults:
  # Automatically select columns detected as foreign-language text
  auto_select_foreign_text: true
  # Skip columns that appear to contain numeric data
  skip_numeric: true
  # Skip columns that appear to contain date values
  skip_dates: true
  # Skip columns that are already in English
  skip_english: true
  # Skip columns that are empty or mostly empty
  skip_empty: true

# Performance settings
performance:
  # Number of texts to translate in each batch (higher = faster but more memory)
  batch_size: 32
  # Maximum number of worker threads for parallel processing
  max_workers: 4
  # Show progress bar during translation
  show_progress: true

# Language detection settings
detection:
  # Minimum confidence score to accept language detection (0.0-1.0)
  confidence_threshold: 0.5
  # Fallback language if detection fails or confidence is too low
  fallback_language: eng_Latn

# Per-column language overrides (optional)
# Uncomment and modify to force specific languages for columns
# column_overrides:
#   description: rus_Cyrl
#   notes: ukr_Cyrl
#   comments: deu_Latn

# Model paths (auto-populated by setup, modify if using custom locations)
models:
  nllb_path: models/nllb-200-distilled-600M
  lid_path: models/lid218e.bin
  argos_path: models/argos
'''
