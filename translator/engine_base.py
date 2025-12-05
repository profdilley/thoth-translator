"""
Abstract base class for translation engines.

This module defines the interface that all translation engines must implement,
ensuring consistent behavior across different backends (NLLB, Argos, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .progress import ProgressTracker


@dataclass
class TranslationResult:
    """Result of a single translation operation."""

    # Original text
    source_text: str

    # Translated text (None if translation failed)
    translated_text: Optional[str]

    # Detected source language code
    source_language: str

    # Target language code
    target_language: str

    # Whether translation was successful
    success: bool

    # Error message if translation failed
    error: Optional[str] = None

    # Confidence score (0-1) if available
    confidence: Optional[float] = None


@dataclass
class BatchTranslationResult:
    """Result of a batch translation operation."""

    # Individual translation results
    results: list[TranslationResult]

    # Number of successful translations
    success_count: int

    # Number of failed translations
    failure_count: int

    # Total processing time in seconds
    processing_time: float

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total = len(self.results)
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100


class TranslationEngine(ABC):
    """
    Abstract base class for translation engines.

    All translation engines (NLLB, Argos, etc.) must implement this interface
    to ensure consistent behavior and easy switching between engines.

    Subclasses must implement:
        - translate(): Translate a single text
        - translate_batch(): Translate multiple texts efficiently
        - is_available(): Check if engine is ready to use
        - get_supported_languages(): List supported language codes
        - load_model(): Load the translation model
        - unload_model(): Unload model to free memory

    Example:
        engine = NLLBEngine()
        engine.load_model()

        result = engine.translate("Привет мир", "rus_Cyrl", "eng_Latn")
        print(result.translated_text)  # "Hello world"

        engine.unload_model()
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the translation engine.

        Args:
            model_path: Path to the model files (optional, uses default if not specified)
        """
        self._model_path = model_path
        self._model_loaded = False
        self._target_language = "eng_Latn"  # Default target is English

    @property
    def model_path(self) -> Optional[str]:
        """Get the model path."""
        return self._model_path

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model_loaded

    @property
    def target_language(self) -> str:
        """Get the target language for translations."""
        return self._target_language

    @target_language.setter
    def target_language(self, lang_code: str) -> None:
        """Set the target language for translations."""
        self._target_language = lang_code

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate a single text.

        Args:
            text: Text to translate
            source_lang: Source language code (engine-specific format)
            target_lang: Target language code (defaults to self.target_language)

        Returns:
            TranslationResult with translated text or error

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If language is not supported
        """
        pass

    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        source_langs: list[str],
        target_lang: Optional[str] = None,
        progress: Optional[ProgressTracker] = None,
    ) -> BatchTranslationResult:
        """
        Translate multiple texts efficiently.

        This method should be optimized for batch processing, which is
        typically much faster than translating texts one by one.

        Args:
            texts: List of texts to translate
            source_langs: List of source language codes (one per text)
            target_lang: Target language code (same for all, defaults to self.target_language)
            progress: Optional progress tracker for cancellation/updates

        Returns:
            BatchTranslationResult with all translation results

        Raises:
            RuntimeError: If model is not loaded
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the engine is available and ready to use.

        Returns:
            True if model files exist and can be loaded
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported source language codes.

        Returns:
            List of language codes in engine-specific format
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the translation model into memory.

        This may take significant time and memory for large models.
        Must be called before translate() or translate_batch().

        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload the translation model to free memory.

        Safe to call multiple times or if model is not loaded.
        """
        pass

    def supports_language(self, lang_code: str) -> bool:
        """
        Check if a specific language is supported.

        Args:
            lang_code: Language code to check

        Returns:
            True if language is supported
        """
        return lang_code.lower() in [
            lang.lower() for lang in self.get_supported_languages()
        ]

    @abstractmethod
    def get_engine_name(self) -> str:
        """
        Get human-readable engine name.

        Returns:
            Engine name (e.g., "NLLB-200", "Argos Translate")
        """
        pass

    @abstractmethod
    def get_engine_id(self) -> str:
        """
        Get engine identifier for configuration.

        Returns:
            Engine ID (e.g., "nllb", "argos")
        """
        pass

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model details
        """
        return {
            "engine": self.get_engine_name(),
            "engine_id": self.get_engine_id(),
            "model_path": self._model_path,
            "loaded": self._model_loaded,
            "target_language": self._target_language,
        }

    def __enter__(self) -> "TranslationEngine":
        """Context manager entry - load model."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - unload model."""
        self.unload_model()

    def _check_model_loaded(self) -> None:
        """
        Verify model is loaded, raise if not.

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model_loaded:
            raise RuntimeError(
                f"{self.get_engine_name()} model is not loaded. "
                f"Call load_model() first."
            )

    def _validate_language(self, lang_code: str, param_name: str = "language") -> None:
        """
        Validate a language code is supported.

        Args:
            lang_code: Language code to validate
            param_name: Parameter name for error message

        Raises:
            ValueError: If language is not supported
        """
        if not self.supports_language(lang_code):
            raise ValueError(
                f"Unsupported {param_name}: {lang_code}. "
                f"Use get_supported_languages() to see available languages."
            )


class TranslationEngineFactory:
    """
    Factory for creating translation engine instances.

    Provides a centralized way to create engines by name or ID,
    with support for custom engine registration.

    Example:
        engine = TranslationEngineFactory.create("nllb", model_path="/models/nllb")
    """

    _engines: dict[str, type[TranslationEngine]] = {}

    @classmethod
    def register(cls, engine_id: str, engine_class: type[TranslationEngine]) -> None:
        """
        Register a translation engine class.

        Args:
            engine_id: Unique engine identifier
            engine_class: TranslationEngine subclass
        """
        cls._engines[engine_id.lower()] = engine_class

    @classmethod
    def create(
        cls,
        engine_id: str,
        model_path: Optional[str] = None,
    ) -> TranslationEngine:
        """
        Create a translation engine instance.

        Args:
            engine_id: Engine identifier (e.g., "nllb", "argos")
            model_path: Optional model path

        Returns:
            TranslationEngine instance

        Raises:
            ValueError: If engine_id is not registered
        """
        engine_id_lower = engine_id.lower()
        if engine_id_lower not in cls._engines:
            available = ", ".join(cls._engines.keys())
            raise ValueError(
                f"Unknown engine: {engine_id}. Available: {available}"
            )

        return cls._engines[engine_id_lower](model_path=model_path)

    @classmethod
    def get_available_engines(cls) -> list[str]:
        """
        Get list of registered engine IDs.

        Returns:
            List of engine identifiers
        """
        return list(cls._engines.keys())

    @classmethod
    def is_registered(cls, engine_id: str) -> bool:
        """
        Check if an engine is registered.

        Args:
            engine_id: Engine identifier

        Returns:
            True if engine is registered
        """
        return engine_id.lower() in cls._engines
