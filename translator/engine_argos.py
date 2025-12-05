"""
Argos Translate translation engine implementation.

This module provides translation using Argos Translate, an open-source
offline translation library. Argos may provide better quality for
some Western European language pairs.

Argos Translate: https://github.com/argosopentech/argos-translate
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .engine_base import (
    BatchTranslationResult,
    TranslationEngine,
    TranslationEngineFactory,
    TranslationResult,
)
from .languages import LanguageMapper
from .progress import ProgressTracker

logger = logging.getLogger(__name__)


class ArgosEngine(TranslationEngine):
    """
    Translation engine using Argos Translate.

    Argos Translate is an open-source offline translation library
    that provides good quality translations for many language pairs,
    particularly Western European languages.

    Features:
        - Fully offline translation
        - Good quality for common language pairs
        - Smaller model sizes than NLLB
        - Easy to install and use

    Limitations:
        - Fewer language pairs than NLLB (no direct support for some languages)
        - May require intermediate translation through English for some pairs
        - Language packs must be downloaded separately

    Example:
        engine = ArgosEngine()
        engine.load_model()

        result = engine.translate("Bonjour le monde", "fr", "en")
        print(result.translated_text)  # "Hello world"
    """

    # Language codes supported by Argos (ISO 639-1)
    # This list may expand as more language packs are released
    SUPPORTED_LANGUAGES = [
        "ar",  # Arabic
        "az",  # Azerbaijani
        "bg",  # Bulgarian
        "bn",  # Bengali
        "ca",  # Catalan
        "cs",  # Czech
        "da",  # Danish
        "de",  # German
        "el",  # Greek
        "en",  # English
        "eo",  # Esperanto
        "es",  # Spanish
        "et",  # Estonian
        "fa",  # Persian
        "fi",  # Finnish
        "fr",  # French
        "ga",  # Irish
        "he",  # Hebrew
        "hi",  # Hindi
        "hr",  # Croatian
        "hu",  # Hungarian
        "id",  # Indonesian
        "it",  # Italian
        "ja",  # Japanese
        "ko",  # Korean
        "lt",  # Lithuanian
        "lv",  # Latvian
        "ms",  # Malay
        "nb",  # Norwegian Bokmål
        "nl",  # Dutch
        "pl",  # Polish
        "pt",  # Portuguese
        "ro",  # Romanian
        "ru",  # Russian
        "sk",  # Slovak
        "sl",  # Slovenian
        "sq",  # Albanian
        "sr",  # Serbian
        "sv",  # Swedish
        "th",  # Thai
        "tl",  # Tagalog
        "tr",  # Turkish
        "uk",  # Ukrainian
        "ur",  # Urdu
        "vi",  # Vietnamese
        "zh",  # Chinese
        "zt",  # Chinese (Traditional)
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize Argos engine.

        Args:
            model_path: Optional path to Argos models directory
        """
        super().__init__(model_path)
        self._installed_languages: set[str] = set()
        self._translation_cache: dict[tuple[str, str], object] = {}
        self._language_mapper = LanguageMapper()

        # Set default target to English ISO code
        self._target_language = "en"

    def get_engine_name(self) -> str:
        """Get human-readable engine name."""
        return "Argos Translate"

    def get_engine_id(self) -> str:
        """Get engine identifier."""
        return "argos"

    def is_available(self) -> bool:
        """Check if Argos Translate is available."""
        try:
            import argostranslate.package
            import argostranslate.translate
            return True
        except ImportError:
            return False

    def get_supported_languages(self) -> list[str]:
        """Get list of supported Argos language codes."""
        return self.SUPPORTED_LANGUAGES.copy()

    def get_installed_languages(self) -> list[str]:
        """
        Get list of languages with installed translation packs.

        Returns:
            List of ISO 639-1 language codes with installed packs
        """
        if not self._model_loaded:
            return []

        try:
            import argostranslate.translate

            installed = set()
            languages = argostranslate.translate.get_installed_languages()
            for lang in languages:
                installed.add(lang.code)

            return sorted(installed)
        except Exception:
            return list(self._installed_languages)

    def load_model(self) -> None:
        """
        Load Argos Translate and discover installed language packs.

        Unlike NLLB which loads a single large model, Argos uses
        separate language packs that are loaded on demand.
        """
        if self._model_loaded:
            logger.info("Argos Translate already initialized")
            return

        try:
            import argostranslate.package
            import argostranslate.translate

            logger.info("Initializing Argos Translate...")

            # Update package index
            try:
                argostranslate.package.update_package_index()
            except Exception as e:
                logger.warning(f"Could not update package index: {e}")

            # Discover installed languages
            languages = argostranslate.translate.get_installed_languages()
            self._installed_languages = {lang.code for lang in languages}

            logger.info(
                f"Argos Translate initialized with {len(self._installed_languages)} languages"
            )

            if not self._installed_languages:
                logger.warning(
                    "No Argos language packs installed. "
                    "Run 'python -m translator.setup --download-models' to install."
                )

            self._model_loaded = True

        except ImportError as e:
            raise RuntimeError(
                "Argos Translate not installed. "
                "Run: pip install argostranslate"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Argos Translate: {e}")
            raise RuntimeError(f"Failed to initialize Argos Translate: {e}") from e

    def unload_model(self) -> None:
        """Clear translation cache and reset state."""
        self._translation_cache.clear()
        self._installed_languages.clear()
        self._model_loaded = False
        logger.info("Argos Translate cache cleared")

    def _get_translator(self, source_lang: str, target_lang: str):
        """
        Get or create a translator for a language pair.

        Args:
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)

        Returns:
            Argos translator object

        Raises:
            ValueError: If language pair is not supported
        """
        cache_key = (source_lang, target_lang)

        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        try:
            import argostranslate.translate

            # Get installed languages
            installed_languages = argostranslate.translate.get_installed_languages()

            # Find source and target language objects
            source_lang_obj = None
            target_lang_obj = None

            for lang in installed_languages:
                if lang.code == source_lang:
                    source_lang_obj = lang
                if lang.code == target_lang:
                    target_lang_obj = lang

            if source_lang_obj is None:
                raise ValueError(
                    f"Source language '{source_lang}' not installed in Argos. "
                    f"Available: {', '.join(sorted(self._installed_languages))}"
                )

            if target_lang_obj is None:
                raise ValueError(
                    f"Target language '{target_lang}' not installed in Argos. "
                    f"Available: {', '.join(sorted(self._installed_languages))}"
                )

            # Get translation object
            translation = source_lang_obj.get_translation(target_lang_obj)

            if translation is None:
                # Try to find a path through English
                if source_lang != "en" and target_lang != "en":
                    logger.info(
                        f"No direct translation {source_lang}->{target_lang}, "
                        "will translate via English"
                    )
                    # We'll handle this in translate() method
                    self._translation_cache[cache_key] = None
                    return None
                else:
                    raise ValueError(
                        f"No translation available for {source_lang} -> {target_lang}"
                    )

            self._translation_cache[cache_key] = translation
            return translation

        except ImportError:
            raise RuntimeError("Argos Translate not properly installed")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate a single text using Argos.

        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1, e.g., 'ru')
            target_lang: Target language code (defaults to 'en')

        Returns:
            TranslationResult with translated text or error
        """
        self._check_model_loaded()

        if target_lang is None:
            target_lang = self._target_language

        # Handle empty text
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                success=True,
            )

        try:
            # Normalize language codes
            source_lang = source_lang.lower()[:2]  # Take first 2 chars
            target_lang = target_lang.lower()[:2]

            # Same language - return as is
            if source_lang == target_lang:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_lang,
                    target_language=target_lang,
                    success=True,
                )

            # Get translator
            translator = self._get_translator(source_lang, target_lang)

            if translator is None:
                # Try translation via English
                if source_lang != "en" and target_lang != "en":
                    # First translate to English
                    to_en = self._get_translator(source_lang, "en")
                    from_en = self._get_translator("en", target_lang)

                    if to_en is None or from_en is None:
                        return TranslationResult(
                            source_text=text,
                            translated_text=None,
                            source_language=source_lang,
                            target_language=target_lang,
                            success=False,
                            error=f"No translation path from {source_lang} to {target_lang}",
                        )

                    # Two-step translation
                    english = to_en.translate(text)
                    translated = from_en.translate(english)
                else:
                    return TranslationResult(
                        source_text=text,
                        translated_text=None,
                        source_language=source_lang,
                        target_language=target_lang,
                        success=False,
                        error=f"No translation available for {source_lang} -> {target_lang}",
                    )
            else:
                # Direct translation
                translated = translator.translate(text)

            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                success=True,
            )

        except ValueError as e:
            return TranslationResult(
                source_text=text,
                translated_text=None,
                source_language=source_lang,
                target_language=target_lang,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Argos translation error: {e}")
            return TranslationResult(
                source_text=text,
                translated_text=None,
                source_language=source_lang,
                target_language=target_lang,
                success=False,
                error=f"Translation failed: {e}",
            )

    def translate_batch(
        self,
        texts: list[str],
        source_langs: list[str],
        target_lang: Optional[str] = None,
        progress: Optional[ProgressTracker] = None,
    ) -> BatchTranslationResult:
        """
        Translate multiple texts.

        Note: Argos doesn't have native batch support like NLLB,
        so this processes texts sequentially but groups by language
        for efficiency.

        Args:
            texts: List of texts to translate
            source_langs: List of source language codes
            target_lang: Target language (same for all)
            progress: Optional progress tracker

        Returns:
            BatchTranslationResult with all translations
        """
        self._check_model_loaded()

        if target_lang is None:
            target_lang = self._target_language

        start_time = time.time()
        results: list[TranslationResult] = []
        success_count = 0
        failure_count = 0

        if len(texts) != len(source_langs):
            raise ValueError(
                f"texts ({len(texts)}) and source_langs ({len(source_langs)}) "
                "must have same length"
            )

        if not texts:
            return BatchTranslationResult(
                results=[],
                success_count=0,
                failure_count=0,
                processing_time=0.0,
            )

        # Process each text
        for idx, (text, src_lang) in enumerate(zip(texts, source_langs)):
            if progress and progress.is_cancelled():
                results.append(
                    TranslationResult(
                        source_text=text,
                        translated_text=None,
                        source_language=src_lang,
                        target_language=target_lang,
                        success=False,
                        error="Translation cancelled",
                    )
                )
                failure_count += 1
                continue

            # Normalize language code
            normalized_src = src_lang.lower()[:2] if src_lang else "en"

            result = self.translate(text, normalized_src, target_lang)
            results.append(result)

            if result.success:
                success_count += 1
            else:
                failure_count += 1

            if progress:
                progress.update(1, f"Translating {idx + 1}/{len(texts)}...")

        processing_time = time.time() - start_time

        return BatchTranslationResult(
            results=results,
            success_count=success_count,
            failure_count=failure_count,
            processing_time=processing_time,
        )

    def install_language_pack(self, from_code: str, to_code: str) -> bool:
        """
        Install a language pack for translation.

        Args:
            from_code: Source language code (ISO 639-1)
            to_code: Target language code (ISO 639-1)

        Returns:
            True if installation successful
        """
        try:
            import argostranslate.package

            # Update package index
            argostranslate.package.update_package_index()

            # Get available packages
            available = argostranslate.package.get_available_packages()

            # Find matching package
            package = None
            for pkg in available:
                if pkg.from_code == from_code and pkg.to_code == to_code:
                    package = pkg
                    break

            if package is None:
                logger.error(
                    f"No package available for {from_code} -> {to_code}"
                )
                return False

            # Download and install
            logger.info(f"Installing {from_code} -> {to_code} package...")
            download_path = package.download()
            argostranslate.package.install_from_path(download_path)

            # Update installed languages
            self._installed_languages.add(from_code)
            self._installed_languages.add(to_code)

            # Clear translation cache
            self._translation_cache.clear()

            logger.info(f"Installed {from_code} -> {to_code}")
            return True

        except Exception as e:
            logger.error(f"Failed to install language pack: {e}")
            return False

    def get_available_packages(self) -> list[tuple[str, str]]:
        """
        Get list of available language packages.

        Returns:
            List of (from_code, to_code) tuples
        """
        try:
            import argostranslate.package

            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()

            return [(pkg.from_code, pkg.to_code) for pkg in available]
        except Exception as e:
            logger.error(f"Failed to get available packages: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            "installed_languages": sorted(self._installed_languages),
            "num_installed": len(self._installed_languages),
            "cache_size": len(self._translation_cache),
        })
        return info


# Register with factory
TranslationEngineFactory.register("argos", ArgosEngine)
