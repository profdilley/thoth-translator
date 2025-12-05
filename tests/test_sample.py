"""
Basic tests for THOTH translation tool.

These tests verify core functionality without requiring model downloads.
For full integration tests with translation, ensure models are installed.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLanguageMapper:
    """Tests for the LanguageMapper class."""

    def test_import(self):
        """Test that LanguageMapper can be imported."""
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()
        assert mapper is not None

    def test_to_nllb(self):
        """Test conversion to NLLB codes."""
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()

        assert mapper.to_nllb("ru") == "rus_Cyrl"
        assert mapper.to_nllb("uk") == "ukr_Cyrl"
        assert mapper.to_nllb("de") == "deu_Latn"
        assert mapper.to_nllb("fr") == "fra_Latn"
        assert mapper.to_nllb("zh") == "zho_Hans"

    def test_to_argos(self):
        """Test conversion to Argos codes."""
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()

        assert mapper.to_argos("rus_Cyrl") == "ru"
        assert mapper.to_argos("ukr_Cyrl") == "uk"
        assert mapper.to_argos("deu_Latn") == "de"
        assert mapper.to_argos("fra_Latn") == "fr"

    def test_get_name(self):
        """Test getting language names."""
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()

        assert mapper.get_name("ru") == "Russian"
        assert mapper.get_name("rus_Cyrl") == "Russian"
        assert mapper.get_name("uk") == "Ukrainian"
        assert mapper.get_name("de") == "German"

    def test_get_all_languages(self):
        """Test getting all languages."""
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()

        languages = mapper.get_all_languages()
        assert len(languages) > 50
        assert any(lang.name == "Russian" for lang in languages)
        assert any(lang.name == "English" for lang in languages)

    def test_is_english(self):
        """Test English detection."""
        from translator.languages import is_english

        assert is_english("eng_Latn") is True
        assert is_english("en") is True
        assert is_english("rus_Cyrl") is False
        assert is_english("de") is False


class TestConfig:
    """Tests for the Config class."""

    def test_default_config(self):
        """Test creating default configuration."""
        from translator.config import Config

        config = Config()
        assert config.default_engine in ("nllb", "argos")
        assert config.performance.batch_size > 0
        assert 0 <= config.detection.confidence_threshold <= 1

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from translator.config import Config

        config = Config()
        config_dict = config.to_dict()

        assert "default_engine" in config_dict
        assert "performance" in config_dict
        assert "detection" in config_dict
        assert "models" in config_dict

    def test_config_validation(self):
        """Test configuration validation."""
        from translator.config import Config

        config = Config()
        errors = config.validate()
        assert len(errors) == 0  # Default config should be valid

        # Test invalid engine
        config.default_engine = "invalid"
        errors = config.validate()
        assert len(errors) > 0


class TestProgress:
    """Tests for the ProgressTracker class."""

    def test_progress_tracker(self):
        """Test basic progress tracking."""
        from translator.progress import ProgressTracker

        tracker = ProgressTracker(total=100)
        tracker.start(message="Testing")

        assert tracker.state.total == 100
        assert tracker.state.current == 0
        assert tracker.state.percentage == 0.0

        tracker.update(50)
        assert tracker.state.current == 50
        assert tracker.state.percentage == 50.0

        tracker.complete()
        assert tracker.state.complete is True

    def test_progress_cancellation(self):
        """Test progress cancellation."""
        from translator.progress import ProgressTracker

        tracker = ProgressTracker(total=100)
        assert tracker.is_cancelled() is False

        tracker.cancel()
        assert tracker.is_cancelled() is True

    def test_progress_bar_format(self):
        """Test progress bar formatting."""
        from translator.progress import format_progress_bar

        bar = format_progress_bar(0)
        assert "░" in bar

        bar = format_progress_bar(50)
        assert "█" in bar
        assert "░" in bar

        bar = format_progress_bar(100)
        assert "░" not in bar


class TestEngineFactory:
    """Tests for the TranslationEngineFactory."""

    def test_factory_registration(self):
        """Test that engines are registered."""
        from translator.engine_base import TranslationEngineFactory

        engines = TranslationEngineFactory.get_available_engines()
        assert "nllb" in engines
        assert "argos" in engines

    def test_factory_create(self):
        """Test engine creation."""
        from translator.engine_base import TranslationEngineFactory

        engine = TranslationEngineFactory.create("nllb")
        assert engine.get_engine_id() == "nllb"
        assert engine.get_engine_name() == "NLLB-200"

        engine = TranslationEngineFactory.create("argos")
        assert engine.get_engine_id() == "argos"
        assert engine.get_engine_name() == "Argos Translate"

    def test_factory_invalid_engine(self):
        """Test invalid engine raises error."""
        from translator.engine_base import TranslationEngineFactory

        with pytest.raises(ValueError):
            TranslationEngineFactory.create("invalid_engine")


class TestProcessor:
    """Tests for the CSVProcessor class."""

    def test_processor_creation(self):
        """Test creating a processor."""
        from translator.processor import CSVProcessor

        processor = CSVProcessor()
        assert processor.is_loaded is False
        assert processor.row_count == 0

    def test_load_sample_csv(self):
        """Test loading the sample CSV file."""
        from translator.processor import CSVProcessor

        sample_path = Path(__file__).parent / "sample.csv"
        if not sample_path.exists():
            pytest.skip("Sample CSV not found")

        processor = CSVProcessor()
        processor.load_file(str(sample_path))

        assert processor.is_loaded is True
        assert processor.row_count > 0
        assert processor.column_count > 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        from translator.processor import CSVProcessor

        processor = CSVProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_file("nonexistent_file.csv")


class TestCSVCreation:
    """Test creating and processing CSV files."""

    def test_create_and_load_csv(self):
        """Test creating a CSV file and loading it."""
        import pandas as pd
        from translator.processor import CSVProcessor

        # Create a temporary CSV
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write("id,name,description\n")
            f.write("1,Test,Hello world\n")
            f.write("2,Test2,Bonjour monde\n")
            temp_path = f.name

        try:
            processor = CSVProcessor()
            processor.load_file(temp_path)

            assert processor.is_loaded
            assert processor.row_count == 2
            assert processor.column_count == 3

        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests (require models)."""

    @pytest.mark.skipif(
        not Path("models/lid218e.bin").exists(),
        reason="LID model not installed",
    )
    def test_language_detection(self):
        """Test language detection with real model."""
        from translator.detector import LanguageDetector

        detector = LanguageDetector()
        detector.load_model()

        result = detector.detect("Привет мир")
        assert result.success
        assert result.nllb_code == "rus_Cyrl"

        result = detector.detect("Hello world")
        assert result.success
        assert result.nllb_code == "eng_Latn"

        detector.unload_model()

    @pytest.mark.skipif(
        not Path("models/nllb-200-distilled-600M").exists(),
        reason="NLLB model not installed",
    )
    def test_nllb_translation(self):
        """Test NLLB translation with real model."""
        from translator.engine_nllb import NLLBEngine

        engine = NLLBEngine()
        engine.load_model()

        result = engine.translate("Bonjour", "fra_Latn", "eng_Latn")
        assert result.success
        assert result.translated_text is not None
        assert len(result.translated_text) > 0

        engine.unload_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
