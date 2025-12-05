"""
Language detection using fastText LID218 model.

This module provides automatic language detection for text using Meta's
fastText language identification model (LID218), which supports 218
languages with high accuracy.

Model: https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin (~130 MB)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .languages import LanguageMapper, is_english

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of language detection for a single text."""

    # Detected language code (fastText format)
    language_code: str

    # NLLB-compatible language code
    nllb_code: str

    # Argos-compatible language code (ISO 639-1)
    argos_code: str

    # Human-readable language name
    language_name: str

    # Confidence score (0-1)
    confidence: float

    # Whether detection was successful
    success: bool

    # Error message if detection failed
    error: Optional[str] = None

    @property
    def is_english(self) -> bool:
        """Check if detected language is English."""
        return self.nllb_code == "eng_Latn"


@dataclass
class ColumnDetectionResult:
    """Aggregated language detection result for a column."""

    # Column name
    column_name: str

    # Dominant language code (NLLB format)
    dominant_language: str

    # Argos-compatible code
    argos_code: str

    # Human-readable name
    language_name: str

    # Average confidence across samples
    average_confidence: float

    # Number of samples analyzed
    sample_count: int

    # Detected column type
    column_type: str  # 'foreign_text', 'english', 'numeric', 'date', 'empty', 'mixed'

    # Whether this column should be translated
    should_translate: bool

    # Individual detection results (for debugging)
    sample_results: list[DetectionResult]


class LanguageDetector:
    """
    Language detector using fastText LID218 model.

    Provides high-accuracy language detection for 218 languages,
    with special handling for multilingual content and column-level
    analysis for CSV processing.

    Features:
        - Single text detection
        - Batch detection
        - Column-level analysis with type inference
        - Confidence thresholds
        - Fallback language support

    Example:
        detector = LanguageDetector("models/lid218e.bin")
        detector.load_model()

        result = detector.detect("Привет мир")
        print(result.language_name)  # "Russian"
        print(result.nllb_code)  # "rus_Cyrl"
    """

    # fastText label prefix
    LABEL_PREFIX = "__label__"

    # Common date patterns for detection
    DATE_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}$",  # ISO format
        r"^\d{2}/\d{2}/\d{4}$",  # US format
        r"^\d{2}\.\d{2}\.\d{4}$",  # EU format
        r"^\d{2}-\d{2}-\d{4}$",  # Alternate format
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # Short format
    ]

    # Minimum text length for reliable detection
    MIN_TEXT_LENGTH = 3

    # Sample size for column analysis
    DEFAULT_SAMPLE_SIZE = 100

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        fallback_language: str = "eng_Latn",
    ) -> None:
        """
        Initialize language detector.

        Args:
            model_path: Path to fastText LID218 model file
            confidence_threshold: Minimum confidence to accept detection
            fallback_language: Language code to use when detection fails
        """
        self._model_path = model_path
        self._model = None
        self._model_loaded = False
        self._confidence_threshold = confidence_threshold
        self._fallback_language = fallback_language
        self._language_mapper = LanguageMapper()

        # Compiled date patterns
        self._date_regexes = [re.compile(p) for p in self.DATE_PATTERNS]

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the fastText LID model.

        Args:
            model_path: Optional override for model path
        """
        if self._model_loaded:
            logger.info("Language detection model already loaded")
            return

        path = model_path or self._model_path
        if path is None:
            path = "models/lid218e.bin"

        try:
            import fasttext

            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Language detection model not found at {path}. "
                    "Run 'python -m translator.setup --download-models' to download."
                )

            logger.info(f"Loading language detection model from {path}...")

            # Suppress fastText warnings
            fasttext.FastText.eprint = lambda x: None

            self._model = fasttext.load_model(str(path))
            self._model_loaded = True

            logger.info("Language detection model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "fastText not installed. Run: pip install fasttext"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load language detection model: {e}")
            raise RuntimeError(
                f"Failed to load language detection model: {e}"
            ) from e

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_loaded = False
        logger.info("Language detection model unloaded")

    def detect(self, text: str) -> DetectionResult:
        """
        Detect the language of a single text.

        Args:
            text: Text to analyze

        Returns:
            DetectionResult with detected language and confidence
        """
        if not self._model_loaded:
            raise RuntimeError(
                "Language detection model not loaded. Call load_model() first."
            )

        # Handle empty or very short text
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return self._create_fallback_result()

        # Clean text for detection
        clean_text = self._clean_text(text)
        if len(clean_text) < self.MIN_TEXT_LENGTH:
            return self._create_fallback_result()

        try:
            # Run prediction
            predictions = self._model.predict(clean_text, k=1)
            label = predictions[0][0]
            confidence = float(predictions[1][0])

            # Extract language code from label
            lang_code = label.replace(self.LABEL_PREFIX, "")

            # Map to NLLB code
            nllb_code = self._map_to_nllb(lang_code)
            lang_info = self._language_mapper.get_language(nllb_code)

            if lang_info:
                return DetectionResult(
                    language_code=lang_code,
                    nllb_code=lang_info.nllb_code,
                    argos_code=lang_info.argos_code,
                    language_name=lang_info.name,
                    confidence=confidence,
                    success=True,
                )
            else:
                # Unknown language, use code directly
                return DetectionResult(
                    language_code=lang_code,
                    nllb_code=nllb_code,
                    argos_code=lang_code[:2] if len(lang_code) >= 2 else lang_code,
                    language_name=f"Unknown ({lang_code})",
                    confidence=confidence,
                    success=True,
                )

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return DetectionResult(
                language_code=self._fallback_language,
                nllb_code=self._fallback_language,
                argos_code="en",
                language_name="English (fallback)",
                confidence=0.0,
                success=False,
                error=str(e),
            )

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        """
        Detect languages for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of DetectionResult objects
        """
        return [self.detect(text) for text in texts]

    def analyze_column(
        self,
        values: list[str],
        column_name: str = "",
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> ColumnDetectionResult:
        """
        Analyze a column to determine its type and language.

        This method samples values from the column to determine:
        - Whether it contains text that needs translation
        - The dominant language of the text
        - Whether it's numeric, date, or already English

        Args:
            values: All values in the column
            column_name: Name of the column (for reporting)
            sample_size: Maximum number of values to sample

        Returns:
            ColumnDetectionResult with analysis
        """
        if not values:
            return ColumnDetectionResult(
                column_name=column_name,
                dominant_language="",
                argos_code="",
                language_name="",
                average_confidence=0.0,
                sample_count=0,
                column_type="empty",
                should_translate=False,
                sample_results=[],
            )

        # Filter out empty values
        non_empty = [v for v in values if v and str(v).strip()]

        if not non_empty:
            return ColumnDetectionResult(
                column_name=column_name,
                dominant_language="",
                argos_code="",
                language_name="",
                average_confidence=0.0,
                sample_count=0,
                column_type="empty",
                should_translate=False,
                sample_results=[],
            )

        # Check for numeric column
        if self._is_numeric_column(non_empty):
            return ColumnDetectionResult(
                column_name=column_name,
                dominant_language="",
                argos_code="",
                language_name="",
                average_confidence=1.0,
                sample_count=len(non_empty),
                column_type="numeric",
                should_translate=False,
                sample_results=[],
            )

        # Check for date column
        if self._is_date_column(non_empty):
            return ColumnDetectionResult(
                column_name=column_name,
                dominant_language="",
                argos_code="",
                language_name="",
                average_confidence=1.0,
                sample_count=len(non_empty),
                column_type="date",
                should_translate=False,
                sample_results=[],
            )

        # Sample values for language detection
        import random
        if len(non_empty) > sample_size:
            sample = random.sample(non_empty, sample_size)
        else:
            sample = non_empty

        # Detect languages
        results = self.detect_batch([str(v) for v in sample])
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return ColumnDetectionResult(
                column_name=column_name,
                dominant_language=self._fallback_language,
                argos_code="en",
                language_name="Unknown",
                average_confidence=0.0,
                sample_count=len(sample),
                column_type="mixed",
                should_translate=True,
                sample_results=results,
            )

        # Count languages
        lang_counts: dict[str, int] = {}
        lang_confidences: dict[str, list[float]] = {}

        for result in successful_results:
            lang = result.nllb_code
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if lang not in lang_confidences:
                lang_confidences[lang] = []
            lang_confidences[lang].append(result.confidence)

        # Find dominant language
        dominant = max(lang_counts.keys(), key=lambda k: lang_counts[k])
        dominant_count = lang_counts[dominant]
        total_count = len(successful_results)
        dominance_ratio = dominant_count / total_count

        # Calculate average confidence for dominant language
        avg_confidence = sum(lang_confidences[dominant]) / len(lang_confidences[dominant])

        # Determine if this is English
        is_english_col = dominant == "eng_Latn"

        # Determine column type
        if is_english_col:
            col_type = "english"
            should_translate = False
        elif dominance_ratio < 0.5:
            col_type = "mixed"
            should_translate = True
        else:
            col_type = "foreign_text"
            should_translate = True

        # Get language info
        lang_info = self._language_mapper.get_language(dominant)
        lang_name = lang_info.name if lang_info else f"Unknown ({dominant})"
        argos_code = lang_info.argos_code if lang_info else dominant[:2]

        return ColumnDetectionResult(
            column_name=column_name,
            dominant_language=dominant,
            argos_code=argos_code,
            language_name=lang_name,
            average_confidence=avg_confidence,
            sample_count=len(sample),
            column_type=col_type,
            should_translate=should_translate,
            sample_results=results,
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean text for language detection.

        Removes URLs, email addresses, and excessive whitespace
        that can confuse the detector.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Convert to string
        text = str(text)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _map_to_nllb(self, fasttext_code: str) -> str:
        """
        Map fastText language code to NLLB format.

        Args:
            fasttext_code: Language code from fastText (e.g., 'rus', 'ukr')

        Returns:
            NLLB format code (e.g., 'rus_Cyrl', 'ukr_Cyrl')
        """
        # fastText uses various formats, try to match
        lang_info = self._language_mapper.get_language(fasttext_code)

        if lang_info:
            return lang_info.nllb_code

        # Try common mappings for fastText codes
        # fastText often uses ISO 639-3 without script
        code_mappings = {
            "rus": "rus_Cyrl",
            "ukr": "ukr_Cyrl",
            "bel": "bel_Cyrl",
            "bul": "bul_Cyrl",
            "mkd": "mkd_Cyrl",
            "srp": "srp_Cyrl",
            "eng": "eng_Latn",
            "deu": "deu_Latn",
            "fra": "fra_Latn",
            "spa": "spa_Latn",
            "ita": "ita_Latn",
            "por": "por_Latn",
            "nld": "nld_Latn",
            "pol": "pol_Latn",
            "ces": "ces_Latn",
            "slk": "slk_Latn",
            "hrv": "hrv_Latn",
            "slv": "slv_Latn",
            "bos": "bos_Latn",
            "lit": "lit_Latn",
            "lvs": "lvs_Latn",
            "est": "est_Latn",
            "fin": "fin_Latn",
            "swe": "swe_Latn",
            "dan": "dan_Latn",
            "nob": "nob_Latn",
            "nno": "nno_Latn",
            "isl": "isl_Latn",
            "hun": "hun_Latn",
            "ron": "ron_Latn",
            "ell": "ell_Grek",
            "tur": "tur_Latn",
            "arb": "arb_Arab",
            "heb": "heb_Hebr",
            "pes": "pes_Arab",
            "hin": "hin_Deva",
            "zho": "zho_Hans",
            "jpn": "jpn_Jpan",
            "kor": "kor_Hang",
            "vie": "vie_Latn",
            "tha": "tha_Thai",
        }

        if fasttext_code in code_mappings:
            return code_mappings[fasttext_code]

        # Return as-is with Latin script assumption
        return f"{fasttext_code}_Latn"

    def _create_fallback_result(self) -> DetectionResult:
        """Create a fallback detection result."""
        lang_info = self._language_mapper.get_language(self._fallback_language)
        return DetectionResult(
            language_code=self._fallback_language,
            nllb_code=self._fallback_language,
            argos_code=lang_info.argos_code if lang_info else "en",
            language_name=lang_info.name if lang_info else "English",
            confidence=0.0,
            success=True,  # Fallback is still a valid result
        )

    def _is_numeric_column(self, values: list[str], threshold: float = 0.9) -> bool:
        """
        Check if a column is primarily numeric.

        Args:
            values: Column values
            threshold: Proportion that must be numeric

        Returns:
            True if column is numeric
        """
        numeric_count = 0
        for value in values:
            try:
                # Try to parse as number
                cleaned = str(value).replace(",", "").replace(" ", "")
                float(cleaned)
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return (numeric_count / len(values)) >= threshold if values else False

    def _is_date_column(self, values: list[str], threshold: float = 0.8) -> bool:
        """
        Check if a column is primarily dates.

        Args:
            values: Column values
            threshold: Proportion that must be dates

        Returns:
            True if column is dates
        """
        date_count = 0
        for value in values:
            str_value = str(value).strip()
            for pattern in self._date_regexes:
                if pattern.match(str_value):
                    date_count += 1
                    break

        return (date_count / len(values)) >= threshold if values else False

    def is_english_text(self, text: str) -> bool:
        """
        Quick check if text appears to be English.

        Args:
            text: Text to check

        Returns:
            True if text is likely English
        """
        result = self.detect(text)
        return result.is_english and result.confidence >= self._confidence_threshold

    def __enter__(self) -> "LanguageDetector":
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload_model()
