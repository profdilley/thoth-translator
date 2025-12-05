"""
CSV/Excel processing logic for THOTH.

This module handles reading, analyzing, and translating tabular data
from CSV and Excel files, with support for large files and progress tracking.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .config import Config
from .detector import ColumnDetectionResult, LanguageDetector
from .engine_base import TranslationEngine, TranslationEngineFactory
from .languages import LanguageMapper
from .progress import ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a single column in the dataset."""

    # Column name
    name: str

    # Index in the dataframe
    index: int

    # Detected language (NLLB format)
    detected_language: str

    # Argos language code
    argos_code: str

    # Human-readable language name
    language_name: str

    # Column type (foreign_text, english, numeric, date, empty, mixed)
    column_type: str

    # Whether column is selected for translation
    selected: bool

    # Detection confidence
    confidence: float

    # Sample values for preview
    sample_values: list[str] = field(default_factory=list)

    # Override language (if user specified)
    override_language: Optional[str] = None

    @property
    def effective_language(self) -> str:
        """Get the language to use for translation (override or detected)."""
        return self.override_language or self.detected_language


@dataclass
class ProcessingResult:
    """Result of processing a file."""

    # Whether processing was successful
    success: bool

    # Output file path (if successful)
    output_path: Optional[str]

    # Number of rows processed
    rows_processed: int

    # Number of columns translated
    columns_translated: int

    # Total cells translated
    cells_translated: int

    # Processing time in seconds
    processing_time: float

    # Error message (if failed)
    error: Optional[str] = None

    # Warnings encountered during processing
    warnings: list[str] = field(default_factory=list)


class CSVProcessor:
    """
    Processor for translating CSV and Excel files.

    Handles the complete workflow of:
    1. Loading files (CSV, XLSX, XLS)
    2. Analyzing columns for language detection
    3. Translating selected columns
    4. Saving results with new translated columns

    Example:
        processor = CSVProcessor(config)
        processor.load_file("input.csv")

        # Analyze columns
        columns = processor.analyze_columns()

        # Select columns to translate
        processor.set_column_selection(["description", "notes"])

        # Translate
        result = processor.translate(engine, progress_callback)

        # Save
        processor.save("output.csv")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        detector: Optional[LanguageDetector] = None,
    ) -> None:
        """
        Initialize the processor.

        Args:
            config: Configuration settings
            detector: Language detector (created if not provided)
        """
        self._config = config or Config()
        self._detector = detector
        self._df: Optional[pd.DataFrame] = None
        self._input_path: Optional[Path] = None
        self._columns: list[ColumnInfo] = []
        self._language_mapper = LanguageMapper()
        self._translated_columns: dict[str, list[str]] = {}

    @property
    def is_loaded(self) -> bool:
        """Check if a file is loaded."""
        return self._df is not None

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Get the loaded dataframe."""
        return self._df

    @property
    def columns(self) -> list[ColumnInfo]:
        """Get analyzed column information."""
        return self._columns

    @property
    def row_count(self) -> int:
        """Get number of rows in loaded file."""
        return len(self._df) if self._df is not None else 0

    @property
    def column_count(self) -> int:
        """Get number of columns in loaded file."""
        return len(self._df.columns) if self._df is not None else 0

    def load_file(self, file_path: str) -> None:
        """
        Load a CSV or Excel file.

        Args:
            file_path: Path to the file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        try:
            if suffix == ".csv":
                # Try different encodings
                for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                    try:
                        self._df = pd.read_csv(path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(
                        f"Could not decode CSV file with any supported encoding"
                    )

            elif suffix in (".xlsx", ".xls"):
                self._df = pd.read_excel(path)

            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    "Supported formats: .csv, .xlsx, .xls"
                )

            self._input_path = path
            self._columns = []
            self._translated_columns = {}

            logger.info(
                f"Loaded file: {path.name} "
                f"({len(self._df)} rows, {len(self._df.columns)} columns)"
            )

        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

    def analyze_columns(
        self,
        progress: Optional[ProgressTracker] = None,
    ) -> list[ColumnInfo]:
        """
        Analyze all columns to detect languages and types.

        Args:
            progress: Optional progress tracker

        Returns:
            List of ColumnInfo objects for all columns
        """
        if self._df is None:
            raise RuntimeError("No file loaded. Call load_file() first.")

        # Ensure detector is available
        if self._detector is None:
            self._detector = LanguageDetector(
                model_path=str(self._config.get_lid_path()),
                confidence_threshold=self._config.detection.confidence_threshold,
                fallback_language=self._config.detection.fallback_language,
            )

        if not self._detector.is_loaded:
            self._detector.load_model()

        self._columns = []
        total_columns = len(self._df.columns)

        if progress:
            progress.start(total_columns, "Analyzing columns...")

        for idx, col_name in enumerate(self._df.columns):
            if progress and progress.is_cancelled():
                break

            # Get column values
            values = self._df[col_name].astype(str).tolist()

            # Analyze with detector
            detection = self._detector.analyze_column(values, col_name)

            # Determine if column should be selected
            should_select = self._should_auto_select(detection)

            # Get sample values for preview
            non_empty = [v for v in values if v and str(v).strip() and v != "nan"]
            samples = non_empty[:5] if non_empty else []

            # Check for config overrides
            override_lang = self._config.column_overrides.get(col_name)

            col_info = ColumnInfo(
                name=col_name,
                index=idx,
                detected_language=detection.dominant_language,
                argos_code=detection.argos_code,
                language_name=detection.language_name,
                column_type=detection.column_type,
                selected=should_select,
                confidence=detection.average_confidence,
                sample_values=samples,
                override_language=override_lang,
            )

            self._columns.append(col_info)

            if progress:
                progress.update(1, f"Analyzed column: {col_name}")

        if progress:
            progress.complete(f"Analyzed {len(self._columns)} columns")

        return self._columns

    def _should_auto_select(self, detection: ColumnDetectionResult) -> bool:
        """
        Determine if a column should be auto-selected for translation.

        Args:
            detection: Column detection result

        Returns:
            True if column should be selected
        """
        defaults = self._config.column_defaults

        # Skip empty columns
        if detection.column_type == "empty" and defaults.skip_empty:
            return False

        # Skip numeric columns
        if detection.column_type == "numeric" and defaults.skip_numeric:
            return False

        # Skip date columns
        if detection.column_type == "date" and defaults.skip_dates:
            return False

        # Skip English columns
        if detection.column_type == "english" and defaults.skip_english:
            return False

        # Auto-select foreign text
        if detection.column_type == "foreign_text" and defaults.auto_select_foreign_text:
            return True

        # Select mixed columns (might have some foreign text)
        if detection.column_type == "mixed":
            return True

        return False

    def set_column_selection(self, columns: list[str]) -> None:
        """
        Set which columns to translate.

        Args:
            columns: List of column names to translate
        """
        column_set = set(columns)
        for col in self._columns:
            col.selected = col.name in column_set

    def set_column_language(self, column_name: str, language_code: str) -> None:
        """
        Override the detected language for a column.

        Args:
            column_name: Name of the column
            language_code: NLLB or Argos language code
        """
        for col in self._columns:
            if col.name == column_name:
                col.override_language = language_code
                break

    def get_selected_columns(self) -> list[ColumnInfo]:
        """Get list of columns selected for translation."""
        return [col for col in self._columns if col.selected]

    def translate(
        self,
        engine: TranslationEngine,
        progress: Optional[ProgressTracker] = None,
        target_language: str = "eng_Latn",
    ) -> ProcessingResult:
        """
        Translate selected columns using the specified engine.

        Args:
            engine: Translation engine to use
            progress: Optional progress tracker
            target_language: Target language code

        Returns:
            ProcessingResult with translation statistics
        """
        if self._df is None:
            return ProcessingResult(
                success=False,
                output_path=None,
                rows_processed=0,
                columns_translated=0,
                cells_translated=0,
                processing_time=0.0,
                error="No file loaded",
            )

        selected = self.get_selected_columns()
        if not selected:
            return ProcessingResult(
                success=False,
                output_path=None,
                rows_processed=0,
                columns_translated=0,
                cells_translated=0,
                processing_time=0.0,
                error="No columns selected for translation",
            )

        start_time = time.time()
        warnings: list[str] = []
        cells_translated = 0

        # Calculate total cells to translate
        total_cells = sum(
            len([v for v in self._df[col.name].astype(str)
                 if v and str(v).strip() and v != "nan"])
            for col in selected
        )

        if progress:
            progress.start(total_cells, "Starting translation...")

        # Convert target language to engine format
        if engine.get_engine_id() == "argos":
            # Argos uses ISO 639-1 codes
            target_lang = self._language_mapper.to_argos(target_language) or "en"
        else:
            target_lang = target_language

        try:
            # Process each selected column
            for col in selected:
                if progress and progress.is_cancelled():
                    break

                logger.info(f"Translating column: {col.name}")

                # Get source language
                source_lang = col.effective_language

                # Convert to engine format
                if engine.get_engine_id() == "argos":
                    source_lang = col.argos_code or \
                                  self._language_mapper.to_argos(source_lang) or \
                                  source_lang[:2]

                # Get column values
                values = self._df[col.name].astype(str).tolist()

                # Prepare texts and languages for batch translation
                texts_to_translate = []
                indices_to_translate = []
                source_langs = []

                for idx, value in enumerate(values):
                    if value and str(value).strip() and value != "nan":
                        texts_to_translate.append(value)
                        indices_to_translate.append(idx)
                        source_langs.append(source_lang)

                # Translate in batches
                translated_values = [""] * len(values)
                batch_size = self._config.performance.batch_size

                for batch_start in range(0, len(texts_to_translate), batch_size):
                    if progress and progress.is_cancelled():
                        break

                    batch_end = min(batch_start + batch_size, len(texts_to_translate))
                    batch_texts = texts_to_translate[batch_start:batch_end]
                    batch_langs = source_langs[batch_start:batch_end]
                    batch_indices = indices_to_translate[batch_start:batch_end]

                    # Translate batch
                    result = engine.translate_batch(
                        batch_texts,
                        batch_langs,
                        target_lang,
                    )

                    # Store results
                    for i, trans_result in enumerate(result.results):
                        orig_idx = batch_indices[i]
                        if trans_result.success and trans_result.translated_text:
                            translated_values[orig_idx] = trans_result.translated_text
                            cells_translated += 1
                        else:
                            # Keep original on failure
                            translated_values[orig_idx] = texts_to_translate[batch_start + i]
                            if trans_result.error:
                                warnings.append(
                                    f"Row {orig_idx + 1}, column '{col.name}': {trans_result.error}"
                                )

                    if progress:
                        progress.update(
                            len(batch_texts),
                            f"Translating {col.name}..."
                        )

                # Store translated column
                self._translated_columns[col.name] = translated_values

            processing_time = time.time() - start_time

            if progress:
                if progress.is_cancelled():
                    progress.set_message("Translation cancelled")
                else:
                    progress.complete(
                        f"Translated {cells_translated} cells in {processing_time:.1f}s"
                    )

            return ProcessingResult(
                success=not (progress and progress.is_cancelled()),
                output_path=None,  # Set when saved
                rows_processed=len(self._df),
                columns_translated=len(selected),
                cells_translated=cells_translated,
                processing_time=processing_time,
                warnings=warnings[:10],  # Limit warnings
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ProcessingResult(
                success=False,
                output_path=None,
                rows_processed=0,
                columns_translated=0,
                cells_translated=0,
                processing_time=time.time() - start_time,
                error=str(e),
            )

    def save(
        self,
        output_path: Optional[str] = None,
        add_suffix: bool = True,
    ) -> str:
        """
        Save the translated data to a file.

        Creates new columns with '_en' suffix containing translations,
        preserving the original columns.

        Args:
            output_path: Output file path (auto-generated if None)
            add_suffix: Whether to add '_translated' suffix to filename

        Returns:
            Path to the saved file
        """
        if self._df is None:
            raise RuntimeError("No file loaded")

        if not self._translated_columns:
            raise RuntimeError("No translations to save")

        # Determine output path
        if output_path is None:
            if self._input_path is None:
                raise RuntimeError("No input path available")

            stem = self._input_path.stem
            suffix = "_translated" if add_suffix else ""
            ext = self._input_path.suffix
            output_path = str(self._input_path.parent / f"{stem}{suffix}{ext}")

        # Create output dataframe with translated columns inserted adjacent to source
        output_df = pd.DataFrame()
        
        for col_name in self._df.columns:
            # Add original column
            output_df[col_name] = self._df[col_name]
            
            # If this column was translated, add translation immediately after
            if col_name in self._translated_columns:
                new_col_name = f"{col_name}_en"
                output_df[new_col_name] = self._translated_columns[col_name]

        # Save based on extension
        output_path_obj = Path(output_path)
        suffix = output_path_obj.suffix.lower()

        if suffix == ".csv":
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif suffix in (".xlsx", ".xls"):
            output_df.to_excel(output_path, index=False)
        else:
            # Default to CSV
            output_path = str(output_path_obj.with_suffix(".csv"))
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved translated file to: {output_path}")
        return output_path

    def get_preview(
        self,
        column_name: str,
        engine: TranslationEngine,
        num_rows: int = 5,
    ) -> list[tuple[str, str]]:
        """
        Get a preview of translations for a column.

        Args:
            column_name: Name of column to preview
            engine: Translation engine to use
            num_rows: Number of rows to preview

        Returns:
            List of (original, translated) tuples
        """
        if self._df is None:
            return []

        # Find column info
        col_info = None
        for col in self._columns:
            if col.name == column_name:
                col_info = col
                break

        if col_info is None:
            return []

        # Get sample values
        values = self._df[column_name].astype(str).tolist()
        non_empty = [(i, v) for i, v in enumerate(values)
                     if v and str(v).strip() and v != "nan"][:num_rows]

        if not non_empty:
            return []

        # Translate samples
        texts = [v for _, v in non_empty]
        source_lang = col_info.effective_language

        # Convert to engine format
        if engine.get_engine_id() == "argos":
            source_lang = col_info.argos_code or source_lang[:2]
            target_lang = "en"
        else:
            target_lang = "eng_Latn"

        source_langs = [source_lang] * len(texts)

        result = engine.translate_batch(texts, source_langs, target_lang)

        # Build preview list
        preview = []
        for text, trans_result in zip(texts, result.results):
            if trans_result.success and trans_result.translated_text:
                preview.append((text, trans_result.translated_text))
            else:
                preview.append((text, "[Translation failed]"))

        return preview

    def reset(self) -> None:
        """Reset processor state."""
        self._df = None
        self._input_path = None
        self._columns = []
        self._translated_columns = {}


def create_processor(config: Optional[Config] = None) -> CSVProcessor:
    """
    Factory function to create a configured processor.

    Args:
        config: Configuration settings

    Returns:
        Configured CSVProcessor instance
    """
    return CSVProcessor(config=config)
