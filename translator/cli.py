"""
Command-line interface for THOTH translation tool.

This module provides a comprehensive CLI for translating CSV/Excel files
from various languages to English using offline translation models.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .detector import LanguageDetector
from .engine_base import TranslationEngineFactory
from .languages import LanguageMapper
from .processor import CSVProcessor
from .progress import ProgressTracker, format_progress_bar, format_progress_line

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CLI:
    """
    Command-line interface for THOTH.

    Provides a complete CLI experience with:
    - File translation with progress display
    - Column selection and language override
    - Engine selection (NLLB/Argos)
    - Quiet and verbose modes
    """

    def __init__(self) -> None:
        """Initialize CLI."""
        self._config: Optional[Config] = None
        self._progress: Optional[ProgressTracker] = None
        self._cancelled = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle Ctrl+C gracefully."""
        print("\n\nCancellation requested...")
        self._cancelled = True
        if self._progress:
            self._progress.cancel()

    def run(self, args: Optional[list[str]] = None) -> int:
        """
        Run the CLI with given arguments.

        Args:
            args: Command-line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success)
        """
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)

        # Handle logging level
        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif parsed_args.quiet:
            logging.getLogger().setLevel(logging.WARNING)

        # Load configuration
        self._config = Config.load(parsed_args.config)

        # Handle commands
        if parsed_args.test:
            return self._run_tests()

        if parsed_args.input_file:
            return self._translate_file(parsed_args)

        # No file provided, show help
        parser.print_help()
        return 0

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="thoth",
            description="THOTH - Translator for Hybrid Offline Text Handling",
            epilog="For more information, see README.md",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Positional arguments
        parser.add_argument(
            "input_file",
            nargs="?",
            help="Input CSV or Excel file to translate",
        )

        # Output options
        parser.add_argument(
            "-o", "--output",
            dest="output_file",
            help="Output file path (default: input_translated.csv)",
        )

        # Column selection
        parser.add_argument(
            "-c", "--columns",
            dest="columns",
            help="Comma-separated list of columns to translate",
        )

        # Language options
        parser.add_argument(
            "-l", "--force-lang",
            dest="force_lang",
            help="Force source language for all columns (e.g., rus_Cyrl)",
        )

        # Engine selection
        parser.add_argument(
            "-e", "--engine",
            choices=["nllb", "argos"],
            default=None,
            help="Translation engine (default: from config)",
        )

        # Configuration
        parser.add_argument(
            "--config",
            dest="config",
            help="Path to configuration file",
        )

        # Output modes
        parser.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="Minimal output",
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Verbose output (debug logging)",
        )

        # Analysis only
        parser.add_argument(
            "--analyze",
            action="store_true",
            help="Only analyze columns, don't translate",
        )

        # Other commands
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run tests with sample data",
        )

        parser.add_argument(
            "--list-languages",
            action="store_true",
            dest="list_languages",
            help="List all supported languages",
        )

        parser.add_argument(
            "--version",
            action="version",
            version="THOTH 1.0.0",
        )

        return parser

    def _translate_file(self, args) -> int:
        """
        Translate a file based on CLI arguments.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        input_path = Path(args.input_file)

        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return 1

        # Handle list-languages flag
        if args.list_languages:
            self._print_languages()
            return 0

        print(f"\n{'='*60}")
        print("  THOTH - Offline Translation Tool")
        print(f"{'='*60}\n")

        try:
            # Create processor
            processor = CSVProcessor(self._config)

            # Load file
            print(f"Loading file: {input_path.name}")
            processor.load_file(str(input_path))
            print(f"  Rows: {processor.row_count:,}")
            print(f"  Columns: {processor.column_count}")
            print()

            # Load detector
            print("Loading language detection model...")
            detector = LanguageDetector(
                model_path=str(self._config.get_lid_path()),
            )
            detector.load_model()
            print()

            # Analyze columns
            print("Analyzing columns...")
            self._progress = ProgressTracker()
            self._progress.on_progress = self._print_progress
            columns = processor.analyze_columns(self._progress)
            print()

            # Print analysis results
            self._print_column_analysis(columns)

            if args.analyze:
                print("\nAnalysis complete. Exiting (--analyze mode).")
                return 0

            # Handle column selection
            if args.columns:
                column_names = [c.strip() for c in args.columns.split(",")]
                processor.set_column_selection(column_names)
                print(f"\nSelected columns: {', '.join(column_names)}")

            # Handle language override
            if args.force_lang:
                for col in processor.columns:
                    if col.selected:
                        col.override_language = args.force_lang
                print(f"Forcing source language: {args.force_lang}")

            # Check selected columns
            selected = processor.get_selected_columns()
            if not selected:
                print("\nNo columns selected for translation.")
                print("Use --columns to specify columns, or check column analysis above.")
                return 1

            print(f"\nColumns to translate: {len(selected)}")
            for col in selected:
                print(f"  - {col.name} ({col.language_name})")

            # Load translation engine
            engine_name = args.engine or self._config.default_engine
            print(f"\nLoading {engine_name.upper()} translation engine...")

            engine = TranslationEngineFactory.create(engine_name)
            engine.load_model()
            print("  Engine ready.")
            print()

            # Translate
            print("Starting translation...")
            self._progress = ProgressTracker()
            self._progress.on_progress = self._print_progress

            result = processor.translate(engine, self._progress)
            print()

            if self._cancelled:
                print("\nTranslation cancelled by user.")
                return 130  # Standard exit code for SIGINT

            if not result.success:
                print(f"\nTranslation failed: {result.error}")
                return 1

            # Save results
            output_path = args.output_file or None
            saved_path = processor.save(output_path)

            # Print summary
            print(f"\n{'='*60}")
            print("  Translation Complete")
            print(f"{'='*60}")
            print(f"  Output file: {saved_path}")
            print(f"  Rows processed: {result.rows_processed:,}")
            print(f"  Columns translated: {result.columns_translated}")
            print(f"  Cells translated: {result.cells_translated:,}")
            print(f"  Time: {result.processing_time:.1f} seconds")

            if result.warnings:
                print(f"\n  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:5]:
                    print(f"    - {warning}")
                if len(result.warnings) > 5:
                    print(f"    ... and {len(result.warnings) - 5} more")

            print()

            # Cleanup
            engine.unload_model()
            detector.unload_model()

            return 0

        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _print_progress(self, state) -> None:
        """Print progress update."""
        if state.complete:
            print(f"\r{state.message:<60}")
        else:
            bar = format_progress_bar(state.percentage, width=30)
            line = f"\r{bar} {state.percentage:5.1f}% | {state.message}"
            print(line[:80], end="", flush=True)

    def _print_column_analysis(self, columns: list) -> None:
        """Print column analysis results."""
        print("\nColumn Analysis:")
        print("-" * 60)
        print(f"{'Column':<25} {'Type':<12} {'Language':<15} {'Select'}")
        print("-" * 60)

        for col in columns:
            select_mark = "[X]" if col.selected else "[ ]"
            lang = col.language_name[:14] if col.language_name else "-"
            print(f"{col.name:<25} {col.column_type:<12} {lang:<15} {select_mark}")

        print("-" * 60)

        selected_count = sum(1 for c in columns if c.selected)
        print(f"Total: {len(columns)} columns, {selected_count} selected for translation")

    def _print_languages(self) -> None:
        """Print list of supported languages."""
        mapper = LanguageMapper()
        languages = mapper.get_all_languages()

        print("\nSupported Languages:")
        print("=" * 70)
        print(f"{'Language':<25} {'NLLB Code':<15} {'Argos':<8} {'Family'}")
        print("-" * 70)

        for lang in sorted(languages, key=lambda x: x.name):
            argos = lang.argos_code if lang.argos_supported else "-"
            print(f"{lang.name:<25} {lang.nllb_code:<15} {argos:<8} {lang.family}")

        print("-" * 70)
        print(f"Total: {len(languages)} languages")
        print()

    def _run_tests(self) -> int:
        """Run tests with sample data."""
        print("\nRunning THOTH tests...")
        print("=" * 40)

        try:
            # Try to import and run pytest
            import pytest

            test_dir = Path(__file__).parent.parent / "tests"
            if test_dir.exists():
                return pytest.main([str(test_dir), "-v"])
            else:
                print("Test directory not found. Running basic checks...")
                return self._run_basic_checks()

        except ImportError:
            print("pytest not installed. Running basic checks...")
            return self._run_basic_checks()

    def _run_basic_checks(self) -> int:
        """Run basic sanity checks."""
        checks_passed = 0
        checks_total = 0

        # Check imports
        print("\nChecking imports...")
        checks_total += 1
        try:
            from . import (
                Config,
                LanguageMapper,
                NLLBEngine,
                ArgosEngine,
                CSVProcessor,
            )
            print("  [OK] All modules imported successfully")
            checks_passed += 1
        except ImportError as e:
            print(f"  [FAIL] Import error: {e}")

        # Check language mapper
        print("\nChecking language mapper...")
        checks_total += 1
        try:
            mapper = LanguageMapper()
            assert mapper.to_nllb("ru") == "rus_Cyrl"
            assert mapper.to_argos("rus_Cyrl") == "ru"
            assert mapper.get_name("rus_Cyrl") == "Russian"
            print("  [OK] Language mapping works correctly")
            checks_passed += 1
        except Exception as e:
            print(f"  [FAIL] Language mapper error: {e}")

        # Check config
        print("\nChecking configuration...")
        checks_total += 1
        try:
            config = Config()
            assert config.default_engine in ("nllb", "argos")
            assert config.performance.batch_size > 0
            print("  [OK] Configuration loads correctly")
            checks_passed += 1
        except Exception as e:
            print(f"  [FAIL] Configuration error: {e}")

        # Check engine factory
        print("\nChecking engine factory...")
        checks_total += 1
        try:
            engines = TranslationEngineFactory.get_available_engines()
            assert "nllb" in engines
            assert "argos" in engines
            print(f"  [OK] Available engines: {', '.join(engines)}")
            checks_passed += 1
        except Exception as e:
            print(f"  [FAIL] Engine factory error: {e}")

        # Summary
        print("\n" + "=" * 40)
        print(f"Tests: {checks_passed}/{checks_total} passed")

        return 0 if checks_passed == checks_total else 1


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    cli = CLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
