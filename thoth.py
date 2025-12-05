#!/usr/bin/env python3
"""
THOTH - Translator for Hybrid Offline Text Handling

A production-ready offline translation tool for translating CSV/Excel
columns from non-English languages to English, running 100% locally
with no internet required after initial setup.

Usage:
    # GUI mode (default)
    python thoth.py --gui

    # CLI mode
    python thoth.py input.csv -o output.csv

    # Specify columns and language
    python thoth.py input.csv --columns "description,notes" --force-lang rus_Cyrl

    # Use Argos engine instead of NLLB
    python thoth.py input.csv --engine argos

    # Download models
    python -m translator.setup --download-models

    # Run tests
    python thoth.py --test

Author: THOTH Development Team
License: MIT
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path

# Add translator package to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        True if all dependencies are available
    """
    missing = []

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import fasttext
    except ImportError:
        missing.append("fasttext")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    return True


def main() -> int:
    """
    Main entry point for THOTH.

    Parses command-line arguments and launches either GUI or CLI mode.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        prog="thoth",
        description="THOTH - Translator for Hybrid Offline Text Handling",
        epilog="""
Examples:
  thoth --gui                          Launch GUI mode
  thoth input.csv                      Translate input.csv (CLI mode)
  thoth input.csv -o output.csv        Specify output file
  thoth input.csv --engine argos       Use Argos engine
  thoth input.csv --force-lang rus_Cyrl Force Russian as source language
  thoth --test                         Run tests

For more information, see README.md
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch graphical user interface",
    )

    # Input file (positional, optional)
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
        help="Translation engine (default: nllb)",
    )

    # Configuration
    parser.add_argument(
        "--config",
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

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Handle --list-languages
    if args.list_languages:
        from translator.languages import LanguageMapper
        mapper = LanguageMapper()

        print("\nSupported Languages:")
        print("=" * 70)
        print(f"{'Language':<25} {'NLLB Code':<15} {'Argos':<8} {'Family'}")
        print("-" * 70)

        for lang in sorted(mapper.LANGUAGES, key=lambda x: x.name):
            argos = lang.argos_code if lang.argos_supported else "-"
            print(f"{lang.name:<25} {lang.nllb_code:<15} {argos:<8} {lang.family}")

        print("-" * 70)
        print(f"Total: {len(mapper.LANGUAGES)} languages")
        return 0

    # Handle --test
    if args.test:
        from translator.cli import CLI
        cli = CLI()
        return cli._run_tests()

    # Handle --gui or no arguments
    if args.gui or (not args.input_file and not args.test):
        try:
            from translator.gui import run_gui
            from translator.config import Config

            config = Config.load(args.config)
            run_gui(config)
            return 0

        except ImportError as e:
            print(f"GUI dependencies not available: {e}")
            print("Try running in CLI mode: python thoth.py input.csv")
            return 1

        except Exception as e:
            print(f"Error launching GUI: {e}")
            return 1

    # CLI mode
    if args.input_file:
        from translator.cli import CLI

        cli = CLI()
        # Build argument list for CLI
        cli_args = [args.input_file]

        if args.output_file:
            cli_args.extend(["-o", args.output_file])
        if args.columns:
            cli_args.extend(["--columns", args.columns])
        if args.force_lang:
            cli_args.extend(["--force-lang", args.force_lang])
        if args.engine:
            cli_args.extend(["--engine", args.engine])
        if args.config:
            cli_args.extend(["--config", args.config])
        if args.quiet:
            cli_args.append("--quiet")
        if args.verbose:
            cli_args.append("--verbose")
        if args.analyze:
            cli_args.append("--analyze")

        return cli.run(cli_args)

    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
