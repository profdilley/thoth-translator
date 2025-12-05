"""
THOTH - Translator for Hybrid Offline Text Handling

A production-ready offline translation tool for translating CSV/Excel columns
from non-English languages to English, running 100% locally with no internet
required after initial setup.

Features:
- Dual translation engine support (NLLB-200 and Argos Translate)
- Automatic language detection via fastText LID218
- Support for 50+ languages including Baltic, Balkan, and Asian languages
- CSV and Excel file processing
- Professional Tkinter GUI and command-line interface
- Progress tracking with cancellation support
- Configurable via YAML configuration file

Usage:
    # GUI mode
    python thoth.py --gui

    # CLI mode
    python thoth.py input.csv -o output.csv

    # Download models
    python -m translator.setup --download-models

Author: THOTH Development Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "THOTH Development Team"
__license__ = "MIT"

from .config import Config
from .languages import LanguageMapper
from .detector import LanguageDetector
from .engine_base import TranslationEngine
from .engine_nllb import NLLBEngine
from .engine_argos import ArgosEngine
from .processor import CSVProcessor
from .progress import ProgressTracker

__all__ = [
    "Config",
    "LanguageMapper",
    "LanguageDetector",
    "TranslationEngine",
    "NLLBEngine",
    "ArgosEngine",
    "CSVProcessor",
    "ProgressTracker",
    "__version__",
]
