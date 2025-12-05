"""
Model downloader for THOTH.

This module handles downloading all required models for offline translation:
- NLLB-200-distilled-600M from HuggingFace (~2.5 GB)
- fastText LID218 language detection model (~130 MB)
- Argos Translate language packs (~1.5 GB total)

Usage:
    python -m translator.setup --download-models
"""

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model URLs and metadata
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
LID_MODEL_SIZE = 131_174_421  # Approximate size in bytes

# NLLB model is downloaded via HuggingFace transformers
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Priority Argos language packs (source -> target)
ARGOS_PRIORITY_PACKS = [
    ("ru", "en"),  # Russian
    ("en", "ru"),
    ("uk", "en"),  # Ukrainian
    ("en", "uk"),
    ("de", "en"),  # German
    ("en", "de"),
    ("fr", "en"),  # French
    ("en", "fr"),
    ("es", "en"),  # Spanish
    ("en", "es"),
    ("zh", "en"),  # Chinese
    ("en", "zh"),
    ("ja", "en"),  # Japanese
    ("en", "ja"),
    ("ko", "en"),  # Korean
    ("en", "ko"),
    ("ar", "en"),  # Arabic
    ("en", "ar"),
    ("pl", "en"),  # Polish
    ("en", "pl"),
    ("sv", "en"),  # Swedish
    ("en", "sv"),
    ("fi", "en"),  # Finnish
    ("en", "fi"),
    ("tr", "en"),  # Turkish
    ("en", "tr"),
    ("el", "en"),  # Greek
    ("en", "el"),
    ("nl", "en"),  # Dutch
    ("en", "nl"),
    ("it", "en"),  # Italian
    ("en", "it"),
    ("pt", "en"),  # Portuguese
    ("en", "pt"),
    ("cs", "en"),  # Czech
    ("en", "cs"),
    ("hu", "en"),  # Hungarian
    ("en", "hu"),
    ("he", "en"),  # Hebrew
    ("en", "he"),
    ("fa", "en"),  # Persian
    ("en", "fa"),
    ("hi", "en"),  # Hindi
    ("en", "hi"),
    ("vi", "en"),  # Vietnamese
    ("en", "vi"),
]


class DownloadProgress:
    """Progress tracker for downloads."""

    def __init__(self, total_size: int, description: str = "") -> None:
        self.total_size = total_size
        self.description = description
        self.downloaded = 0
        self.last_percent = -1

    def update(self, chunk_size: int) -> None:
        """Update progress with new chunk."""
        self.downloaded += chunk_size
        percent = int((self.downloaded / self.total_size) * 100) if self.total_size > 0 else 0

        if percent != self.last_percent:
            self.last_percent = percent
            bar_width = 40
            filled = int(bar_width * percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            mb_downloaded = self.downloaded / (1024 * 1024)
            mb_total = self.total_size / (1024 * 1024)

            print(
                f"\r{self.description}: {bar} {percent}% "
                f"({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True,
            )

    def complete(self) -> None:
        """Mark download as complete."""
        print()  # New line after progress bar


def download_file(
    url: str,
    dest_path: Path,
    expected_size: Optional[int] = None,
    description: str = "",
) -> bool:
    """
    Download a file with progress display.

    Args:
        url: URL to download from
        dest_path: Destination file path
        expected_size: Expected file size for progress (optional)
        description: Description for progress display

    Returns:
        True if download successful
    """
    try:
        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", expected_size or 0))

            progress = DownloadProgress(total_size, description)

            # Download in chunks
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))

            progress.complete()

        logger.info(f"Downloaded: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_lid_model(models_dir: Path) -> bool:
    """
    Download the fastText LID218 model.

    Args:
        models_dir: Directory to save models

    Returns:
        True if successful
    """
    lid_path = models_dir / "lid218e.bin"

    if lid_path.exists():
        logger.info(f"LID model already exists: {lid_path}")
        return True

    print("\nDownloading fastText LID218 model (~130 MB)...")
    return download_file(
        LID_MODEL_URL,
        lid_path,
        LID_MODEL_SIZE,
        "LID218",
    )


def download_nllb_model(models_dir: Path) -> bool:
    """
    Download NLLB-200 model using HuggingFace transformers.

    Args:
        models_dir: Directory to save models

    Returns:
        True if successful
    """
    nllb_path = models_dir / "nllb-200-distilled-600M"

    if nllb_path.exists() and any(nllb_path.iterdir()):
        logger.info(f"NLLB model already exists: {nllb_path}")
        return True

    print("\nDownloading NLLB-200-distilled-600M model (~2.5 GB)...")
    print("This may take several minutes depending on your connection.")

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        tokenizer.save_pretrained(str(nllb_path))

        print("Downloading model weights...")
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        model.save_pretrained(str(nllb_path))

        logger.info(f"NLLB model saved to: {nllb_path}")
        return True

    except ImportError:
        logger.error(
            "transformers library not installed. "
            "Run: pip install transformers torch sentencepiece"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to download NLLB model: {e}")
        return False


def download_argos_packs(models_dir: Path, packs: Optional[list[tuple[str, str]]] = None) -> bool:
    """
    Download Argos Translate language packs.

    Args:
        models_dir: Directory for Argos models
        packs: List of (from_code, to_code) tuples (uses priority list if None)

    Returns:
        True if successful
    """
    packs = packs or ARGOS_PRIORITY_PACKS

    print(f"\nDownloading Argos language packs ({len(packs)} packages)...")

    try:
        import argostranslate.package
        import argostranslate.translate

        # Update package index
        print("Updating Argos package index...")
        argostranslate.package.update_package_index()

        # Get available packages
        available = argostranslate.package.get_available_packages()
        available_map = {
            (pkg.from_code, pkg.to_code): pkg
            for pkg in available
        }

        # Download each pack
        downloaded = 0
        failed = 0

        for from_code, to_code in packs:
            pack_key = (from_code, to_code)

            if pack_key not in available_map:
                logger.warning(f"Package not available: {from_code} -> {to_code}")
                continue

            package = available_map[pack_key]

            # Check if already installed
            installed = argostranslate.translate.get_installed_languages()
            installed_pairs = set()
            for lang in installed:
                for translation in lang.translations:
                    installed_pairs.add((lang.code, translation.to_lang.code))

            if pack_key in installed_pairs:
                print(f"  Already installed: {from_code} -> {to_code}")
                downloaded += 1
                continue

            try:
                print(f"  Downloading: {from_code} -> {to_code}...")
                download_path = package.download()
                argostranslate.package.install_from_path(download_path)
                downloaded += 1
                print(f"    Installed: {from_code} -> {to_code}")
            except Exception as e:
                logger.warning(f"Failed to install {from_code} -> {to_code}: {e}")
                failed += 1

        print(f"\nArgos packs: {downloaded} installed, {failed} failed")
        return failed == 0

    except ImportError:
        logger.error(
            "argostranslate library not installed. "
            "Run: pip install argostranslate"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to download Argos packs: {e}")
        return False


def setup_models(
    models_dir: Optional[Path] = None,
    download_nllb: bool = True,
    download_lid: bool = True,
    download_argos: bool = True,
) -> bool:
    """
    Download all required models.

    Args:
        models_dir: Directory to save models
        download_nllb: Whether to download NLLB model
        download_lid: Whether to download LID model
        download_argos: Whether to download Argos packs

    Returns:
        True if all downloads successful
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  THOTH Model Setup")
    print("=" * 60)
    print(f"\nModels directory: {models_dir}")

    success = True

    if download_lid:
        if not download_lid_model(models_dir):
            success = False

    if download_nllb:
        if not download_nllb_model(models_dir):
            success = False

    if download_argos:
        if not download_argos_packs(models_dir):
            # Argos is optional, don't fail completely
            logger.warning("Some Argos packs failed to download")

    print("\n" + "=" * 60)
    if success:
        print("  Model setup complete!")
        print("  You can now run THOTH offline.")
    else:
        print("  Some downloads failed. Check errors above.")
    print("=" * 60)

    return success


def check_models(models_dir: Optional[Path] = None) -> dict[str, bool]:
    """
    Check which models are installed.

    Args:
        models_dir: Directory to check

    Returns:
        Dictionary mapping model name to existence status
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"

    status = {
        "lid": (models_dir / "lid218e.bin").exists(),
        "nllb": (models_dir / "nllb-200-distilled-600M").exists(),
    }

    # Check Argos
    try:
        import argostranslate.translate
        languages = argostranslate.translate.get_installed_languages()
        status["argos"] = len(languages) > 0
        status["argos_languages"] = len(languages)
    except ImportError:
        status["argos"] = False
        status["argos_languages"] = 0

    return status


def print_status(models_dir: Optional[Path] = None) -> None:
    """Print model installation status."""
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"

    status = check_models(models_dir)

    print("\nModel Status:")
    print("-" * 40)
    print(f"  LID218 (language detection): {'✓' if status['lid'] else '✗'}")
    print(f"  NLLB-200 (translation):      {'✓' if status['nllb'] else '✗'}")
    print(f"  Argos (translation):         {'✓' if status['argos'] else '✗'}", end="")
    if status.get('argos_languages', 0) > 0:
        print(f" ({status['argos_languages']} languages)")
    else:
        print()
    print("-" * 40)


def main() -> int:
    """Main entry point for setup module."""
    parser = argparse.ArgumentParser(
        description="THOTH Model Setup - Download required translation models",
    )

    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all required models",
    )

    parser.add_argument(
        "--download-nllb",
        action="store_true",
        help="Download only NLLB model",
    )

    parser.add_argument(
        "--download-lid",
        action="store_true",
        help="Download only LID model",
    )

    parser.add_argument(
        "--download-argos",
        action="store_true",
        help="Download only Argos packs",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Custom models directory",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check model installation status",
    )

    args = parser.parse_args()

    if args.status:
        print_status(args.models_dir)
        return 0

    if args.download_models:
        success = setup_models(args.models_dir)
        return 0 if success else 1

    if args.download_nllb:
        models_dir = args.models_dir or Path(__file__).parent.parent / "models"
        success = download_nllb_model(models_dir)
        return 0 if success else 1

    if args.download_lid:
        models_dir = args.models_dir or Path(__file__).parent.parent / "models"
        success = download_lid_model(models_dir)
        return 0 if success else 1

    if args.download_argos:
        models_dir = args.models_dir or Path(__file__).parent.parent / "models"
        success = download_argos_packs(models_dir)
        return 0 if success else 1

    # No action specified
    parser.print_help()
    print("\nCurrent model status:")
    print_status(args.models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
