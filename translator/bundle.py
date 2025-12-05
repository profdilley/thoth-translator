"""
Client bundling utility for THOTH.

This module creates self-contained distribution packages that include
all source code, models, and configuration needed to run THOTH offline
on client machines.

Usage:
    python -m translator.bundle --output translation_bundle.zip
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Files and directories to include in the bundle
INCLUDE_PATTERNS = [
    "thoth.py",
    "translator/*.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
    "tests/*.py",
    "tests/*.csv",
]

# Directories that contain large files
MODEL_DIRS = [
    "models/nllb-200-distilled-600M",
    "models/lid218e.bin",
]

# Files to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".gitignore",
    ".env",
    "*.egg-info",
    "dist",
    "build",
    ".pytest_cache",
]


class BundleCreator:
    """
    Creates self-contained THOTH distribution bundles.

    Packages all necessary files including:
    - Python source code
    - Configuration files
    - Documentation
    - Translation models (optional)
    - Test files
    """

    def __init__(self, source_dir: Optional[Path] = None) -> None:
        """
        Initialize bundle creator.

        Args:
            source_dir: Root directory of THOTH installation
        """
        self.source_dir = source_dir or Path(__file__).parent.parent
        self.manifest: list[tuple[str, str]] = []  # (source, archive_name)

    def create_bundle(
        self,
        output_path: str,
        include_models: bool = True,
        include_argos: bool = False,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> str:
        """
        Create a distribution bundle.

        Args:
            output_path: Path for the output zip file
            include_models: Whether to include NLLB and LID models
            include_argos: Whether to include Argos language packs
            compression: ZIP compression level

        Returns:
            Path to the created bundle
        """
        output_path = Path(output_path)

        print("=" * 60)
        print("  THOTH Bundle Creator")
        print("=" * 60)
        print(f"\nSource directory: {self.source_dir}")
        print(f"Output file: {output_path}")
        print(f"Include models: {include_models}")
        print()

        # Collect files
        self._collect_source_files()

        if include_models:
            self._collect_model_files()

        if include_argos:
            self._collect_argos_models()

        # Create the bundle
        print(f"\nCreating bundle with {len(self.manifest)} files...")

        with zipfile.ZipFile(output_path, "w", compression) as zf:
            for source_path, archive_name in self.manifest:
                print(f"  Adding: {archive_name}")
                zf.write(source_path, archive_name)

        # Calculate bundle size
        bundle_size = output_path.stat().st_size
        size_mb = bundle_size / (1024 * 1024)

        print("\n" + "=" * 60)
        print(f"  Bundle created successfully!")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Location: {output_path}")
        print("=" * 60)

        # Print installation instructions
        self._print_instructions()

        return str(output_path)

    def _collect_source_files(self) -> None:
        """Collect Python source and configuration files."""
        print("Collecting source files...")

        # Main entry point
        thoth_py = self.source_dir / "thoth.py"
        if thoth_py.exists():
            self.manifest.append((str(thoth_py), "thoth-translator/thoth.py"))

        # Translator package
        translator_dir = self.source_dir / "translator"
        if translator_dir.exists():
            for py_file in translator_dir.glob("*.py"):
                archive_name = f"thoth-translator/translator/{py_file.name}"
                self.manifest.append((str(py_file), archive_name))

        # Configuration
        config_yaml = self.source_dir / "config.yaml"
        if config_yaml.exists():
            self.manifest.append((str(config_yaml), "thoth-translator/config.yaml"))

        # Requirements
        requirements = self.source_dir / "requirements.txt"
        if requirements.exists():
            self.manifest.append((str(requirements), "thoth-translator/requirements.txt"))

        # README
        readme = self.source_dir / "README.md"
        if readme.exists():
            self.manifest.append((str(readme), "thoth-translator/README.md"))

        # Tests
        tests_dir = self.source_dir / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("*"):
                if test_file.suffix in (".py", ".csv"):
                    archive_name = f"thoth-translator/tests/{test_file.name}"
                    self.manifest.append((str(test_file), archive_name))

        print(f"  Found {len(self.manifest)} source files")

    def _collect_model_files(self) -> None:
        """Collect translation and detection model files."""
        models_dir = self.source_dir / "models"

        if not models_dir.exists():
            logger.warning("Models directory not found. Run setup first.")
            return

        print("Collecting model files...")
        model_count = 0

        # LID model
        lid_model = models_dir / "lid218e.bin"
        if lid_model.exists():
            archive_name = "thoth-translator/models/lid218e.bin"
            self.manifest.append((str(lid_model), archive_name))
            model_count += 1
            print(f"  Found LID model ({lid_model.stat().st_size / 1024 / 1024:.1f} MB)")

        # NLLB model
        nllb_dir = models_dir / "nllb-200-distilled-600M"
        if nllb_dir.exists():
            for model_file in nllb_dir.rglob("*"):
                if model_file.is_file():
                    rel_path = model_file.relative_to(models_dir)
                    archive_name = f"thoth-translator/models/{rel_path}"
                    self.manifest.append((str(model_file), archive_name))
                    model_count += 1

            # Calculate NLLB size
            nllb_size = sum(
                f.stat().st_size for f in nllb_dir.rglob("*") if f.is_file()
            )
            print(f"  Found NLLB model ({nllb_size / 1024 / 1024 / 1024:.2f} GB)")

        print(f"  Total: {model_count} model files")

    def _collect_argos_models(self) -> None:
        """Collect Argos Translate models."""
        print("Collecting Argos models...")

        try:
            import argostranslate.package

            # Get Argos data directory
            data_dir = Path(argostranslate.package.get_installed_packages()[0].package_path).parent

            if data_dir.exists():
                argos_count = 0
                for pkg_file in data_dir.rglob("*"):
                    if pkg_file.is_file():
                        rel_path = pkg_file.relative_to(data_dir)
                        archive_name = f"thoth-translator/models/argos/{rel_path}"
                        self.manifest.append((str(pkg_file), archive_name))
                        argos_count += 1

                print(f"  Found {argos_count} Argos model files")

        except Exception as e:
            logger.warning(f"Could not collect Argos models: {e}")

    def _print_instructions(self) -> None:
        """Print installation instructions for the bundle."""
        print("\n" + "-" * 60)
        print("Installation Instructions:")
        print("-" * 60)
        print("""
1. Extract the bundle:
   unzip translation_bundle.zip

2. Create a virtual environment:
   cd thoth-translator
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run THOTH:
   # GUI mode
   python thoth.py --gui

   # CLI mode
   python thoth.py input.csv -o output.csv

No internet connection required after setup!
""")


def create_minimal_bundle(output_path: str) -> str:
    """
    Create a minimal bundle without models.

    The client will need to download models separately using
    python -m translator.setup --download-models

    Args:
        output_path: Path for the output zip file

    Returns:
        Path to the created bundle
    """
    creator = BundleCreator()
    return creator.create_bundle(
        output_path,
        include_models=False,
        include_argos=False,
    )


def create_full_bundle(output_path: str) -> str:
    """
    Create a full bundle with NLLB and LID models.

    Args:
        output_path: Path for the output zip file

    Returns:
        Path to the created bundle
    """
    creator = BundleCreator()
    return creator.create_bundle(
        output_path,
        include_models=True,
        include_argos=False,
    )


def create_complete_bundle(output_path: str) -> str:
    """
    Create a complete bundle with all models including Argos.

    Args:
        output_path: Path for the output zip file

    Returns:
        Path to the created bundle
    """
    creator = BundleCreator()
    return creator.create_bundle(
        output_path,
        include_models=True,
        include_argos=True,
    )


def verify_bundle(bundle_path: str) -> bool:
    """
    Verify the integrity of a bundle.

    Args:
        bundle_path: Path to the bundle file

    Returns:
        True if bundle is valid
    """
    print(f"Verifying bundle: {bundle_path}")

    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            # Check for required files
            required = [
                "thoth-translator/thoth.py",
                "thoth-translator/translator/__init__.py",
                "thoth-translator/requirements.txt",
            ]

            names = zf.namelist()

            for req in required:
                if req not in names:
                    print(f"  Missing: {req}")
                    return False

            # Test archive integrity
            bad_file = zf.testzip()
            if bad_file:
                print(f"  Corrupted: {bad_file}")
                return False

            print(f"  Files: {len(names)}")
            print(f"  Status: Valid")
            return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main() -> int:
    """Main entry point for bundle module."""
    parser = argparse.ArgumentParser(
        description="THOTH Bundle Creator - Create distribution packages",
    )

    parser.add_argument(
        "--output", "-o",
        default="translation_bundle.zip",
        help="Output bundle path (default: translation_bundle.zip)",
    )

    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Create minimal bundle without models",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Create full bundle with NLLB and LID models",
    )

    parser.add_argument(
        "--complete",
        action="store_true",
        help="Create complete bundle with all models",
    )

    parser.add_argument(
        "--verify",
        metavar="BUNDLE",
        help="Verify an existing bundle",
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Source directory (default: auto-detect)",
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_bundle(args.verify)
        return 0 if success else 1

    if args.minimal:
        create_minimal_bundle(args.output)
        return 0

    if args.complete:
        create_complete_bundle(args.output)
        return 0

    # Default: full bundle
    create_full_bundle(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
