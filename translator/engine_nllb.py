"""
NLLB-200 translation engine implementation.

This module provides translation using Meta's No Language Left Behind (NLLB)
model, which supports 200 languages including many low-resource languages.

Model: facebook/nllb-200-distilled-600M (~2.5 GB)
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


class NLLBEngine(TranslationEngine):
    """
    Translation engine using Meta's NLLB-200 model.

    NLLB-200 (No Language Left Behind) is a massively multilingual
    translation model supporting 200 languages with state-of-the-art
    quality for many language pairs.

    Features:
        - 200 language support including low-resource languages
        - High-quality translations for most language pairs
        - Efficient batched inference
        - Runs on CPU or GPU

    Model variants:
        - nllb-200-distilled-600M: Recommended, good balance of quality/speed (~2.5 GB)
        - nllb-200-distilled-1.3B: Higher quality, slower (~5 GB)
        - nllb-200-3.3B: Highest quality, requires significant resources (~13 GB)

    Example:
        engine = NLLBEngine()
        engine.load_model()

        result = engine.translate("Привет мир", "rus_Cyrl", "eng_Latn")
        print(result.translated_text)  # "Hello world"
    """

    # Default model identifier on HuggingFace
    DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"

    # Maximum sequence length for the model
    MAX_LENGTH = 512

    # All 200 NLLB language codes
    SUPPORTED_LANGUAGES = [
        "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn",
        "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab",
        "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn",
        "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
        "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
        "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn",
        "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
        "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
        "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn",
        "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn",
        "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn",
        "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
        "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn",
        "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
        "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", "kaz_Cyrl", "kbp_Latn",
        "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
        "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn",
        "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn",
        "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "mag_Deva", "mai_Deva",
        "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn",
        "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr",
        "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn",
        "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
        "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
        "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng",
        "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn",
        "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "als_Latn",
        "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
        "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn",
        "tha_Thai", "tir_Ethi", "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn",
        "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng",
        "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
        "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn",
        "yue_Hant", "zho_Hans", "zho_Hant", "zul_Latn",
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize NLLB engine.

        Args:
            model_path: Path to local model or HuggingFace model ID
                       (defaults to facebook/nllb-200-distilled-600M)
        """
        super().__init__(model_path)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._language_mapper = LanguageMapper()

        # Use default model if path not specified
        if self._model_path is None:
            self._model_path = self.DEFAULT_MODEL_ID

    def get_engine_name(self) -> str:
        """Get human-readable engine name."""
        return "NLLB-200"

    def get_engine_id(self) -> str:
        """Get engine identifier."""
        return "nllb"

    def is_available(self) -> bool:
        """Check if NLLB is available."""
        try:
            # Check if transformers is installed
            import transformers
            return True
        except ImportError:
            return False

    def get_supported_languages(self) -> list[str]:
        """Get list of supported NLLB language codes."""
        return self.SUPPORTED_LANGUAGES.copy()

    def load_model(self) -> None:
        """
        Load the NLLB model into memory.

        This downloads the model from HuggingFace on first use if not
        already cached or if a local path is not provided.
        """
        if self._model_loaded:
            logger.info("NLLB model already loaded")
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading NLLB model from {self._model_path}...")

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using CUDA GPU for inference")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
                logger.info("Using Apple MPS for inference")
            else:
                self._device = "cpu"
                logger.info("Using CPU for inference")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                src_lang="eng_Latn",
            )

            # Load model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_path,
            )

            # Move to device
            self._model = self._model.to(self._device)
            self._model.eval()

            self._model_loaded = True
            logger.info("NLLB model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "Required packages not installed. "
                "Run: pip install transformers torch sentencepiece"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise RuntimeError(f"Failed to load NLLB model: {e}") from e

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._model_loaded = False

        # Force garbage collection
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("NLLB model unloaded")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate a single text using NLLB.

        Args:
            text: Text to translate
            source_lang: Source language in NLLB format (e.g., 'rus_Cyrl')
            target_lang: Target language in NLLB format (defaults to 'eng_Latn')

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
            # Validate languages
            self._validate_language(source_lang, "source_lang")
            self._validate_language(target_lang, "target_lang")

            import torch

            # Set source language for tokenizer
            self._tokenizer.src_lang = source_lang

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=self.MAX_LENGTH,
                truncation=True,
            ).to(self._device)

            # Get target language token ID
            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(target_lang)

            # Generate translation
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=self.MAX_LENGTH,
                    num_beams=5,
                    early_stopping=True,
                )

            # Decode
            translated = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

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
            logger.error(f"Translation error: {e}")
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
        Translate multiple texts efficiently using batching.

        For optimal performance, texts with the same source language
        are grouped and translated together.

        Args:
            texts: List of texts to translate
            source_langs: List of source language codes (one per text)
            target_lang: Target language (same for all texts)
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

        # Validate inputs
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

        try:
            import torch

            # Group texts by source language for efficient batching
            lang_groups: dict[str, list[tuple[int, str]]] = {}
            for idx, (text, src_lang) in enumerate(zip(texts, source_langs)):
                if src_lang not in lang_groups:
                    lang_groups[src_lang] = []
                lang_groups[src_lang].append((idx, text))

            # Pre-allocate results list
            results = [None] * len(texts)  # type: ignore

            # Process each language group
            for src_lang, group_items in lang_groups.items():
                if progress and progress.is_cancelled():
                    break

                # Set tokenizer source language
                self._tokenizer.src_lang = src_lang

                # Get target token ID
                try:
                    forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(target_lang)
                except Exception:
                    # Language not supported
                    for idx, text in group_items:
                        results[idx] = TranslationResult(
                            source_text=text,
                            translated_text=None,
                            source_language=src_lang,
                            target_language=target_lang,
                            success=False,
                            error=f"Unsupported source language: {src_lang}",
                        )
                        failure_count += 1
                    continue

                # Process in sub-batches for memory efficiency
                batch_size = 8  # Smaller batch for memory efficiency
                for batch_start in range(0, len(group_items), batch_size):
                    if progress and progress.is_cancelled():
                        break

                    batch_items = group_items[batch_start:batch_start + batch_size]
                    batch_indices = [item[0] for item in batch_items]
                    batch_texts = [item[1] for item in batch_items]

                    # Handle empty texts
                    non_empty_indices = []
                    non_empty_texts = []
                    for i, (idx, text) in enumerate(batch_items):
                        if text and text.strip():
                            non_empty_indices.append(i)
                            non_empty_texts.append(text)
                        else:
                            # Empty text - just copy it
                            results[idx] = TranslationResult(
                                source_text=text,
                                translated_text=text,
                                source_language=src_lang,
                                target_language=target_lang,
                                success=True,
                            )
                            success_count += 1

                    if non_empty_texts:
                        try:
                            # Tokenize batch
                            inputs = self._tokenizer(
                                non_empty_texts,
                                return_tensors="pt",
                                padding=True,
                                max_length=self.MAX_LENGTH,
                                truncation=True,
                            ).to(self._device)

                            # Generate translations
                            with torch.no_grad():
                                outputs = self._model.generate(
                                    **inputs,
                                    forced_bos_token_id=forced_bos_token_id,
                                    max_length=self.MAX_LENGTH,
                                    num_beams=5,
                                    early_stopping=True,
                                )

                            # Decode results
                            translations = self._tokenizer.batch_decode(
                                outputs,
                                skip_special_tokens=True,
                            )

                            # Store results
                            for i, trans in zip(non_empty_indices, translations):
                                idx = batch_indices[i]
                                results[idx] = TranslationResult(
                                    source_text=batch_texts[i],
                                    translated_text=trans,
                                    source_language=src_lang,
                                    target_language=target_lang,
                                    success=True,
                                )
                                success_count += 1

                        except Exception as e:
                            logger.error(f"Batch translation error: {e}")
                            # Fall back to individual translation
                            for i in non_empty_indices:
                                idx = batch_indices[i]
                                text = batch_texts[i]
                                result = self.translate(text, src_lang, target_lang)
                                results[idx] = result
                                if result.success:
                                    success_count += 1
                                else:
                                    failure_count += 1

                    # Update progress
                    if progress:
                        progress.update(
                            len(batch_items),
                            f"Translating from {src_lang}..."
                        )

            # Handle any cancelled items
            for idx, result in enumerate(results):
                if result is None:
                    results[idx] = TranslationResult(
                        source_text=texts[idx],
                        translated_text=None,
                        source_language=source_langs[idx],
                        target_language=target_lang,
                        success=False,
                        error="Translation cancelled",
                    )
                    failure_count += 1

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Return partial results with failures
            for idx in range(len(texts)):
                if results[idx] is None:
                    results[idx] = TranslationResult(
                        source_text=texts[idx],
                        translated_text=None,
                        source_language=source_langs[idx],
                        target_language=target_lang,
                        success=False,
                        error=str(e),
                    )
                    failure_count += 1

        processing_time = time.time() - start_time

        return BatchTranslationResult(
            results=results,  # type: ignore
            success_count=success_count,
            failure_count=failure_count,
            processing_time=processing_time,
        )

    def get_model_info(self) -> dict:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            "model_id": self._model_path,
            "device": self._device,
            "max_length": self.MAX_LENGTH,
            "num_languages": len(self.SUPPORTED_LANGUAGES),
        })
        return info


# Register with factory
TranslationEngineFactory.register("nllb", NLLBEngine)
