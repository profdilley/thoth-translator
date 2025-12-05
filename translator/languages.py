"""
Language code mappings for THOTH translation tool.

This module provides comprehensive language code mappings between:
- NLLB-200 format (e.g., 'rus_Cyrl', 'ukr_Cyrl')
- Argos Translate ISO 639-1 format (e.g., 'ru', 'uk')
- fastText LID218 BCP-47 format (e.g., 'rus', 'ukr')
- Human-readable language names

Supports 50+ languages including Slavic, Baltic, Nordic, Western European,
East Asian, and Middle Eastern language families.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LanguageInfo:
    """Complete language information including all code formats."""

    # Human-readable name
    name: str

    # NLLB-200 format code (e.g., 'rus_Cyrl')
    nllb_code: str

    # ISO 639-1 two-letter code for Argos (e.g., 'ru')
    argos_code: str

    # fastText LID code (typically ISO 639-3 or BCP-47)
    lid_code: str

    # Script type for display purposes
    script: str

    # Whether Argos supports this language (may need manual update)
    argos_supported: bool = True

    # Language family for grouping
    family: str = "Other"


class LanguageMapper:
    """
    Provides bidirectional mapping between language code formats.

    Supports conversion between NLLB, Argos, and fastText language codes,
    as well as lookup by human-readable language name.

    Example:
        mapper = LanguageMapper()
        nllb_code = mapper.to_nllb("ru")  # Returns "rus_Cyrl"
        argos_code = mapper.to_argos("rus_Cyrl")  # Returns "ru"
        name = mapper.get_name("rus_Cyrl")  # Returns "Russian"
    """

    # Complete language database
    # Organized by language family for clarity
    LANGUAGES: list[LanguageInfo] = [
        # ============================================
        # SLAVIC / EASTERN EUROPEAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Russian",
            nllb_code="rus_Cyrl",
            argos_code="ru",
            lid_code="rus",
            script="Cyrillic",
            family="Slavic"
        ),
        LanguageInfo(
            name="Ukrainian",
            nllb_code="ukr_Cyrl",
            argos_code="uk",
            lid_code="ukr",
            script="Cyrillic",
            family="Slavic"
        ),
        LanguageInfo(
            name="Belarusian",
            nllb_code="bel_Cyrl",
            argos_code="be",
            lid_code="bel",
            script="Cyrillic",
            argos_supported=False,
            family="Slavic"
        ),
        LanguageInfo(
            name="Polish",
            nllb_code="pol_Latn",
            argos_code="pl",
            lid_code="pol",
            script="Latin",
            family="Slavic"
        ),
        LanguageInfo(
            name="Czech",
            nllb_code="ces_Latn",
            argos_code="cs",
            lid_code="ces",
            script="Latin",
            family="Slavic"
        ),
        LanguageInfo(
            name="Slovak",
            nllb_code="slk_Latn",
            argos_code="sk",
            lid_code="slk",
            script="Latin",
            family="Slavic"
        ),
        LanguageInfo(
            name="Bulgarian",
            nllb_code="bul_Cyrl",
            argos_code="bg",
            lid_code="bul",
            script="Cyrillic",
            family="Slavic"
        ),
        LanguageInfo(
            name="Macedonian",
            nllb_code="mkd_Cyrl",
            argos_code="mk",
            lid_code="mkd",
            script="Cyrillic",
            argos_supported=False,
            family="Slavic"
        ),
        LanguageInfo(
            name="Serbian",
            nllb_code="srp_Cyrl",
            argos_code="sr",
            lid_code="srp",
            script="Cyrillic",
            family="Slavic"
        ),
        LanguageInfo(
            name="Croatian",
            nllb_code="hrv_Latn",
            argos_code="hr",
            lid_code="hrv",
            script="Latin",
            family="Slavic"
        ),
        LanguageInfo(
            name="Slovenian",
            nllb_code="slv_Latn",
            argos_code="sl",
            lid_code="slv",
            script="Latin",
            family="Slavic"
        ),
        LanguageInfo(
            name="Bosnian",
            nllb_code="bos_Latn",
            argos_code="bs",
            lid_code="bos",
            script="Latin",
            argos_supported=False,
            family="Slavic"
        ),

        # ============================================
        # BALTIC LANGUAGES
        # ============================================
        LanguageInfo(
            name="Lithuanian",
            nllb_code="lit_Latn",
            argos_code="lt",
            lid_code="lit",
            script="Latin",
            family="Baltic"
        ),
        LanguageInfo(
            name="Latvian",
            nllb_code="lvs_Latn",
            argos_code="lv",
            lid_code="lvs",
            script="Latin",
            family="Baltic"
        ),
        LanguageInfo(
            name="Estonian",
            nllb_code="est_Latn",
            argos_code="et",
            lid_code="est",
            script="Latin",
            family="Finno-Ugric"
        ),

        # ============================================
        # NORDIC / SCANDINAVIAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Swedish",
            nllb_code="swe_Latn",
            argos_code="sv",
            lid_code="swe",
            script="Latin",
            family="Germanic"
        ),
        LanguageInfo(
            name="Norwegian Bokmål",
            nllb_code="nob_Latn",
            argos_code="nb",
            lid_code="nob",
            script="Latin",
            family="Germanic"
        ),
        LanguageInfo(
            name="Norwegian Nynorsk",
            nllb_code="nno_Latn",
            argos_code="nn",
            lid_code="nno",
            script="Latin",
            argos_supported=False,
            family="Germanic"
        ),
        LanguageInfo(
            name="Danish",
            nllb_code="dan_Latn",
            argos_code="da",
            lid_code="dan",
            script="Latin",
            family="Germanic"
        ),
        LanguageInfo(
            name="Finnish",
            nllb_code="fin_Latn",
            argos_code="fi",
            lid_code="fin",
            script="Latin",
            family="Finno-Ugric"
        ),
        LanguageInfo(
            name="Icelandic",
            nllb_code="isl_Latn",
            argos_code="is",
            lid_code="isl",
            script="Latin",
            family="Germanic"
        ),

        # ============================================
        # WESTERN EUROPEAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="German",
            nllb_code="deu_Latn",
            argos_code="de",
            lid_code="deu",
            script="Latin",
            family="Germanic"
        ),
        LanguageInfo(
            name="French",
            nllb_code="fra_Latn",
            argos_code="fr",
            lid_code="fra",
            script="Latin",
            family="Romance"
        ),
        LanguageInfo(
            name="Spanish",
            nllb_code="spa_Latn",
            argos_code="es",
            lid_code="spa",
            script="Latin",
            family="Romance"
        ),
        LanguageInfo(
            name="Italian",
            nllb_code="ita_Latn",
            argos_code="it",
            lid_code="ita",
            script="Latin",
            family="Romance"
        ),
        LanguageInfo(
            name="Portuguese",
            nllb_code="por_Latn",
            argos_code="pt",
            lid_code="por",
            script="Latin",
            family="Romance"
        ),
        LanguageInfo(
            name="Dutch",
            nllb_code="nld_Latn",
            argos_code="nl",
            lid_code="nld",
            script="Latin",
            family="Germanic"
        ),
        LanguageInfo(
            name="Romanian",
            nllb_code="ron_Latn",
            argos_code="ro",
            lid_code="ron",
            script="Latin",
            family="Romance"
        ),
        LanguageInfo(
            name="Hungarian",
            nllb_code="hun_Latn",
            argos_code="hu",
            lid_code="hun",
            script="Latin",
            family="Finno-Ugric"
        ),
        LanguageInfo(
            name="Greek",
            nllb_code="ell_Grek",
            argos_code="el",
            lid_code="ell",
            script="Greek",
            family="Hellenic"
        ),
        LanguageInfo(
            name="Albanian",
            nllb_code="sqi_Latn",
            argos_code="sq",
            lid_code="sqi",
            script="Latin",
            family="Albanian"
        ),

        # ============================================
        # EAST ASIAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Chinese (Simplified)",
            nllb_code="zho_Hans",
            argos_code="zh",
            lid_code="zho_Hans",
            script="Han (Simplified)",
            family="Sino-Tibetan"
        ),
        LanguageInfo(
            name="Chinese (Traditional)",
            nllb_code="zho_Hant",
            argos_code="zt",
            lid_code="zho_Hant",
            script="Han (Traditional)",
            argos_supported=False,
            family="Sino-Tibetan"
        ),
        LanguageInfo(
            name="Cantonese",
            nllb_code="yue_Hant",
            argos_code="yue",
            lid_code="yue",
            script="Han (Traditional)",
            argos_supported=False,
            family="Sino-Tibetan"
        ),
        LanguageInfo(
            name="Japanese",
            nllb_code="jpn_Jpan",
            argos_code="ja",
            lid_code="jpn",
            script="Japanese",
            family="Japonic"
        ),
        LanguageInfo(
            name="Korean",
            nllb_code="kor_Hang",
            argos_code="ko",
            lid_code="kor",
            script="Hangul",
            family="Koreanic"
        ),
        LanguageInfo(
            name="Vietnamese",
            nllb_code="vie_Latn",
            argos_code="vi",
            lid_code="vie",
            script="Latin",
            family="Austroasiatic"
        ),
        LanguageInfo(
            name="Thai",
            nllb_code="tha_Thai",
            argos_code="th",
            lid_code="tha",
            script="Thai",
            family="Kra-Dai"
        ),
        LanguageInfo(
            name="Indonesian",
            nllb_code="ind_Latn",
            argos_code="id",
            lid_code="ind",
            script="Latin",
            family="Austronesian"
        ),
        LanguageInfo(
            name="Malay",
            nllb_code="zsm_Latn",
            argos_code="ms",
            lid_code="zsm",
            script="Latin",
            family="Austronesian"
        ),

        # ============================================
        # MIDDLE EASTERN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Arabic",
            nllb_code="arb_Arab",
            argos_code="ar",
            lid_code="arb",
            script="Arabic",
            family="Semitic"
        ),
        LanguageInfo(
            name="Hebrew",
            nllb_code="heb_Hebr",
            argos_code="he",
            lid_code="heb",
            script="Hebrew",
            family="Semitic"
        ),
        LanguageInfo(
            name="Turkish",
            nllb_code="tur_Latn",
            argos_code="tr",
            lid_code="tur",
            script="Latin",
            family="Turkic"
        ),
        LanguageInfo(
            name="Persian",
            nllb_code="pes_Arab",
            argos_code="fa",
            lid_code="pes",
            script="Arabic",
            family="Iranian"
        ),
        LanguageInfo(
            name="Azerbaijani",
            nllb_code="azj_Latn",
            argos_code="az",
            lid_code="azj",
            script="Latin",
            family="Turkic"
        ),
        LanguageInfo(
            name="Georgian",
            nllb_code="kat_Geor",
            argos_code="ka",
            lid_code="kat",
            script="Georgian",
            argos_supported=False,
            family="Kartvelian"
        ),
        LanguageInfo(
            name="Armenian",
            nllb_code="hye_Armn",
            argos_code="hy",
            lid_code="hye",
            script="Armenian",
            argos_supported=False,
            family="Armenian"
        ),

        # ============================================
        # SOUTH ASIAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Hindi",
            nllb_code="hin_Deva",
            argos_code="hi",
            lid_code="hin",
            script="Devanagari",
            family="Indo-Aryan"
        ),
        LanguageInfo(
            name="Urdu",
            nllb_code="urd_Arab",
            argos_code="ur",
            lid_code="urd",
            script="Arabic",
            family="Indo-Aryan"
        ),
        LanguageInfo(
            name="Bengali",
            nllb_code="ben_Beng",
            argos_code="bn",
            lid_code="ben",
            script="Bengali",
            family="Indo-Aryan"
        ),
        LanguageInfo(
            name="Tamil",
            nllb_code="tam_Taml",
            argos_code="ta",
            lid_code="tam",
            script="Tamil",
            family="Dravidian"
        ),
        LanguageInfo(
            name="Telugu",
            nllb_code="tel_Telu",
            argos_code="te",
            lid_code="tel",
            script="Telugu",
            argos_supported=False,
            family="Dravidian"
        ),
        LanguageInfo(
            name="Marathi",
            nllb_code="mar_Deva",
            argos_code="mr",
            lid_code="mar",
            script="Devanagari",
            argos_supported=False,
            family="Indo-Aryan"
        ),
        LanguageInfo(
            name="Gujarati",
            nllb_code="guj_Gujr",
            argos_code="gu",
            lid_code="guj",
            script="Gujarati",
            argos_supported=False,
            family="Indo-Aryan"
        ),
        LanguageInfo(
            name="Punjabi",
            nllb_code="pan_Guru",
            argos_code="pa",
            lid_code="pan",
            script="Gurmukhi",
            argos_supported=False,
            family="Indo-Aryan"
        ),

        # ============================================
        # AFRICAN LANGUAGES
        # ============================================
        LanguageInfo(
            name="Swahili",
            nllb_code="swh_Latn",
            argos_code="sw",
            lid_code="swh",
            script="Latin",
            family="Bantu"
        ),
        LanguageInfo(
            name="Afrikaans",
            nllb_code="afr_Latn",
            argos_code="af",
            lid_code="afr",
            script="Latin",
            family="Germanic"
        ),

        # ============================================
        # ENGLISH (Target language)
        # ============================================
        LanguageInfo(
            name="English",
            nllb_code="eng_Latn",
            argos_code="en",
            lid_code="eng",
            script="Latin",
            family="Germanic"
        ),
    ]

    def __init__(self) -> None:
        """Initialize the language mapper with lookup dictionaries."""
        # Build lookup dictionaries for fast access
        self._by_nllb: dict[str, LanguageInfo] = {}
        self._by_argos: dict[str, LanguageInfo] = {}
        self._by_lid: dict[str, LanguageInfo] = {}
        self._by_name: dict[str, LanguageInfo] = {}

        for lang in self.LANGUAGES:
            self._by_nllb[lang.nllb_code.lower()] = lang
            self._by_argos[lang.argos_code.lower()] = lang
            self._by_lid[lang.lid_code.lower()] = lang
            self._by_name[lang.name.lower()] = lang

    def get_language(self, code: str) -> Optional[LanguageInfo]:
        """
        Get language info by any code format (NLLB, Argos, LID, or name).

        Args:
            code: Language code in any supported format

        Returns:
            LanguageInfo if found, None otherwise
        """
        code_lower = code.lower()

        # Try each lookup in order of specificity
        if code_lower in self._by_nllb:
            return self._by_nllb[code_lower]
        if code_lower in self._by_argos:
            return self._by_argos[code_lower]
        if code_lower in self._by_lid:
            return self._by_lid[code_lower]
        if code_lower in self._by_name:
            return self._by_name[code_lower]

        return None

    def to_nllb(self, code: str) -> Optional[str]:
        """
        Convert any language code to NLLB format.

        Args:
            code: Language code in any supported format

        Returns:
            NLLB code (e.g., 'rus_Cyrl') or None if not found
        """
        lang = self.get_language(code)
        return lang.nllb_code if lang else None

    def to_argos(self, code: str) -> Optional[str]:
        """
        Convert any language code to Argos (ISO 639-1) format.

        Args:
            code: Language code in any supported format

        Returns:
            Argos code (e.g., 'ru') or None if not found
        """
        lang = self.get_language(code)
        return lang.argos_code if lang else None

    def to_lid(self, code: str) -> Optional[str]:
        """
        Convert any language code to fastText LID format.

        Args:
            code: Language code in any supported format

        Returns:
            LID code (e.g., 'rus') or None if not found
        """
        lang = self.get_language(code)
        return lang.lid_code if lang else None

    def get_name(self, code: str) -> Optional[str]:
        """
        Get human-readable language name from any code format.

        Args:
            code: Language code in any supported format

        Returns:
            Language name (e.g., 'Russian') or None if not found
        """
        lang = self.get_language(code)
        return lang.name if lang else None

    def is_argos_supported(self, code: str) -> bool:
        """
        Check if a language is supported by Argos Translate.

        Args:
            code: Language code in any supported format

        Returns:
            True if Argos supports this language, False otherwise
        """
        lang = self.get_language(code)
        return lang.argos_supported if lang else False

    def get_all_languages(self) -> list[LanguageInfo]:
        """
        Get list of all supported languages.

        Returns:
            List of LanguageInfo objects for all languages
        """
        return self.LANGUAGES.copy()

    def get_languages_by_family(self, family: str) -> list[LanguageInfo]:
        """
        Get all languages belonging to a language family.

        Args:
            family: Language family name (e.g., 'Slavic', 'Germanic')

        Returns:
            List of LanguageInfo objects in that family
        """
        return [lang for lang in self.LANGUAGES if lang.family.lower() == family.lower()]

    def get_nllb_codes(self) -> list[str]:
        """Get list of all NLLB language codes."""
        return [lang.nllb_code for lang in self.LANGUAGES]

    def get_argos_codes(self) -> list[str]:
        """Get list of all Argos (ISO 639-1) language codes."""
        return [lang.argos_code for lang in self.LANGUAGES]

    def get_argos_supported_languages(self) -> list[LanguageInfo]:
        """Get list of languages supported by Argos Translate."""
        return [lang for lang in self.LANGUAGES if lang.argos_supported]

    def format_language_display(self, code: str) -> str:
        """
        Format language info for display in GUI/CLI.

        Args:
            code: Language code in any format

        Returns:
            Formatted string like "Russian (rus_Cyrl)"
        """
        lang = self.get_language(code)
        if lang:
            return f"{lang.name} ({lang.nllb_code})"
        return code

    def get_display_list(self, engine: str = "nllb") -> list[tuple[str, str]]:
        """
        Get list of (display_name, code) tuples for GUI dropdown.

        Args:
            engine: 'nllb' or 'argos' - determines which code format to return

        Returns:
            List of (display_name, code) tuples sorted by name
        """
        result = []
        for lang in self.LANGUAGES:
            if engine == "argos" and not lang.argos_supported:
                continue
            code = lang.nllb_code if engine == "nllb" else lang.argos_code
            result.append((f"{lang.name} ({code})", code))

        return sorted(result, key=lambda x: x[0])


# Pre-instantiated mapper for convenience
_default_mapper = LanguageMapper()


def get_mapper() -> LanguageMapper:
    """Get the default LanguageMapper instance."""
    return _default_mapper


# Convenience functions that use the default mapper
def to_nllb(code: str) -> Optional[str]:
    """Convert any language code to NLLB format."""
    return _default_mapper.to_nllb(code)


def to_argos(code: str) -> Optional[str]:
    """Convert any language code to Argos format."""
    return _default_mapper.to_argos(code)


def get_name(code: str) -> Optional[str]:
    """Get human-readable language name."""
    return _default_mapper.get_name(code)


def is_english(code: str) -> bool:
    """Check if a language code represents English."""
    lang = _default_mapper.get_language(code)
    return lang is not None and lang.nllb_code == "eng_Latn"
