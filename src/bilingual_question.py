import os
import urllib
from functools import lru_cache

import fasttext
from langdetect import detect_langs

from ._load_env import cfg
from .llm_build import load_translator
from .loggers import Logger

logger = Logger.get_logger(__name__)


def get_fasttext_model():
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        logger.info("🔽 Downloading FastText language identification model (lid.176.ftz)...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, model_path)
        logger.info("✅ Download complete.")
    return fasttext.load_model(model_path)


try:
    _FASTTEXT_AVAILABLE = True
    _FASTTEXT_MODEL = get_fasttext_model()
except Exception as e:
    logger.warning(f"⚠️ FastText model not available ({e}), using fallback detector.")
    _FASTTEXT_AVAILABLE = False

logger.info(f"_FASTTEXT_AVAILABLE: {_FASTTEXT_AVAILABLE}")

# -------------------------


@lru_cache(maxsize=1)
def _load_translators():
    # loaded once, on first use
    return load_translator(cfg.translate_model_it_en), load_translator(cfg.translate_model_en_it)


def init_translators():
    """Warm up translators at startup. Returns (it_en, en_it)."""
    return _load_translators()


class BilingualQuestion:
    FASTTEXT_CONFIDENCE_THRESHOLD = 0.4

    ITALIAN_WORLDS = {
        "ciao",
        "salve",
        "arrivederci",
        "grazie",
        "addio",
        "il",
        "la",
        "lo",
        "gli",
        "le",
        "di",
        "dove",
        "da",
        "quando",
        "che",
        "non",
        "per",
        "come",
        "stai",
    }

    ENGLISH_WORDS = {
        "hi",
        "hello",
        "thanks",
        "bye",
        "goodbye",
        "the",
        "is",
        "are",
        "thank",
        "you",
        "how",
        "what",
        "where",
        "when",
        "who",
        "why",
    }

    def __init__(self, text: str):
        self.text = text.strip()
        self.lang = self._detect_language(self.text)

        if self.lang not in ("it", "en"):
            raise ValueError(f"Unsupported language '{self.lang}'. Only 'it' and 'en' are supported.")

        translator_it_en, translator_en_it = _load_translators()

        # Translate
        if self.lang == "it":
            self.it = self.text
            self.en = translator_it_en(self.text)[0]["translation_text"]
        else:
            self.en = self.text
            self.it = translator_en_it(self.text)[0]["translation_text"]

    def translate_to_italian(self, text: str) -> str:
        """Assume the input is in English"""
        text_it = _load_translators()[1](text.strip())[0]["translation_text"]
        return text_it

    # -------------------------
    # 🔍 Detection Methods
    # -------------------------

    def _detect_language(self, text: str) -> str:
        """Automatically choose best detection method."""
        if _FASTTEXT_AVAILABLE:
            lang = self._detect_fasttext(text)
        else:
            lang = self._robust_detect(text)
        return lang

    def _detect_simple_heuristic(self, text: str) -> str:
        """Simple heuristic detection for very short text."""
        if any(c in text.lower() for c in ["è", "é", "ò", "à", "ì", "ù"]) or text.lower() in self.ITALIAN_WORLDS:
            return "it"
        if text.lower() in self.ENGLISH_WORDS:
            return "en"

    def _detect_fasttext(self, text: str) -> str:
        """Use fastText for reliable detection, even on short text."""

        logger.debug("fastText detection")

        if not text.strip():
            return "en"

        # Quick heuristic for short text
        if len(text.split()) < 2:
            heuristic_lang = self._detect_simple_heuristic(text)
            if heuristic_lang:
                return heuristic_lang

        prediction = _FASTTEXT_MODEL.predict(text.replace("\n", " "))

        logger.debug(f"fastText prediction: {prediction}")

        if prediction[1][0] < self.FASTTEXT_CONFIDENCE_THRESHOLD:
            return self._robust_detect(text)

        lang = prediction[0][0].replace("__label__", "")
        return lang.split("_")[0]  # remove regional code, e.g., 'en_uk' -> 'en'

    def _robust_detect(self, text: str) -> str:
        """Fallback detection using langdetect + heuristic."""

        logger.debug("Robust detection")

        text = text.strip()

        # Quick heuristic for short text
        if len(text.split()) < 2:
            heuristic_lang = self._detect_simple_heuristic(text)
            if heuristic_lang:
                return heuristic_lang

        try:
            langs = detect_langs(text)
            best = max(langs, key=lambda x: x.prob)
            if best.prob < 0.8:
                return self._heuristic_detect(text)
            return best.lang
        except Exception:
            return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> str:
        """Fallback rules for very short or ambiguous text."""
        t = text.lower()
        if any(w in t for w in self.ITALIAN_WORLDS):
            return "it"
        if any(w in t for w in self.ENGLISH_WORDS):
            return "en"
        return "en"  # Default to English if uncertain

    def __repr__(self):
        return f"BilingualQuestion(it={self.it!r}, en={self.en!r})"

    def get(self, lang: str) -> str:
        """Return the question in the requested language."""
        if lang not in ("it", "en"):
            raise ValueError("Language must be 'it' or 'en'.")
        return getattr(self, lang)
