import os
import urllib

import fasttext
from langdetect import detect_langs
from transformers import pipeline

from ._load_env import cfg


def get_fasttext_model():
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        print("🔽 Downloading FastText language identification model (lid.176.ftz)...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, model_path)
        print("✅ Download complete.")
    return fasttext.load_model(model_path)


try:
    _FASTTEXT_AVAILABLE = True
    _FASTTEXT_MODEL = get_fasttext_model()
except Exception as e:
    print(f"⚠️ FastText model not available ({e}), using fallback detector.")
    _FASTTEXT_AVAILABLE = False

print("_FASTTEXT_AVAILABLE: ", _FASTTEXT_AVAILABLE)

translator_it_en = pipeline("translation", model=cfg.translate_model_it_en)
translator_en_it = pipeline("translation", model=cfg.translate_model_en_it)


class BilingualQuestion:
    def __init__(self, text: str):
        self.text = text.strip()
        self.lang = self._detect_language(self.text)

        if self.lang not in ("it", "en"):
            raise ValueError(f"Unsupported language '{self.lang}'. Only 'it' and 'en' are supported.")

        # Translate
        if self.lang == "it":
            self.it = self.text
            self.en = translator_it_en(self.text)[0]["translation_text"]
        else:
            self.en = self.text
            self.it = translator_en_it(self.text)[0]["translation_text"]

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

    def _detect_fasttext(self, text: str) -> str:
        """Use fastText for reliable detection, even on short text."""
        if not text.strip():
            return "en"
        prediction = _FASTTEXT_MODEL.predict(text.replace("\n", " "))
        lang = prediction[0][0].replace("__label__", "")
        return lang.split("_")[0]  # remove regional code, e.g., 'en_uk' -> 'en'

    def _robust_detect(self, text: str) -> str:
        """Fallback detection using langdetect + heuristic."""
        text = text.strip()
        # Quick heuristic for short text
        if len(text.split()) < 2:
            if any(c in text.lower() for c in ["è", "ò", "à", "ì", "ù"]) or text.lower() in {"ciao", "grazie"}:
                return "it"
            if text.lower() in {"hi", "hello", "thanks"}:
                return "en"

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
        if any(w in t for w in ["il", "la", "che", "non", "per", "ciao", "grazie", "come", "stai"]):
            return "it"
        if any(w in t for w in ["the", "is", "are", "hello", "hi", "thank", "you", "how", "what"]):
            return "en"
        return "en"  # Default to English if uncertain

    def __repr__(self):
        return f"BilingualQuestion(it={self.it!r}, en={self.en!r})"

    def get(self, lang: str) -> str:
        """Return the question in the requested language."""
        if lang not in ("it", "en"):
            raise ValueError("Language must be 'it' or 'en'.")
        return getattr(self, lang)
