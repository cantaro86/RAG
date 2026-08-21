import importlib
import os
import re
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

from filelock import FileLock
from langdetect import DetectorFactory, detect_langs

from agentic_rag._load_env import EFFECTIVE_HF_HOME, PROJECT_ROOT, hf_online_enabled
from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)

# langdetect otherwise varies probabilities between processes for short text.
DetectorFactory.seed = 0

FASTTEXT_MODEL_NAME = "lid.176.ftz"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_CACHE_PATH = Path(EFFECTIVE_HF_HOME) / "fasttext" / FASTTEXT_MODEL_NAME
LEGACY_FASTTEXT_PATH = PROJECT_ROOT / FASTTEXT_MODEL_NAME

_FASTTEXT_MODEL: Any | None = None
_FASTTEXT_AVAILABLE: bool | None = None
_FASTTEXT_LOCK = threading.Lock()
_FASTTEXT_PROCESS_LOCK_TIMEOUT = 120
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fasttext_candidates() -> tuple[Path, ...]:
    candidates = []
    for path in (FASTTEXT_CACHE_PATH, LEGACY_FASTTEXT_PATH):
        if path.is_file() and path not in candidates:
            candidates.append(path)
    return tuple(candidates)


def _load_usable_candidate(fasttext, candidates: tuple[Path, ...]):
    last_error = None
    for model_path in candidates:
        try:
            return fasttext.load_model(str(model_path)), model_path, None
        except Exception as exc:
            last_error = exc
            logger.warning("Ignoring unusable FastText model at %s: %s", model_path, exc)
    return None, None, last_error


def _download_fasttext_model(target: Path, fasttext):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        logger.info("Downloading FastText language identification model to %s", target)
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, temporary_path)
        if temporary_path.stat().st_size == 0:
            raise OSError("Downloaded FastText model is empty")
        model = fasttext.load_model(str(temporary_path))
        os.replace(temporary_path, target)
        logger.info("FastText model download complete")
        return model
    finally:
        temporary_path.unlink(missing_ok=True)


def get_fasttext_model(*, online: bool | None = None):
    """Load the FastText model once, downloading it only in effective online mode."""
    global _FASTTEXT_AVAILABLE, _FASTTEXT_MODEL

    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL

    effective_online = hf_online_enabled(True if online is None else online)

    with _FASTTEXT_LOCK:
        if _FASTTEXT_MODEL is not None:
            return _FASTTEXT_MODEL

        candidates = _fasttext_candidates()
        fasttext = importlib.import_module("fasttext") if candidates or effective_online else None
        if candidates:
            model, model_path, last_error = _load_usable_candidate(fasttext, candidates)
            if model is not None:
                _FASTTEXT_MODEL = model
                _FASTTEXT_AVAILABLE = True
                logger.info("FastText language model loaded from %s", model_path)
                return _FASTTEXT_MODEL
        else:
            last_error = None

        if not effective_online:
            _FASTTEXT_AVAILABLE = False
            if last_error is not None:
                raise RuntimeError("No usable cached FastText language model is available") from last_error
            raise FileNotFoundError(f"FastText model is not cached at {FASTTEXT_CACHE_PATH} and runtime is offline")

        FASTTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_path = FASTTEXT_CACHE_PATH.with_name(f"{FASTTEXT_CACHE_PATH.name}.lock")
        with FileLock(lock_path, timeout=_FASTTEXT_PROCESS_LOCK_TIMEOUT):
            model, model_path, _last_error = _load_usable_candidate(fasttext, _fasttext_candidates())
            if model is not None:
                _FASTTEXT_MODEL = model
                _FASTTEXT_AVAILABLE = True
                logger.info("FastText language model loaded from %s", model_path)
                return _FASTTEXT_MODEL

            try:
                _FASTTEXT_MODEL = _download_fasttext_model(FASTTEXT_CACHE_PATH, fasttext)
            except Exception as exc:
                _FASTTEXT_AVAILABLE = False
                logger.warning("FastText model download or validation failed: %s", exc)
                raise
            _FASTTEXT_AVAILABLE = True
            logger.info("FastText language model loaded from %s", FASTTEXT_CACHE_PATH)
            return _FASTTEXT_MODEL


class DetectLanguage:
    FASTTEXT_CONFIDENCE_THRESHOLD = 0.2

    MED_PREFIX = "[medical context] "

    ITALIAN_WORDS = {
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
    ITALIAN_STRONG_WORDS = {"addio", "arrivederci", "ciao", "grazie"}
    ITALIAN_SHORT_PHRASES = {("fa", "male")}

    def __init__(self, text: str, *, online: bool | None = None):
        self.text = text.strip()
        if not self.text:
            raise ValueError("Input text must not be blank.")

        self.online = online
        self.lang = self._detect_language(self.text)
        if self.lang != "it":
            raise ValueError(f"Unsupported language '{self.lang}'. Only italian is supported.")

    def _detect_language(self, text: str) -> str:
        """Automatically choose the best available detection method."""
        try:
            model = get_fasttext_model(online=self.online)
        except Exception:
            return self._robust_detect(text)
        fasttext_lang = self._detect_fasttext(text, model)
        if fasttext_lang == "it":
            return fasttext_lang
        fallback_lang = self._robust_detect(text)
        return fallback_lang if fallback_lang == "it" else fasttext_lang

    def _italian_heuristic(self, text: str) -> bool:
        normalized = text.casefold()
        words = tuple(_WORD_RE.findall(normalized))
        return (
            words in self.ITALIAN_SHORT_PHRASES
            or bool(set(words) & self.ITALIAN_STRONG_WORDS)
            or (len(words) == 1 and words[0] in self.ITALIAN_WORDS)
            or any(character in normalized for character in "èéòàìù")
        )

    def _detect_simple_heuristic(self, text: str) -> str | None:
        """Apply token-based rules to very short text."""
        return "it" if self._italian_heuristic(text) else None

    def _detect_fasttext(self, text: str, model=None) -> str:
        """Use FastText for reliable detection, including short text."""
        logger.debug("FastText detection")

        if len(text.split()) <= 2:
            heuristic_lang = self._detect_simple_heuristic(text)
            if heuristic_lang:
                return heuristic_lang

        model = model or get_fasttext_model(online=getattr(self, "online", None))
        clean_text = text.replace("\n", " ")
        if type(model).__module__ == "fasttext.FastText" and hasattr(model, "f"):
            predictions = model.f.predict(f"{clean_text}\n", 1, 0.0, "strict")
            if predictions:
                probabilities, labels = zip(*predictions, strict=False)
            else:
                probabilities, labels = (), ()
            prediction = labels, probabilities
        else:
            prediction = model.predict(clean_text)
        logger.debug("FastText prediction: %s", prediction)

        if not prediction[0] or not prediction[1]:
            return self._robust_detect(text)
        if prediction[1][0] < self.FASTTEXT_CONFIDENCE_THRESHOLD:
            return self._robust_detect(text)

        lang = prediction[0][0].replace("__label__", "")
        return lang.split("_")[0]

    def _robust_detect(self, text: str) -> str:
        """Use langdetect and token-based rules when FastText is unavailable."""
        logger.debug("Robust detection")

        if len(text.split()) <= 2:
            heuristic_lang = self._detect_simple_heuristic(text)
            if heuristic_lang:
                return heuristic_lang

        try:
            langs = detect_langs(text)
            best = max(langs, key=lambda result: result.prob)
            if best.lang == "it":
                return best.lang
            if best.prob < 0.8:
                return self._heuristic_detect(text)
            return best.lang
        except Exception:
            return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> str:
        """Apply fallback rules to short or ambiguous text."""
        return "it" if self._italian_heuristic(text) else "unknown"

    def __repr__(self):
        return f"DetectLanguage({self.text!r})"

    def get(self, lang: str) -> str:
        """Return the question in Italian."""
        if lang != "it":
            raise ValueError("Language must be italian.")
        return self.text
