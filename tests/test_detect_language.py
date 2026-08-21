import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agentic_rag.build_faiss as build_faiss
import agentic_rag.detect_language as language

pytestmark = pytest.mark.cpu


def _reset_fasttext(monkeypatch: pytest.MonkeyPatch, cache_path: Path, legacy_path: Path) -> None:
    monkeypatch.setattr(language, "FASTTEXT_CACHE_PATH", cache_path)
    monkeypatch.setattr(language, "LEGACY_FASTTEXT_PATH", legacy_path)
    monkeypatch.setattr(language, "_FASTTEXT_MODEL", None)
    monkeypatch.setattr(language, "_FASTTEXT_AVAILABLE", None)


class FastTextPrediction:
    __module__ = "fasttext.FastText"

    def __init__(self, label: str, confidence: float) -> None:
        self.f = SimpleNamespace(
            predict=MagicMock(return_value=[(confidence, f"__label__{label}")]),
        )


def test_language_detection_accepts_italian_and_rejects_obvious_foreign_text(monkeypatch):
    """Accept a normal Italian request while rejecting clear foreign-language controls."""
    monkeypatch.setattr(language, "get_fasttext_model", MagicMock(side_effect=FileNotFoundError))

    detected = language.DetectLanguage("Buongiorno, vorrei informazioni sulla preparazione.", online=False)

    assert detected.lang == "it"
    for text in (
        "Hello. What is the preparation for Colon-TC?",
        "Bonjour! Comment se préparer pour le Colon-TC?",
        "Guten Tag, ich möchte Informationen über die Vorbereitung haben.",
    ):
        with pytest.raises(ValueError, match="Only italian is supported"):
            language.DetectLanguage(text, online=False)


@pytest.mark.parametrize(
    ("question", "fasttext_label", "confidence"),
    [
        pytest.param("Mi date un camice?", "es", 0.4055, id="camice-fasttext-es"),
        pytest.param("La sonda fa male?", "de", 0.4516, id="sonda-fasttext-de"),
        pytest.param("Fa male?", "en", 0.1245, id="male-fasttext-low-confidence-en"),
        pytest.param("Mi fanno la TAC al colon?", "eo", 0.6513, id="tac-fasttext-eo"),
        pytest.param("Mi fanno la colonscopia finta?", "eo", 0.4172, id="colonscopia-fasttext-eo"),
        pytest.param("Mi gonfiano la pancia?", "eo", 0.7175, id="pancia-fasttext-eo"),
        pytest.param("Ho paura del contrasto, me lo fate?", "es", 0.5203, id="contrasto-fasttext-es"),
    ],
)
def test_short_italian_questions_survive_fasttext_false_positives(
    monkeypatch,
    question,
    fasttext_label,
    confidence,
):
    """Replay recorded FastText labels and confidences to correct both threshold paths."""
    model = FastTextPrediction(fasttext_label, confidence)
    monkeypatch.setattr(language, "get_fasttext_model", lambda *, online: model)

    detected = language.DetectLanguage(question, online=False)

    assert detected.lang == "it"


@pytest.mark.parametrize(
    "question",
    [
        "Mi date un camice?",
        "La sonda fa male?",
        "Fa male?",
        "Mi fanno la TAC al colon?",
        "Mi fanno la colonscopia finta?",
        "Mi gonfiano la pancia?",
        "Ho paura del contrasto, me lo fate?",
    ],
)
def test_short_italian_questions_work_without_fasttext(monkeypatch, question):
    """Accept the affected Italian questions when only the fallback detector is available."""
    monkeypatch.setattr(language, "get_fasttext_model", MagicMock(side_effect=FileNotFoundError))

    detected = language.DetectLanguage(question, online=False)

    assert detected.lang == "it"


@pytest.mark.parametrize(
    "question",
    [
        "Mi gusta la comida?",
        "La sonde fait mal?",
        "Does it hurt?",
    ],
)
def test_short_non_italian_questions_remain_rejected_without_fasttext(monkeypatch, question):
    """Reject short foreign controls instead of accepting shared Romance-language words."""
    monkeypatch.setattr(language, "get_fasttext_model", MagicMock(side_effect=FileNotFoundError))

    with pytest.raises(ValueError, match="Only italian is supported"):
        language.DetectLanguage(question, online=False)


def test_short_english_introduction_overrides_fasttext_italian_false_positive(monkeypatch):
    """Reject an English introduction that FastText previously labeled Italian above threshold."""
    model = FastTextPrediction("it", 0.24926041)
    monkeypatch.setattr(language, "get_fasttext_model", lambda *, online: model)

    with pytest.raises(ValueError, match="Only italian is supported"):
        language.DetectLanguage("Hi, I am Nicola", online=False)


@pytest.mark.parametrize("text", ["Mi chiamo Nicola", "Ciao, sono Nicola"])
def test_italian_introductions_are_not_rejected_as_english(monkeypatch, text):
    """Keep Italian introductions valid while rejecting the English regression phrase."""
    monkeypatch.setattr(language, "get_fasttext_model", lambda *, online: FastTextPrediction("it", 0.9))

    assert language.DetectLanguage(text, online=False).lang == "it"


@pytest.mark.parametrize(
    ("fasttext_lang", "fallback_lang", "expected", "fallback_calls"),
    [
        ("it", "fr", "it", 0),
        ("es", "it", "it", 1),
        ("es", "fr", "es", 1),
    ],
)
def test_language_ensemble_truth_table(
    monkeypatch,
    fasttext_lang,
    fallback_lang,
    expected,
    fallback_calls,
):
    """Prefer Italian evidence from either detector without replacing one foreign label with another."""
    detector = object.__new__(language.DetectLanguage)
    detector.online = False
    fallback = MagicMock(return_value=fallback_lang)
    monkeypatch.setattr(language, "get_fasttext_model", MagicMock(return_value=object()))
    monkeypatch.setattr(detector, "_detect_fasttext", MagicMock(return_value=fasttext_lang))
    monkeypatch.setattr(detector, "_robust_detect", fallback)

    assert detector._detect_language("testo con almeno tre parole") == expected
    assert fallback.call_count == fallback_calls


@pytest.mark.parametrize(
    ("confidence", "expected", "fallback_calls"),
    [(0.1999, "it", 1), (0.2, "es", 0)],
)
def test_fasttext_confidence_threshold(monkeypatch, confidence, expected, fallback_calls):
    """Use the fallback below the FastText threshold but trust a prediction exactly at it."""
    detector = object.__new__(language.DetectLanguage)
    detector.online = False
    fallback = MagicMock(return_value="it")
    monkeypatch.setattr(detector, "_robust_detect", fallback)

    assert detector._detect_fasttext("testo con tre parole", FastTextPrediction("es", confidence)) == expected
    assert fallback.call_count == fallback_calls


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        ([SimpleNamespace(lang="it", prob=0.55), SimpleNamespace(lang="ro", prob=0.45)], "it"),
        ([SimpleNamespace(lang="es", prob=0.60), SimpleNamespace(lang="it", prob=0.40)], "unknown"),
        ([SimpleNamespace(lang="fr", prob=0.90)], "fr"),
    ],
)
def test_langdetect_fallback_decisions(monkeypatch, candidates, expected):
    """Handle low-confidence Italian, ambiguous foreign, and confident foreign results deterministically."""
    detector = object.__new__(language.DetectLanguage)
    monkeypatch.setattr(language, "detect_langs", MagicMock(return_value=candidates))

    assert detector._robust_detect("testo neutro senza indizi") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Fa male?", "it"),
        ("FA MALE!", "it"),
        ("grazie, davvero", "it"),
        ("Mi gusta la comida?", "unknown"),
        ("La sonde fait mal?", "unknown"),
        ("male fa", "unknown"),
    ],
)
def test_italian_heuristic_uses_exact_phrases_and_strong_words(text, expected):
    """Accept explicit Italian evidence without treating shared Romance words as sufficient."""
    detector = object.__new__(language.DetectLanguage)

    assert detector._heuristic_detect(text) == expected


def test_langdetect_seed_is_deterministic():
    """Keep langdetect probabilities stable across processes and CI runs."""
    assert language.DetectorFactory.seed == 0


def test_fasttext_prediction_avoids_numpy_copy_false_wrapper():
    """Use FastText's native predictor instead of its NumPy-incompatible wrapper."""

    class NativeFastTextModel:
        __module__ = "fasttext.FastText"

        def __init__(self):
            self.f = SimpleNamespace(predict=MagicMock(return_value=[(0.99, "__label__it")]))

        def predict(self, _text):
            raise AssertionError("The NumPy-incompatible Python wrapper must not be used")

    detector = object.__new__(language.DetectLanguage)
    detector.online = False
    model = NativeFastTextModel()

    assert detector._detect_fasttext("testo italiano", model) == "it"
    model.f.predict.assert_called_once_with("testo italiano\n", 1, 0.0, "strict")


def test_fasttext_generic_prediction_interface():
    """Support non-native FastText-compatible models through their public prediction method."""
    detector = object.__new__(language.DetectLanguage)
    detector.online = False
    model = SimpleNamespace(predict=MagicMock(return_value=(["__label__it"], [0.99])))

    assert detector._detect_fasttext("testo italiano", model) == "it"
    model.predict.assert_called_once_with("testo italiano")


def test_fasttext_empty_prediction_uses_fallback(monkeypatch):
    """Use the fallback detector when FastText returns no label or confidence."""
    detector = object.__new__(language.DetectLanguage)
    detector.online = False
    fallback = MagicMock(return_value="it")
    monkeypatch.setattr(detector, "_robust_detect", fallback)
    model = SimpleNamespace(predict=MagicMock(return_value=([], [])))

    assert detector._detect_fasttext("testo italiano", model) == "it"
    fallback.assert_called_once_with("testo italiano")


@pytest.mark.parametrize(("text", "expected"), [("grazie davvero", "it"), ("neutral words here", "unknown")])
def test_langdetect_exception_uses_heuristic(monkeypatch, text, expected):
    """Use token heuristics when langdetect cannot classify the input."""
    detector = object.__new__(language.DetectLanguage)
    monkeypatch.setattr(language, "detect_langs", MagicMock(side_effect=ValueError("ambiguous")))

    assert detector._robust_detect(text) == expected


def test_fasttext_is_lazy_and_uses_legacy_model_offline(monkeypatch, tmp_path):
    """Load a legacy cached FastText model lazily without attempting a download."""
    reloaded = importlib.reload(language)
    assert reloaded._FASTTEXT_MODEL is None
    assert reloaded._FASTTEXT_AVAILABLE is None

    legacy_path = tmp_path / "legacy" / language.FASTTEXT_MODEL_NAME
    legacy_path.parent.mkdir()
    legacy_path.write_bytes(b"model")
    _reset_fasttext(monkeypatch, tmp_path / "cache" / language.FASTTEXT_MODEL_NAME, legacy_path)
    monkeypatch.setattr(language, "hf_online_enabled", lambda _online: False)
    loaded_model = object()
    fasttext_module = SimpleNamespace(load_model=MagicMock(return_value=loaded_model))
    monkeypatch.setattr(language.importlib, "import_module", MagicMock(return_value=fasttext_module))
    download = MagicMock()
    monkeypatch.setattr(language.urllib.request, "urlretrieve", download)

    assert language.get_fasttext_model() is loaded_model
    fasttext_module.load_model.assert_called_once_with(str(legacy_path))
    download.assert_not_called()


def test_fasttext_offline_cache_miss_never_downloads(monkeypatch, tmp_path):
    """Reject an offline cache miss without making a network request."""
    _reset_fasttext(
        monkeypatch,
        tmp_path / "cache" / language.FASTTEXT_MODEL_NAME,
        tmp_path / "legacy" / language.FASTTEXT_MODEL_NAME,
    )
    monkeypatch.setattr(language, "hf_online_enabled", lambda online: online)
    download = MagicMock()
    monkeypatch.setattr(language.urllib.request, "urlretrieve", download)

    with pytest.raises(FileNotFoundError, match="runtime is offline"):
        language.get_fasttext_model(online=False)
    download.assert_not_called()


def test_fasttext_download_is_atomic_and_cleans_temporary_file(monkeypatch, tmp_path):
    """Publish a validated download atomically and remove its temporary file."""
    cache_path = tmp_path / "cache" / language.FASTTEXT_MODEL_NAME
    _reset_fasttext(monkeypatch, cache_path, tmp_path / "missing" / language.FASTTEXT_MODEL_NAME)
    monkeypatch.setattr(language, "hf_online_enabled", lambda online: online)
    loaded_model = object()
    fasttext_module = SimpleNamespace(load_model=MagicMock(return_value=loaded_model))
    monkeypatch.setattr(language.importlib, "import_module", MagicMock(return_value=fasttext_module))

    def download(_url, filename):
        Path(filename).write_bytes(b"fasttext-model")

    monkeypatch.setattr(language.urllib.request, "urlretrieve", download)

    assert language.get_fasttext_model() is loaded_model
    assert cache_path.read_bytes() == b"fasttext-model"
    assert list(cache_path.parent.glob(f".{cache_path.name}.*.tmp")) == []


def test_fasttext_tries_legacy_after_corrupt_cache(monkeypatch, tmp_path):
    """Use the legacy model when the primary cached FastText model is corrupt."""
    cache_path = tmp_path / "cache" / language.FASTTEXT_MODEL_NAME
    legacy_path = tmp_path / "legacy" / language.FASTTEXT_MODEL_NAME
    cache_path.parent.mkdir()
    legacy_path.parent.mkdir()
    cache_path.write_bytes(b"corrupt")
    legacy_path.write_bytes(b"valid")
    _reset_fasttext(monkeypatch, cache_path, legacy_path)
    loaded_model = object()
    load_model = MagicMock(side_effect=[ValueError("corrupt cache"), loaded_model])
    monkeypatch.setattr(
        language.importlib,
        "import_module",
        MagicMock(return_value=SimpleNamespace(load_model=load_model)),
    )
    download = MagicMock()
    monkeypatch.setattr(language.urllib.request, "urlretrieve", download)

    assert language.get_fasttext_model(online=False) is loaded_model
    assert [invocation.args[0] for invocation in load_model.call_args_list] == [str(cache_path), str(legacy_path)]
    download.assert_not_called()


def test_fasttext_does_not_publish_or_memoize_corrupt_download(monkeypatch, tmp_path):
    """Discard a corrupt download and allow a later FastText load to recover."""
    cache_path = tmp_path / "cache" / language.FASTTEXT_MODEL_NAME
    _reset_fasttext(monkeypatch, cache_path, tmp_path / "legacy" / language.FASTTEXT_MODEL_NAME)
    monkeypatch.setattr(language, "hf_online_enabled", lambda online: online)
    recovered_model = object()
    load_model = MagicMock(side_effect=[ValueError("corrupt download"), recovered_model])
    monkeypatch.setattr(
        language.importlib,
        "import_module",
        MagicMock(return_value=SimpleNamespace(load_model=load_model)),
    )

    def download(_url, filename):
        Path(filename).write_bytes(b"download")

    monkeypatch.setattr(language.urllib.request, "urlretrieve", download)

    with pytest.raises(ValueError, match="corrupt download"):
        language.get_fasttext_model(online=True)
    assert not cache_path.exists()
    assert language._FASTTEXT_MODEL is None

    assert language.get_fasttext_model(online=True) is recovered_model
    assert cache_path.is_file()


def test_fasttext_download_uses_interprocess_lock(monkeypatch, tmp_path):
    """Serialize FastText downloads across processes with the configured lock timeout."""
    cache_path = tmp_path / "cache" / language.FASTTEXT_MODEL_NAME
    _reset_fasttext(monkeypatch, cache_path, tmp_path / "legacy" / language.FASTTEXT_MODEL_NAME)
    monkeypatch.setattr(language, "hf_online_enabled", lambda online: online)
    loaded_model = object()
    monkeypatch.setattr(
        language.importlib,
        "import_module",
        MagicMock(return_value=SimpleNamespace(load_model=MagicMock(return_value=loaded_model))),
    )
    monkeypatch.setattr(
        language.urllib.request,
        "urlretrieve",
        lambda _url, filename: Path(filename).write_bytes(b"model"),
    )
    process_lock = MagicMock()
    process_lock.__enter__.return_value = process_lock
    file_lock = MagicMock(return_value=process_lock)
    monkeypatch.setattr(language, "FileLock", file_lock)

    assert language.get_fasttext_model(online=True) is loaded_model
    file_lock.assert_called_once_with(
        cache_path.with_name(f"{cache_path.name}.lock"),
        timeout=language._FASTTEXT_PROCESS_LOCK_TIMEOUT,
    )


def test_fasttext_loader_is_thread_safe(monkeypatch, tmp_path):
    """Load and memoize the FastText model once across concurrent callers."""
    cache_path = tmp_path / language.FASTTEXT_MODEL_NAME
    cache_path.write_bytes(b"model")
    _reset_fasttext(monkeypatch, cache_path, tmp_path / "legacy.ftz")
    loaded_model = object()
    load_model = MagicMock(return_value=loaded_model)
    monkeypatch.setattr(
        language.importlib,
        "import_module",
        MagicMock(return_value=SimpleNamespace(load_model=load_model)),
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        models = list(executor.map(lambda _index: language.get_fasttext_model(), range(8)))

    assert models == [loaded_model] * 8
    load_model.assert_called_once_with(str(cache_path))


def test_language_validation_uses_exact_membership_and_token_boundaries(monkeypatch):
    """Validate exact language codes, token-aware heuristics, blank input, and accessor codes."""
    detector = object.__new__(language.DetectLanguage)
    assert detector._heuristic_detect("hello") == "unknown"
    assert detector._heuristic_detect("grazie, davvero") == "it"

    monkeypatch.setattr(language.DetectLanguage, "_detect_language", lambda _self, _text: "i")
    with pytest.raises(ValueError, match="Unsupported language 'i'"):
        language.DetectLanguage("testo")

    monkeypatch.setattr(language.DetectLanguage, "_detect_language", lambda _self, _text: "it")
    detected = language.DetectLanguage("  ciao  ")
    assert detected.get("it") == "ciao"
    with pytest.raises(ValueError, match="Language must be italian"):
        detected.get("i")
    with pytest.raises(ValueError, match="must not be blank"):
        language.DetectLanguage("  \t ")


def test_language_detector_forwards_explicit_offline_policy(monkeypatch):
    """Forward the explicit offline policy to the lazy FastText loader."""
    model_loader = MagicMock(side_effect=FileNotFoundError)
    monkeypatch.setattr(language, "get_fasttext_model", model_loader)
    monkeypatch.setattr(language.DetectLanguage, "_robust_detect", lambda _self, _text: "it")

    detected = language.DetectLanguage("testo italiano", online=False)

    assert detected.online is False
    model_loader.assert_called_once_with(online=False)


def test_index_language_detector_recognizes_long_italian_and_english_text():
    """Detect language metadata for sufficiently long indexing text."""
    italian = (
        "Il paziente deve assumere il farmaco ogni mattina a digiuno "
        "e seguire le indicazioni del medico curante con la massima attenzione."
    )
    english = (
        "The patient should take the medication every morning on an empty "
        "stomach and carefully follow all the doctor instructions provided."
    )

    assert build_faiss._detect_lang_safe(italian) == "it"
    assert build_faiss._detect_lang_safe(english) == "en"


@pytest.mark.parametrize("text", [None, "", "x" * 119])
def test_index_language_detector_skips_missing_and_short_text(monkeypatch, text):
    """Return unknown without invoking langdetect below the indexing length threshold."""
    detect = MagicMock()
    monkeypatch.setattr(build_faiss, "detect", detect)

    assert build_faiss._detect_lang_safe(text) == "unknown"
    detect.assert_not_called()


def test_index_language_detector_calls_langdetect_at_exact_threshold(monkeypatch):
    """Invoke langdetect when indexing text reaches exactly 120 characters."""
    detect = MagicMock(return_value="it")
    monkeypatch.setattr(build_faiss, "detect", detect)
    text = "x" * 120

    assert build_faiss._detect_lang_safe(text) == "it"
    detect.assert_called_once_with(text)


def test_index_language_detector_handles_langdetect_failure(monkeypatch):
    """Return unknown when langdetect cannot classify indexing text."""
    monkeypatch.setattr(build_faiss, "detect", MagicMock(side_effect=ValueError("ambiguous")))

    assert build_faiss._detect_lang_safe("x" * 120) == "unknown"
