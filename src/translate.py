import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import argostranslate.package
import argostranslate.translate
from argostranslate.translate import ITranslation
from langchain_core.documents import Document
from tqdm import tqdm

from src._load_env import console
from src.loggers import Logger

logger = Logger.get_logger(__name__)


# ============================================================================
# Model Loading & Initialization
# ============================================================================


def setup_argos_translation():
    """
    Download and install Argos translation packages if not already installed.
    """
    logger.info("Checking Argos Translate packages...")

    # Check what's already installed
    installed_languages = argostranslate.translate.get_installed_languages()
    installed_codes = {lang.code for lang in installed_languages}

    # Check if we have both it and en
    if "it" in installed_codes and "en" in installed_codes:
        # Check if translations exist
        try:
            it_lang = next(filter(lambda x: x.code == "it", installed_languages))
            en_lang = next(filter(lambda x: x.code == "en", installed_languages))

            if it_lang.get_translation(en_lang) and en_lang.get_translation(it_lang):
                logger.info("✅ Translation packages already installed")
                return
        except Exception as e:  # ✅ Only catch exceptions, not system exits
            logger.warning(f"Translation check failed: {e}")
            pass

    logger.info("Installing Argos Translate packages...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Install Italian <-> English packages
    for from_code, to_code in [("it", "en"), ("en", "it")]:
        package = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages), None)
        if package:
            logger.info(f"Installing {from_code} -> {to_code} translation package")
            argostranslate.package.install_from_path(package.download())
        else:
            logger.warning(f"Package {from_code} -> {to_code} not found")

    logger.info("✅ Argos Translate setup complete")


@lru_cache(maxsize=2)
def _get_translation_model(from_lang: str, to_lang: str) -> ITranslation:
    """
    Load and cache translation model for specific language pair.
    maxsize=2 cache up to 2 different combinations of (from_lang, to_lang).

    Args:
        from_lang: Source language code ('it' or 'en')
        to_lang: Target language code ('it' or 'en')

    Returns:
        Cached translation model
    """
    installed_languages = argostranslate.translate.get_installed_languages()

    from_language = next(filter(lambda x: x.code == from_lang, installed_languages), None)
    to_language = next(filter(lambda x: x.code == to_lang, installed_languages), None)

    if from_language is None or to_language is None:
        raise RuntimeError(f"Language pair {from_lang}->{to_lang} not installed. Run setup_argos_translation() first.")

    return from_language.get_translation(to_language)


def init_translators():
    """
    Warm up translators at startup.
    Pre-loads both it->en and en->it models.
    """
    logger.info("Initializing translation models...")
    try:
        # Try to load models, if not installed, set them up first
        _get_translation_model("it", "en")
        _get_translation_model("en", "it")
        logger.info("✅ Translation models initialized")
    except RuntimeError:
        logger.info("Translation packages not found, installing...")
        setup_argos_translation()
        _get_translation_model("it", "en")
        _get_translation_model("en", "it")
        logger.info("✅ Translation models initialized")


# ============================================================================
# Text Preprocessing
# ============================================================================


def unwrap_pdf_wrapped_lines(text: str) -> str:
    """
    Clean up PDF text artifacts: unwrap hyphenated line breaks,
    join lines within paragraphs, preserve paragraph boundaries.
    """
    text = text.strip("\n")

    # Join hyphenated line breaks: "prepara-\nzione" -> "preparazione"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join non-hyphen line breaks inside paragraphs: "\n" -> " "
    # but preserve paragraph breaks "\n\n"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text


# ============================================================================
# Translation Functions
# ============================================================================


def _translate_text_core(text: str, translation_model: ITranslation, max_segment_size: int = 500) -> str:
    """Core translation logic without model loading"""
    text = unwrap_pdf_wrapped_lines(text)

    if len(text) <= max_segment_size:
        return translation_model.translate(text)

    paragraphs = text.split("\n\n")
    translated_paras = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            translated_paras.append("")
            continue
        translated = translation_model.translate(para)

        expansion_ratio = len(translated) / len(para) if para else 0
        if expansion_ratio > 10:
            logger.warning(
                f"Translation expansion {expansion_ratio:.1f}x detected. "
                f"Possible quality issue. Original: {para} \n Translated: {translated}"
            )

        translated_paras.append(translated)

    return "\n\n".join(translated_paras)


def translate_text(text: str, src_lang: str = "it", tgt_lang: str = "en", max_segment_size: int = 500) -> str:
    """
    Translate text using Argos Translate.
    Handles both short and long text automatically.

    Args:
        text: Text to translate
        src_lang: Source language code ('it' or 'en')
        tgt_lang: Target language code ('it' or 'en')
        max_segment_size: Max characters per segment for quality

    Returns:
        Translated text
    """
    model = _get_translation_model(src_lang, tgt_lang)
    return _translate_text_core(text, model, max_segment_size)


# ============================================================================
# Sequential Translation (Simple)
# ============================================================================


def translate_docs_sequential(docs: list[Document], src_lang: str = "it", tgt_lang: str = "en") -> list[Document]:
    """
    Translate documents sequentially (no parallelization).
    Simpler, but slower for large document sets.

    Args:
        docs: List of Document objects to translate
        src_lang: Source language code
        tgt_lang: Target language code

    Returns:
        List of translated Document objects
    """
    logger.info(f"Starting sequential translation of {len(docs)} chunks")
    console.print(f"🗣️  Translating {len(docs)} chunks ({src_lang} → {tgt_lang})...", style="bold")

    out = []
    for i, d in enumerate(tqdm(docs, desc="Translate chunks", unit="chunk")):
        try:
            translation = translate_text(d.page_content, src_lang=src_lang, tgt_lang=tgt_lang)

            # Create new document with translation
            new_meta = dict(d.metadata)
            new_meta["orig_lang"] = src_lang
            new_meta["translated_to"] = tgt_lang
            new_meta["orig_page_content"] = d.page_content

            out.append(Document(page_content=translation, metadata=new_meta))

            # Log progress every 50 chunks
            if (i + 1) % 50 == 0:
                logger.info(f"Translated {i + 1}/{len(docs)} chunks")
                console.print(f"  ✓ {i + 1}/{len(docs)} chunks done", style="green")

        except Exception as e:
            logger.error(f"Translation failed for chunk {i} ({d.metadata.get('source', 'unknown')}): {e}")
            console.print(f"  ❌ Chunk {i} failed: {e}", style="red")

            # Keep original with error metadata
            new_meta = dict(d.metadata)
            new_meta["translation_error"] = str(e)
            out.append(Document(page_content=d.page_content, metadata=new_meta))

    logger.info(f"✅ Translation complete: {len(out)} chunks processed")
    console.print(f"✅ Translation complete: {len(out)} chunks processed", style="bold green")

    return out


# ============================================================================
# Parallel Translation (Fast)
# ============================================================================


def _init_worker_process(src_lang: str, tgt_lang: str):
    """Initialize translation model in worker process (for parallel execution)"""
    global _worker_translation_model

    installed_languages = argostranslate.translate.get_installed_languages()
    from_language = next(filter(lambda x: x.code == src_lang, installed_languages))
    to_language = next(filter(lambda x: x.code == tgt_lang, installed_languages))

    _worker_translation_model = from_language.get_translation(to_language)


def _translate_chunk_worker(text: str, max_segment_size: int = 500) -> str:
    """Parallel version - uses pre-loaded global model"""
    global _worker_translation_model
    return _translate_text_core(text, _worker_translation_model, max_segment_size)


def translate_docs_parallel(
    docs: list[Document], src_lang: str = "it", tgt_lang: str = "en", max_workers: int = 2
) -> list[Document]:
    """
    Translate documents in parallel using ProcessPoolExecutor.
    Faster for large document sets (hundreds of chunks).

    Args:
        docs: List of Document objects to translate
        src_lang: Source language code
        tgt_lang: Target language code
        max_workers: Number of parallel workers

    Returns:
        List of translated Document objects
    """

    logger.info(f"Starting parallel translation of {len(docs)} chunks with {max_workers} workers")
    console.print(
        f"🗣️  Translating {len(docs)} chunks ({src_lang} → {tgt_lang}) with {max_workers} workers...", style="bold"
    )

    translations = [None] * len(docs)

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker_process, initargs=(src_lang, tgt_lang)
    ) as executor:
        # Submit all translation tasks
        from concurrent.futures import as_completed

        future_to_idx = {
            executor.submit(_translate_chunk_worker, doc.page_content): idx for idx, doc in enumerate(docs)
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(docs), desc="Translate chunks", unit="chunk"):
            idx = future_to_idx[future]
            try:
                translation = future.result()
                translations[idx] = translation
            except Exception as e:
                logger.error(f"Translation failed for chunk {idx}: {e}")
                translations[idx] = None  # Mark as failed

    # Build output documents
    out = []
    for i, (doc, translation) in enumerate(zip(docs, translations, strict=False)):
        if translation is not None:
            new_meta = dict(doc.metadata)
            new_meta["orig_lang"] = src_lang
            new_meta["translated_to"] = tgt_lang
            new_meta["orig_page_content"] = doc.page_content

            out.append(Document(page_content=translation, metadata=new_meta))
        else:
            # Keep original if translation failed
            console.print(f"  ❌ Chunk {i} failed", style="red")
            new_meta = dict(doc.metadata)
            new_meta["translation_error"] = "Parallel translation failed"
            out.append(Document(page_content=doc.page_content, metadata=new_meta))

    logger.info(f"✅ Translation complete: {len(out)} chunks processed")
    console.print(f"✅ Translation complete: {len(out)} chunks processed", style="bold green")

    return out


# ============================================================================
# Main Translation Function (Auto-selects strategy)
# ============================================================================


def translate_docs(
    docs: list[Document], src_lang: str = "it", tgt_lang: str = "en", parallel: bool = True, max_workers: int = 2
) -> list[Document]:
    """
    Translate documents with automatic strategy selection.

    Args:
        docs: List of Document objects to translate
        src_lang: Source language code ('it' or 'en')
        tgt_lang: Target language code ('it' or 'en')
        parallel: Use parallel processing (default True for >10 docs)
        max_workers: Number of parallel workers (only if parallel=True)

    Returns:
        List of translated Document objects
    """
    # Auto-select strategy based on document count
    if parallel and len(docs) > 10:
        return translate_docs_parallel(docs, src_lang, tgt_lang, max_workers)
    else:
        return translate_docs_sequential(docs, src_lang, tgt_lang)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================


def translate_docs_it_to_en(
    docs: list[Document], src_lang: str = "it", tgt_lang: str = "en", parallel: bool = True, max_workers: int = 2
) -> list[Document]:
    """Legacy function name - redirects to translate_docs()"""
    return translate_docs(docs, src_lang=src_lang, tgt_lang=tgt_lang, parallel=parallel, max_workers=max_workers)
