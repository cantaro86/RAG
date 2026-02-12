import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import deepl
from langchain_core.documents import Document
from tqdm import tqdm

from src._load_env import cfg, console
from src.loggers import Logger

logger = Logger.get_logger(__name__)


# ============================================================================
# DeepL Configuration & Initialization
# ============================================================================


@lru_cache(maxsize=1)
def _get_deepl_translator() -> deepl.Translator:
    """
    Create and cache DeepL Translator instance.

    Returns:
        Cached DeepL Translator object
    """
    if not hasattr(cfg, "deepl_api_key") or not cfg.deepl_api_key:
        raise RuntimeError(
            "DeepL API key not configured. Set cfg.deepl_api_key. Get free key at: https://www.deepl.com/pro-api"
        )

    translator = deepl.Translator(cfg.deepl_api_key)

    # Log usage
    try:
        usage = translator.get_usage()
        if usage.character.limit:
            remaining = usage.character.limit - usage.character.count
            logger.info(f"DeepL: {remaining:,} / {usage.character.limit:,} chars remaining")
    except Exception as e:
        logger.warning(f"Could not retrieve DeepL usage: {e}")

    return translator


def init_translators():
    """Initialize and verify DeepL translator."""
    logger.info("Initializing DeepL translator...")
    _get_deepl_translator()
    logger.info("✅ DeepL translator initialized")


# ============================================================================
# Text Preprocessing
# ============================================================================


def unwrap_pdf_wrapped_lines(text: str) -> str:
    """Clean up PDF text artifacts."""
    text = text.strip("\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # Join lines, keep paragraphs
    text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces
    return text


# ============================================================================
# Translation Functions
# ============================================================================


def translate_text(text: str, src_lang: str = "it", tgt_lang: str = "en", preserve_formatting: bool = False) -> str:
    """
    Translate text using DeepL.

    Args:
        text: Text to translate
        src_lang: Source language code ('it' or 'en')
        tgt_lang: Target language code ('it' or 'en')
        preserve_formatting: If True, preserve markdown/newlines (for LLM output).
                           If False, clean PDF artifacts (for document chunks).

    Returns:
        Translated text
    """
    translator = _get_deepl_translator()

    # Only clean PDF artifacts if NOT preserving formatting
    if not preserve_formatting:
        text = unwrap_pdf_wrapped_lines(text)

    # Convert to DeepL language codes
    src_lang_deepl = src_lang.upper()  # 'it' -> 'IT', 'en' -> 'EN'

    # Target: Must specify variant for English
    if tgt_lang.lower() == "en":
        tgt_lang_deepl = "EN-US"  # or "EN-GB" for British English
    else:
        tgt_lang_deepl = tgt_lang.upper()  # 'IT'

    # Translate
    result = translator.translate_text(
        text,
        source_lang=src_lang_deepl,
        target_lang=tgt_lang_deepl,
        tag_handling="html" if preserve_formatting else None,
        preserve_formatting=preserve_formatting,
    )

    translated_text = result.text

    # Unescape HTML entities when tag_handling is used
    if preserve_formatting:
        translated_text = html.unescape(translated_text)

    return translated_text


def _translate_single_doc(doc: Document, src_lang: str, tgt_lang: str) -> Document:
    """
    Translate a single document (for parallel processing).

    Args:
        doc: Document to translate
        src_lang: Source language
        tgt_lang: Target language

    Returns:
        Translated Document or original with error metadata
    """
    try:
        translation = translate_text(doc.page_content, src_lang, tgt_lang)

        new_meta = dict(doc.metadata)
        new_meta["orig_lang"] = src_lang
        new_meta["translated_to"] = tgt_lang
        new_meta["orig_page_content"] = doc.page_content
        new_meta["translation_service"] = "deepl"

        return Document(page_content=translation, metadata=new_meta)

    except deepl.DeepLException as e:
        logger.error(f"DeepL translation failed: {e}")
        new_meta = dict(doc.metadata)
        new_meta["translation_error"] = str(e)
        return Document(page_content=doc.page_content, metadata=new_meta)

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        new_meta = dict(doc.metadata)
        new_meta["translation_error"] = str(e)
        return Document(page_content=doc.page_content, metadata=new_meta)


# ============================================================================
# Document Translation
# ============================================================================


def translate_docs(
    docs: list[Document], src_lang: str = "it", tgt_lang: str = "en", parallel: bool = True, max_workers: int = 2
) -> list[Document]:
    """
    Translate documents using DeepL with optional parallelization.

    Args:
        docs: List of Document objects to translate
        src_lang: Source language code ('it' or 'en')
        tgt_lang: Target language code ('it' or 'en')
        parallel: Use parallel processing (default: True for >10 docs)
        max_workers: Number of parallel threads (default: 2)

    Returns:
        List of translated Document objects
    """

    logger.info(
        f"Translating {len(docs)} chunks ({src_lang} → {tgt_lang}) "
        f"{'with ' + str(max_workers) + ' threads' if parallel else 'sequentially'}"
    )
    console.print(f"🗣️  Translating {len(docs)} chunks ({src_lang} → {tgt_lang}) with DeepL...", style="bold")

    if parallel:
        # Parallel translation with threads
        translated_docs = [None] * len(docs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_translate_single_doc, doc, src_lang, tgt_lang): idx for idx, doc in enumerate(docs)
            }

            for future in tqdm(
                as_completed(future_to_idx), total=len(docs), desc=f"Translating ({max_workers} workers)", unit="chunk"
            ):
                idx = future_to_idx[future]
                translated_docs[idx] = future.result()

        out = translated_docs
    else:
        # Sequential translation
        out = []
        for doc in tqdm(docs, desc="Translating sequentially...", unit="chunk"):
            translated = _translate_single_doc(doc, src_lang, tgt_lang)
            out.append(translated)

    # Log progress every 50 chunks
    success_count = sum(1 for d in out if "translation_error" not in d.metadata)
    error_count = len(out) - success_count

    logger.info(f"✅ Translation complete: {success_count} succeeded, {error_count} failed")
    console.print(f"✅ Translation complete: {success_count}/{len(out)} succeeded", style="bold green")

    # Log DeepL usage
    try:
        translator = _get_deepl_translator()
        usage = translator.get_usage()
        if usage.character.limit:
            remaining = usage.character.limit - usage.character.count
            console.print(f"📊 DeepL: {remaining:,} / {usage.character.limit:,} chars remaining", style="blue")
    except Exception as e:
        logger.warning(f"Failed to retrieve DeepL usage: {e}")

    return out


# ============================================================================
# Backward Compatibility
# ============================================================================


def translate_docs_it_to_en(
    docs: list[Document], src_lang: str = "it", tgt_lang: str = "en", parallel: bool = True, max_workers: int = 2
) -> list[Document]:
    """Legacy function name - redirects to translate_docs()"""
    return translate_docs(docs, src_lang, tgt_lang, parallel, max_workers)
