import re
from functools import lru_cache

import torch
from langchain_core.documents import Document
from tqdm import tqdm

from src._load_env import DEVICE, cfg, console
from src.llm_build import load_translator
from src.loggers import Logger

logger = Logger.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_translators():
    """
    Load single NLLB model for bidirectional translation.
    Returns (model, tokenizer) tuple - loaded once on first use.
    """
    return load_translator(cfg.translate_model)


def init_translators():
    """Warm up translators at startup. Returns (it_en, en_it)."""
    return _load_translators()


def translate_short_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text using NLLB model.

    Args:
        text: Text to translate
        src_lang: Source language code ('ita_Latn' or 'eng_Latn')
        tgt_lang: Target language code ('ita_Latn' or 'eng_Latn')
    """
    model, tokenizer = _load_translators()

    # Set source language
    tokenizer.src_lang = src_lang

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Move to device
    if DEVICE in ["cuda", "mps"]:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate translation
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_new_tokens=128, num_beams=5
    )

    # Decode
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


# The best batch size is hardware-dependent.

# return_tensors="pt": returns tokenized outputs as torch.Tensor objects.
# Without it you’d typically get Python lists.
#
# padding=True pads shorter sequences in the batch so that all examples have the same length,
# which is required to stack them into a single tensor batch.
# With True, padding is usually to the length of the longest sequence in that batch (dynamic padding)
#
# truncation=True: if an input is longer than the limit, it is cut to fit the maximum length
#
# max_length=max_input_tokens: the maximum number of tokens used for padding/truncation (not characters).
# If truncation=True, longer sequences are truncated to this value;
# if padding uses a fixed strategy, it pads up to this value.

# forced_bos_token_id=forced_bos: forces the first generated token to be a specific token id.
# For NLLB this is how you force the target language at generation time (you pass the id of eng_Latn, ita_Latn, etc.).

# num_beams=num_beams: beam search width.
# Higher values explore more candidate translations and often improve quality, but increase compute/latency.
# It doesn’t “translate more text”; it mostly changes which translation is chosen.

# max_new_tokens=max_new_tokens: caps how many new tokens the model may generate (output length limit).

# early_stopping=True: in beam search,
# stops once the algorithm decides continuing won’t improve the best finished hypotheses

# English: often ~0.7–1.3 tokens per word (so 256 tokens might be ~200–350 words).


# This split the text into a list of sentences based on punctuation.
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def unwrap_pdf_wrapped_lines(text: str) -> str:
    # Keep blank lines as paragraph separators
    text = text.strip("\n")

    # Join hyphenated line breaks: "prepara-\nzione" -> "preparazione"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join non-hyphen line breaks inside paragraphs: "\n" -> " "
    # but preserve paragraph breaks "\n\n"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text


def translate_paragraphs_by_sentences(
    text: str,
    *,
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str,
    batch_size: int = 32,
    max_input_tokens: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 5,
) -> str:
    tokenizer.src_lang = src_lang

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    text = unwrap_pdf_wrapped_lines(text)
    paragraphs = text.split("\n\n")

    out_paras = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            out_paras.append("")
            continue

        sents = _SENT_SPLIT.split(para)

        translated = []
        for i in range(0, len(sents), batch_size):
            batch = [s for s in sents[i : i + batch_size] if s.strip()]
            if not batch:
                continue

            enc = tokenizer(
                batch,
                return_tensors="pt",  # pythorch tensor instead of list
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            if device in ("cuda", "mps"):
                enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    early_stopping=True,
                )
            # Extend list by appending elements from the iterable.
            translated.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

        out_paras.append(" ".join(t.strip() for t in translated if t.strip()))

    return "\n\n".join(out_paras)


def translate_docs_it_to_en(docs: list[Document], src_lang: str, tgt_lang: str) -> list[Document]:
    """
    Translate Italian documents to English with progress bar and logging.
    """
    model, tokenizer = _load_translators()

    logger.info(f"Starting translation of {len(docs)} Italian chunks")
    console.print(f"🗣️  Translating {len(docs)} Italian chunks to English...", style="bold")

    out = []
    for i, d in enumerate(tqdm(docs, desc="Translate chunks", unit="chunk")):
        try:
            en_text = translate_paragraphs_by_sentences(
                d.page_content,
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                device=DEVICE,
            )

            # clone doc, keep provenance
            new_meta = dict(d.metadata)
            new_meta["orig_lang"] = "it"
            new_meta["translated_to"] = "en"
            new_meta["orig_page_content"] = d.page_content

            out.append(Document(page_content=en_text, metadata=new_meta))

            # Optional: log every Nth chunk for long jobs
            if (i + 1) % 50 == 0:
                logger.info(f"Translated {i + 1}/{len(docs)} chunks")
                console.print(f"  ✓ {i + 1}/{len(docs)} chunks done", style="green")

        except Exception as e:
            logger.error(f"Translation failed for chunk {i} ({d.metadata.get('source', 'unknown')}): {e}")
            console.print(f"  ❌ Chunk {i} failed: {e}", style="red")
            # Keep original chunk with error metadata instead of crashing
            new_meta = dict(d.metadata)
            new_meta["translation_error"] = str(e)
            out.append(Document(page_content=d.page_content, metadata=new_meta))

    logger.info(f"Translation complete: {len(out)} chunks processed")
    console.print(f"✅ Translation complete: {len(out)} chunks processed", style="bold green")

    return out


def translate_long_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate long text (e.g., final RAG answer) using sentence-level batching.
    Avoids truncation by splitting into manageable chunks.
    """
    model, tokenizer = _load_translators()

    return translate_paragraphs_by_sentences(
        text,
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=DEVICE,
        batch_size=16,  # smaller batch for long sentences
        max_input_tokens=256,  # per-sentence token limit
        max_new_tokens=256,  # allows longer outputs per sentence
        num_beams=5,  # balance quality/speed
    )
