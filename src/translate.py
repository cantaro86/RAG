import re

import torch

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
    batch_size: int = 16,
    max_input_tokens: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
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
            batch = sents[i : i + batch_size]
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
