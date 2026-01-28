from __future__ import annotations

import re

import torch


def translate_tableish_text_preserve_lines(
    text: str,
    *,
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_input_tokens: int = 256,  # smaller blocks
    min_new_tokens: int = 64,
    max_new_tokens_cap: int = 256,  # cap per block; avoids long-run truncation surprises
    num_beams: int = 5,
) -> str:
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    lines = text.split("\n")

    def is_separator_line(line: str) -> bool:
        s = line.strip()
        return s == "" or s == "|" or re.fullmatch(r"[|_\-—]+", s) is not None

    blocks: list[tuple[str, bool]] = []
    cur: list[str] = []

    def flush_cur():
        nonlocal cur
        if cur:
            blocks.append(("\n".join(cur), True))
            cur = []

    for line in lines:
        if is_separator_line(line):
            flush_cur()
            blocks.append((line, False))
            continue

        cand = ("\n".join(cur + [line])) if cur else line
        n_tokens = len(tokenizer(cand, add_special_tokens=True).input_ids)

        if n_tokens <= max_input_tokens:
            cur.append(line)
        else:
            flush_cur()
            cur.append(line)

    flush_cur()

    # translate only translatable blocks
    translated = [None] * len(blocks)
    idxs = [i for i, (_, tr) in enumerate(blocks) if tr]
    texts = [blocks[i][0] for i in idxs]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        if device in ("cuda", "mps"):
            enc = {k: v.to(device) for k, v in enc.items()}

        # choose output budget per batch example based on input length
        # (simple heuristic: output <= ~2x input, clipped)
        in_lens = enc["attention_mask"].sum(dim=1).tolist()
        max_new = max(
            min_new_tokens,
            min(max_new_tokens_cap, int(max(in_lens) * 2.0)),
        )

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=num_beams,
                early_stopping=True,
                max_new_tokens=max_new,
            )

        print("input_tokens_max:", int(enc["attention_mask"].sum(dim=1).max()))
        print("output_tokens_max:", max(len(x) for x in gen))

        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, tr_text in enumerate(dec):
            translated[idxs[i + j]] = tr_text

    # rebuild preserving separators and \n between blocks
    out_parts: list[str] = []
    for (block_text, tr), tr_text in zip(blocks, translated, strict=False):
        if not tr:
            out_parts.append(block_text)
        else:
            out_parts.append(tr_text)

    return "\n".join(out_parts)
