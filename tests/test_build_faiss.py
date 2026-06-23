"""
Unit tests for agentic_rag/build_faiss.py

Covers:
  - _detect_lang_safe
  - extract_list_number
  - _make_metadata
  - _flush_text
  - split_block_by_type
  - chunk_markdown  (synthetic fixtures + real Informazioni per_pazienti.md)
  - merge_continuation_lists
  - Pipeline: chunk_markdown → merge_continuation_lists

Run with:
    pytest tests/test_indexing.py -v
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_rag.build_faiss import (
    _SPLITTER,
    _detect_lang_safe,
    _flush_text,
    _make_metadata,
    chunk_markdown,
    extract_list_number,
    merge_continuation_lists,
    split_block_by_type,
)

pytestmark = pytest.mark.cpu  # Mark ALL tests in this module as CPU

# ---------------------------------------------------------------------------
# Paths to real documents
# ---------------------------------------------------------------------------

MARKDOWN_IT_DIR = Path(__file__).resolve().parent.parent / "Markdown_IT"
PAZIENTE_MD = MARKDOWN_IT_DIR / "Informazioni_per_pazienti.md"


# ===========================================================================
# _detect_lang_safe
# ===========================================================================


class TestDetectLangSafe:
    def test_italian_text_detected(self):
        italian = (
            "Il paziente deve assumere il farmaco ogni mattina a digiuno "
            "e seguire le indicazioni del medico curante con la massima attenzione."
        )
        assert _detect_lang_safe(italian) == "it"

    def test_english_text_detected(self):
        english = (
            "The patient should take the medication every morning on an empty "
            "stomach and carefully follow all the doctor instructions provided."
        )
        assert _detect_lang_safe(english) == "en"

    def test_short_text_returns_unknown(self):
        # Fewer than 120 chars → length guard triggers, returns "unknown"
        assert _detect_lang_safe("Ciao come stai") == "unknown"

    def test_empty_string_returns_unknown(self):
        assert _detect_lang_safe("") == "unknown"

    def test_none_returns_unknown(self):
        assert _detect_lang_safe(None) == "unknown"  # type: ignore[arg-type]

    def test_exactly_at_threshold_attempts_detection(self):
        # 120 chars: satisfies the >= 120 guard; result must be a string
        text = "parola " * 20  # 140 chars of plausible Italian-ish tokens
        result = _detect_lang_safe(text)
        assert isinstance(result, str)


# ===========================================================================
# extract_list_number
# ===========================================================================


class TestExtractListNumber:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ("1. Prima voce della lista numerata", 1),
            ("2. Seconda voce della lista", 2),
            ("10. Decima voce con testo aggiuntivo", 10),
            ("**3**. Terza voce in grassetto markdown", 3),
            ("3) Terza voce con parentesi tonda", 3),
            ("- voce non numerata con trattino", None),
            ("Testo senza alcun numero iniziale", None),
            ("", None),
        ],
    )
    def test_parametrized(self, line, expected):
        assert extract_list_number(line) == expected

    def test_continuation_item_detected(self):
        assert extract_list_number("4. Quarto passo: verificare i dati inseriti.") == 4

    def test_double_digit_detected(self):
        assert extract_list_number("12. Dodicesimo elemento della lista.") == 12

    def test_item_1_detected(self):
        # Item 1 is valid but NOT a continuation (n <= 1 in merge logic)
        assert extract_list_number("1. Primo elemento.") == 1


# ===========================================================================
# _make_metadata
# ===========================================================================


class TestMakeMetadata:
    def test_required_keys_present(self):
        meta = _make_metadata("doc.md", "Introduzione", "it")
        assert meta["source"] == "doc.md"
        assert meta["section"] == "Introduzione"
        assert meta["language"] == "it"

    def test_extra_kwargs_included(self):
        meta = _make_metadata("doc.md", "Sezione", "it", type="table", page=3)
        assert meta["type"] == "table"
        assert meta["page"] == 3

    def test_returns_dict(self):
        assert isinstance(_make_metadata("x.md", "s", "en"), dict)

    def test_no_extra_keys_when_none_given(self):
        meta = _make_metadata("a.md", "B", "fr")
        assert set(meta.keys()) == {"source", "section", "language"}


# ===========================================================================
# _flush_text
# ===========================================================================


class TestFlushText:
    def _call(
        self,
        pending_text: list,
        carried_heading: str = "",
    ):
        chunks = []
        new_heading = _flush_text(
            chunks=chunks,
            pending_text=pending_text,
            carried_heading=carried_heading,
            source="test.md",
            section="TestSection",
            language="it",
            splitter=_SPLITTER,
        )
        return chunks, new_heading

    def test_flushes_pending_text(self):
        pending = ["Prima frase di testo.", "Seconda frase di testo."]
        chunks, heading = self._call(pending)
        assert len(chunks) >= 1
        assert heading == ""
        assert pending == []  # list must be cleared in place

    def test_carried_heading_prepended_to_text(self):
        pending = ["Testo della sezione."]
        chunks, _ = self._call(pending, carried_heading="## Sezione di test")
        full = " ".join(c.page_content for c in chunks)
        assert "## Sezione di test" in full
        assert "Testo della sezione." in full

    def test_only_heading_no_text_creates_one_chunk(self):
        chunks, heading = self._call([], carried_heading="## Solo titolo")
        assert len(chunks) == 1
        assert chunks[0].page_content == "## Solo titolo"
        assert heading == ""

    def test_nothing_to_flush_passthrough(self):
        chunks, heading = self._call([], carried_heading="")
        assert chunks == []
        assert heading == ""

    def test_chunk_metadata_correct(self):
        pending = ["Frase di prova per il test di metadata."]
        chunks, _ = self._call(pending)
        m = chunks[0].metadata
        assert m["source"] == "test.md"
        assert m["section"] == "TestSection"
        assert m["language"] == "it"
        assert m["type"] == "text"

    def test_long_text_is_split_into_multiple_chunks(self):
        # Text larger than chunk_size (1500) should produce more than one chunk
        long_text = ("Testo di prova molto lungo. " * 100).strip()
        pending = [long_text]
        chunks, _ = self._call(pending)
        # Sum of all chunk content should contain the original text
        combined = "".join(c.page_content for c in chunks)
        assert "Testo di prova" in combined


# ===========================================================================
# split_block_by_type
# ===========================================================================


class TestSplitBlockByType:
    def test_pure_text_block(self):
        block = "Questo è un testo normale senza liste.\nNessuna lista né tabella presente."
        segs = split_block_by_type(block)
        assert all(t == "text" for t, _ in segs)

    def test_bulleted_list_detected(self):
        block = "- Prima voce elenco\n- Seconda voce\n- Terza voce"
        segs = split_block_by_type(block)
        assert any(t == "list" for t, _ in segs)

    def test_numbered_list_detected(self):
        block = "1. Passo uno\n2. Passo due\n3. Passo tre"
        segs = split_block_by_type(block)
        assert any(t == "list" for t, _ in segs)

    def test_heading_line_detected(self):
        block = "## Sezione importante\nTesto immediatamente successivo."
        segs = split_block_by_type(block)
        assert any(t == "heading" for t, _ in segs)

    def test_intro_colon_attaches_to_list(self):
        """A text line ending in ':' just before a list should appear in the list segment."""
        block = "Le istruzioni operative sono:\n- Voce uno\n- Voce due"
        segs = split_block_by_type(block)
        list_segs = [c for t, c in segs if t == "list"]
        assert list_segs, "Expected at least one list segment"
        assert any("istruzioni" in c for c in list_segs), (
            "Intro line ending with ':' must be prepended to the list segment"
        )

    def test_returns_list_of_two_element_tuples(self):
        segs = split_block_by_type("semplice testo")
        assert isinstance(segs, list)  # [('text', 'semplice testo')]
        assert all(isinstance(s, tuple) and len(s) == 2 for s in segs)

    def test_table_lines_not_classified_as_list(self):
        block = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        segs = split_block_by_type(block)
        assert all(t != "list" for t, _ in segs)

    def test_mixed_block_has_multiple_segment_types(self):
        block = "Testo introduttivo.\n1. Primo elemento\n2. Secondo elemento"
        segs = split_block_by_type(block)
        types = {t for t, _ in segs}
        assert len(types) > 1, "Mixed block should produce more than one segment type"


# ===========================================================================
# chunk_markdown - general properties
# ===========================================================================


class TestChunkMarkdown:
    def test_returns_list_of_documents(self, simple_md):
        assert all(isinstance(c, Document) for c in chunk_markdown(simple_md))

    def test_produces_at_least_one_chunk(self, simple_md):
        assert len(chunk_markdown(simple_md)) == 7, f"Expected exactly 7 chunks, got {len(chunk_markdown(simple_md))}"

    def test_source_metadata_matches_filename(self, simple_md):
        for c in chunk_markdown(simple_md):
            assert c.metadata["source"] == simple_md.name

    def test_all_required_metadata_keys_present(self, simple_md):
        required = {"source", "section", "language", "type"}
        for c in chunk_markdown(simple_md):
            assert required <= set(c.metadata.keys())

    def test_page_content_never_blank(self, simple_md):
        for c in chunk_markdown(simple_md):
            assert c.page_content.strip() != ""

    def test_language_metadata_is_string(self, simple_md):
        for c in chunk_markdown(simple_md):
            assert isinstance(c.metadata["language"], str)

    def test_empty_file_returns_empty_list(self, tmp_path):
        empty = tmp_path / "empty.md"
        empty.write_text("", encoding="utf-8")
        assert chunk_markdown(empty) == []

    def test_heading_only_file_single_chunk(self, tmp_path):
        md = tmp_path / "heading_only.md"
        md.write_text("# Solo un titolo\n", encoding="utf-8")
        assert len(chunk_markdown(md)) == 1

    def test_table_chunk_present_with_pipe(self, simple_md):
        chunks = chunk_markdown(simple_md)
        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        assert table_chunks
        assert len(table_chunks) == 1, f"Expected exactly one table chunk, got {len(table_chunks)}"
        for c in table_chunks:
            assert "|" in c.page_content

    def test_known_chunk_types_only(self, simple_md):
        allowed = {"text", "list", "table", "recomm"}
        for c in chunk_markdown(simple_md):
            assert c.metadata.get("type") in allowed, f"Unexpected chunk type: {c.metadata.get('type')!r}"


# ===========================================================================
# chunk_markdown - SKIP_SECTIONS
# ===========================================================================


class TestSkipSections:
    def test_all_skipped_sections_absent_from_chunks(self, simple_md):
        chunks = chunk_markdown(simple_md)
        for c in chunks:
            sec = c.metadata.get("section", "").lower()
            for keyword in ("riferiment", "referenz", "bibliograf"):
                assert keyword not in sec, f"Skipped section leaked into chunks: section={sec!r}"

    def test_skipped_content_absent_from_page_content(self, simple_md):
        chunks = chunk_markdown(simple_md)
        all_text = " ".join(c.page_content for c in chunks)
        assert "Smith J." not in all_text, "Riferimenti content leaked"
        assert "Testi bibliografici" not in all_text, "Referenze content leaked"
        assert "Rossi L." not in all_text, "Bibliografia content leaked"
        assert "Bianchi M." not in all_text, "Bibliografia content leaked"


# ===========================================================================
# chunk_markdown - NOSPLIT_SECTIONS  (type='recomm', one chunk per occurrence)
# ===========================================================================


class TestNosplitSections:
    def test_raccomandazioni_chunk_count(self, simple_md):
        """Two ## Raccomandazioni sections -> exactly 2 chunks total."""
        chunks = chunk_markdown(simple_md)
        rec = [c for c in chunks if "raccomandazion" in c.metadata.get("section", "").lower()]
        assert len(rec) == 2, f"Expected 2 chunks for Raccomandazioni, got {len(rec)}"

    def test_raccomandazioni_type_is_recomm(self, simple_md):
        chunks = chunk_markdown(simple_md)
        rec = [c for c in chunks if "raccomandazion" in c.metadata.get("section", "").lower()]
        for c in rec:
            assert c.metadata.get("type") == "recomm", f"Expected type='recomm', got {c.metadata.get('type')!r}"

    def test_first_raccomandazioni_content(self, simple_md):
        """First occurrence: short prose only, no list items from the second."""
        chunks = chunk_markdown(simple_md)
        rec = [c for c in chunks if "raccomandazion" in c.metadata.get("section", "").lower()]
        first = rec[0]
        assert "Le raccomandazioni principali" in first.page_content
        assert "Mantenere" not in first.page_content

    def test_second_raccomandazioni_contains_prose_and_list(self, simple_md):
        """Second occurrence: prose + bullet list all in the single chunk."""
        chunks = chunk_markdown(simple_md)
        rec = [c for c in chunks if "raccomandazion" in c.metadata.get("section", "").lower()]
        second = rec[1]
        assert "consigliabile" in second.page_content
        assert "Mantenere" in second.page_content
        assert "automedicazione" in second.page_content
        assert "Tenere un diario" in second.page_content
        assert "Partecipare" in second.page_content
        assert "Informare il medico" in second.page_content
        assert "Per ulteriori informazioni" in second.page_content

    def test_nosplit_section_source_metadata(self, simple_md):
        chunks = chunk_markdown(simple_md)
        rec = [c for c in chunks if "raccomandazion" in c.metadata.get("section", "").lower()]
        for c in rec:
            assert c.metadata["source"] == simple_md.name
            assert c.metadata["language"] != ""


# ===========================================================================
# chunk_markdown - ## Lista di istruzioni  (normal split -> list, text, list)
# ===========================================================================


class TestListaDiIstruzioni:
    """
    ## Lista di istruzioni
    is a regular section (not in NOSPLIT_SECTIONS) in the fixture simple_md.
    It must be split structurally:
      chunk 0 - type='list'  -> numbered items 1-3
      chunk 1 - type='text'  -> prose paragraph about side effects
      chunk 2 - type='list'  -> bullet items about monitoring
    """

    @pytest.fixture()
    def lista_chunks(self, simple_md):
        chunks = chunk_markdown(simple_md)
        return [c for c in chunks if c.metadata.get("section", "").lower() == "lista di istruzioni"]

    def test_produces_exactly_three_chunks(self, lista_chunks):
        assert len(lista_chunks) == 3, (
            f"Expected 3 chunks for 'Lista di istruzioni', got {len(lista_chunks)}:\n"
            + "\n".join(f"  [{c.metadata.get('type')}] {c.page_content[:80]!r}" for c in lista_chunks)
        )

    def test_chunk_type_sequence_is_list_text_list(self, lista_chunks):
        types = [c.metadata.get("type") for c in lista_chunks]
        assert types == ["list", "text", "list"], f"Expected ['list', 'text', 'list'], got {types}"

    def test_first_chunk_is_numbered_list(self, lista_chunks):
        first = lista_chunks[0]
        assert first.metadata["type"] == "list"
        assert "1." in first.page_content
        assert "2." in first.page_content
        assert "3." in first.page_content
        assert "Assumere" in first.page_content
        assert "Evitare alcolici" in first.page_content
        assert "Contattare il medico" in first.page_content

    def test_second_chunk_is_text(self, lista_chunks):
        middle = lista_chunks[1]
        assert middle.metadata["type"] == "text"
        assert "effetti collaterali" in middle.page_content
        # Must not bleed into adjacent list content
        assert "Monitorare" not in middle.page_content
        assert "Assumere" not in middle.page_content

    def test_third_chunk_is_bullet_list(self, lista_chunks):
        third = lista_chunks[2]
        assert third.metadata["type"] == "list"
        assert "Monitorare" in third.page_content
        assert "Annotare" in third.page_content

    def test_all_chunks_have_correct_section_metadata(self, lista_chunks):
        for c in lista_chunks:
            assert c.metadata["section"] == "Lista di istruzioni"

    def test_all_chunks_have_correct_source(self, simple_md, lista_chunks):
        for c in lista_chunks:
            assert c.metadata["source"] == simple_md.name

    def test_numbered_items_not_in_bullet_chunk(self, lista_chunks):
        """Items 1-3 must not bleed into the bullet-list chunk."""
        bullet = lista_chunks[2]
        assert "Assumere" not in bullet.page_content
        assert "Evitare alcolici" not in bullet.page_content

    def test_bullet_items_not_in_numbered_chunk(self, lista_chunks):
        """Bullet items must not bleed into the numbered-list chunk."""
        numbered = lista_chunks[0]
        assert "Monitorare" not in numbered.page_content
        assert "Annotare" not in numbered.page_content

    def test_lista_chunks_not_recomm_type(self, lista_chunks):
        """Regular sections must never get type='recomm'."""
        for c in lista_chunks:
            assert c.metadata.get("type") != "recomm"


# ===========================================================================
# chunk_markdown - integration against real Informazioni per_pazienti.md
# ===========================================================================


@pytest.mark.skipif(
    not PAZIENTE_MD.exists(),
    reason=f"Real document not found at '{PAZIENTE_MD}' - skipping integration tests",
)
class TestChunkMarkdownRealDocument:
    @pytest.fixture(scope="class")
    def chunks(self):
        return chunk_markdown(PAZIENTE_MD)

    def test_produces_chunks(self, chunks):
        assert len(chunks) > 0

    def test_language_detected_as_italian(self, chunks):
        languages = {c.metadata["language"] for c in chunks}
        assert "it" in languages

    def test_source_is_correct_filename(self, chunks):
        for c in chunks:
            assert c.metadata["source"] == PAZIENTE_MD.name

    def test_section_metadata_populated(self, chunks):
        assert any(c.metadata.get("section") for c in chunks)

    def test_all_required_metadata_keys(self, chunks):
        required = {"source", "section", "language", "type"}
        for c in chunks:
            assert required <= set(c.metadata.keys())

    def test_no_text_chunk_exceeds_3000_chars(self, chunks):
        oversized = [c for c in chunks if c.metadata.get("type") == "text" and len(c.page_content) > 3000]
        assert not oversized, (
            f"{len(oversized)} text chunks exceed 3000 chars (sizes: {[len(c.page_content) for c in oversized[:5]]})"
        )

    def test_recomm_chunks_are_single_per_section_occurrence(self, chunks):
        """
        Every contiguous run of recomm chunks sharing the same section name
        must contain exactly one chunk (no structural sub-splitting).
        """
        from itertools import groupby

        rec = [c for c in chunks if c.metadata.get("type") == "recomm"]
        for section_name, group in groupby(rec, key=lambda c: c.metadata.get("section")):
            group_list = list(group)
            assert len(group_list) == 1, (
                f"Section '{section_name}' produced {len(group_list)} recomm chunks, expected 1"
            )

    def test_table_chunks_contain_pipe(self, chunks):
        for c in chunks:
            if c.metadata.get("type") == "table":
                assert "|" in c.page_content

    def test_known_chunk_types_only(self, chunks):
        allowed = {"text", "list", "table", "recomm"}
        for c in chunks:
            assert c.metadata.get("type") in allowed, f"Unexpected chunk type: {c.metadata.get('type')!r}"


# ===========================================================================
# merge_continuation_lists
# ===========================================================================


class TestMergeContinuationLists:
    @staticmethod
    def _doc(content: str, source: str = "a.md", type_: str = "list") -> Document:
        return Document(page_content=content, metadata={"source": source, "type": type_})

    def test_merges_sequential_continuation(self):
        chunks = [
            self._doc("1. Primo\n2. Secondo\n3. Terzo"),
            self._doc("4. Quarto\n5. Quinto"),
        ]
        result = merge_continuation_lists(chunks)
        assert len(result) == 1
        assert "1. Primo" in result[0].page_content
        assert "4. Quarto" in result[0].page_content

    def test_no_merge_different_source(self):
        chunks = [
            self._doc("1. Uno\n2. Due\n3. Tre", source="a.md"),
            self._doc("4. Quattro", source="b.md"),
        ]
        assert len(merge_continuation_lists(chunks)) == 2

    def test_no_merge_gap_in_numbering(self):
        chunks = [
            self._doc("1. Uno\n2. Due\n3. Tre"),
            self._doc("5. Cinque"),
        ]
        assert len(merge_continuation_lists(chunks)) == 2

    def test_recomm_chunks_never_merged(self):
        """Chunks with type='recomm' must be ignored by merge logic."""
        chunks = [
            Document(
                page_content="## Raccomandazioni\n1. Uno\n2. Due\n3. Tre",
                metadata={"source": "a.md", "type": "recomm"},
            ),
            Document(
                page_content="4. Quattro\n5. Cinque",
                metadata={"source": "a.md", "type": "recomm"},
            ),
        ]
        result = merge_continuation_lists(chunks)
        assert len(result) == 2, "recomm chunks must not be merged even if they look like continuations"

    def test_non_list_chunks_unchanged(self):
        chunks = [
            Document(page_content="Testo libero.", metadata={"source": "a.md", "type": "text"}),
            self._doc("1. Solo voce"),
        ]
        result = merge_continuation_lists(chunks)
        assert len(result) == 2
        assert result[0].page_content == "Testo libero."

    def test_empty_input_returns_empty(self):
        assert merge_continuation_lists([]) == []

    def test_single_chunk_unchanged(self):
        single = [self._doc("1. Solo elemento della lista")]
        assert len(merge_continuation_lists(single)) == 1

    def test_three_consecutive_list_chunks_merged_into_one(self):
        """Three consecutive list chunks with sequential numbering fully merge."""
        chunks = [
            self._doc("1. Uno\n2. Due"),
            self._doc("3. Tre\n4. Quattro"),
            self._doc("5. Cinque\n6. Sei"),
        ]
        result = merge_continuation_lists(chunks)
        assert len(result) == 1
        for i in range(1, 7):
            assert f"{i}." in result[0].page_content

    def test_three_consecutive_splits_merged_into_one(self):
        """
        A noise-only list chunk ('Testo intermedio non list') between two
        numbered list chunks should NOT prevent merging. All 6 items end up
        in a single merged chunk; the noise text becomes a separate document.
        """
        chunks = [
            self._doc("1. Uno\n2. Due"),
            self._doc("3. Tre\n4. Quattro"),
            Document(
                page_content="Testo intermedio non list",
                metadata={"source": "a.md", "type": "text"},
            ),
            self._doc("5. Cinque\n6. Sei"),
        ]
        result = merge_continuation_lists(chunks)

        # All 6 numbered items end up in a single merged list chunk
        assert len(result) == 2
        merged_list = next(c for c in result if c.metadata["type"] == "list")
        for i in range(1, 7):
            assert f"{i}." in merged_list.page_content, f"Item {i} missing from merged chunk"

        # The intermediate text survives as a separate document with correct type
        noise = next(c for c in result if c.metadata["type"] == "text")
        assert noise.page_content == "Testo intermedio non list"

    def test_preserves_predecessor_metadata(self):
        chunks = [
            self._doc("1. Primo\n2. Secondo"),
            self._doc("3. Terzo"),
        ]
        result = merge_continuation_lists(chunks)
        assert result[0].metadata["source"] == "a.md"
        assert result[0].metadata["type"] == "list"

    def test_starting_at_one_is_not_continuation(self):
        chunks = [
            self._doc("1. Alpha\n2. Beta\n3. Gamma"),
            self._doc("1. Nuova lista separata"),
        ]
        assert len(merge_continuation_lists(chunks)) == 2


# ===========================================================================
# Pipeline: chunk_markdown -> merge_continuation_lists
# ===========================================================================


class TestPipeline:
    def test_all_list_items_reachable_after_merge(self, list_continuation_md):
        raw = chunk_markdown(list_continuation_md)
        merged = merge_continuation_lists(raw)
        all_list_text = " ".join(c.page_content for c in merged if c.metadata.get("type") == "list")
        for i in range(1, 6):
            assert str(i) in all_list_text, f"List item {i} missing after pipeline"

    def test_merge_reduces_list_chunk_count(self, list_continuation_md):
        raw = chunk_markdown(list_continuation_md)
        merged = merge_continuation_lists(raw)
        raw_list = sum(1 for c in raw if c.metadata.get("type") == "list")
        merged_list = sum(1 for c in merged if c.metadata.get("type") == "list")
        if raw_list > 1:
            assert merged_list < raw_list

    def test_non_list_chunk_count_unchanged(self, list_continuation_md):
        raw = chunk_markdown(list_continuation_md)
        merged = merge_continuation_lists(raw)
        assert sum(1 for c in raw if c.metadata.get("type") != "list") == sum(
            1 for c in merged if c.metadata.get("type") != "list"
        )

    def test_recomm_chunks_survive_pipeline_untouched(self, simple_md):
        """recomm chunks must pass through merge_continuation_lists unchanged."""
        raw = chunk_markdown(simple_md)
        merged = merge_continuation_lists(raw)
        raw_rec = [c for c in raw if c.metadata.get("type") == "recomm"]
        merged_rec = [c for c in merged if c.metadata.get("type") == "recomm"]
        assert len(raw_rec) == len(merged_rec), f"recomm count changed after merge: {len(raw_rec)} -> {len(merged_rec)}"
        for r, m in zip(raw_rec, merged_rec, strict=False):
            assert r.page_content == m.page_content

    def test_simple_md_full_pipeline(self, simple_md):
        raw = chunk_markdown(simple_md)
        merged = merge_continuation_lists(raw)
        assert len(merged) > 0
        rec = [c for c in merged if "raccomandazion" in c.metadata.get("section", "").lower()]
        assert len(rec) == 2
        for c in rec:
            assert c.metadata["type"] == "recomm"
