import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

import agentic_rag.dizionario as dizionario
from agentic_rag.dizionario import SynonymStore, load_synonym_groups_from_excel

pytestmark = pytest.mark.cpu


def test_excel_groups_are_split_deduplicated_and_merged_transitively(monkeypatch, tmp_path):
    """Verify workbook rows become deterministic, transitively merged synonym groups."""
    workbook = tmp_path / "dictionary.xlsx"
    frame = pd.DataFrame(
        [
            {"term": " Colon TC / CTC ", "synonyms": "colonografia, TC colon, ctc"},
            {"term": "colon virtuale", "synonyms": "esame virtuale"},
            {"term": "tc colon", "synonyms": "COLON VIRTUALE"},
            {"term": pd.NA, "synonyms": pd.NA},
        ]
    )
    read_excel = MagicMock(return_value=frame)
    monkeypatch.setattr(dizionario.pd, "read_excel", read_excel)

    groups = load_synonym_groups_from_excel(workbook)

    assert groups == (
        (
            "Colon TC",
            "CTC",
            "colonografia",
            "TC colon",
            "COLON VIRTUALE",
            "esame virtuale",
        ),
    )
    read_excel.assert_called_once_with(workbook, header=1, names=["term", "synonyms"])


def test_synonym_store_matches_boundaries_and_reloads_changed_files(monkeypatch, tmp_path):
    """Verify longest-term matching, defensive copies, caching, and mtime reloads."""
    workbook = tmp_path / "dictionary.xlsx"
    workbook.touch()
    initial_groups = (
        ("TC", "Colon TC", "colonografia"),
        ("prep", "preparazione"),
    )
    changed_groups = (("nuovo", "termine"),)
    loader = MagicMock(side_effect=[initial_groups, changed_groups])
    monkeypatch.setattr(dizionario, "load_synonym_groups_from_excel", loader)

    store = SynonymStore(workbook)

    assert store.find_matched_terms("La COLON   TC richiede preparazione") == {
        "Colon TC": ["colonografia", "TC"],
        "preparazione": ["prep"],
    }
    assert store.find_matched_terms("CTC") == {}
    assert store.collect_expansion_terms("Colon TC e preparazione") == [
        "colonografia",
        "Colon TC",
        "TC",
        "preparazione",
        "prep",
    ]
    assert store.find_matching_groups("Colon TC") == [["colonografia", "Colon TC", "TC"]]
    assert store._get_groups_ref() is initial_groups
    groups_copy = store.get_groups()
    groups_copy[0].append("mutazione")
    assert store.get_groups()[0] == list(initial_groups[0])
    loader.assert_called_once_with(workbook)

    changed_mtime = workbook.stat().st_mtime + 10
    os.utime(workbook, (changed_mtime, changed_mtime))

    assert store.get_groups() == [["nuovo", "termine"]]
    assert loader.call_count == 2


def test_synonym_helpers_handle_empty_values_and_preserve_first_spelling():
    """Verify low-level synonym helpers handle blanks and normalized duplicates."""
    assert dizionario.split_terms(pd.NA) == []
    assert dizionario.split_terms("  ") == []
    assert dizionario.split_terms("uno / due, tre") == ["uno", "due", "tre"]
    assert dizionario.dedup_preserve_order(["  Colon TC ", "colon   tc", "CTC", ""]) == [
        "Colon TC",
        "CTC",
    ]
    assert dizionario.merge_overlapping_groups([]) == []
