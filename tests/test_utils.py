from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_rag.utils import PATIENT_DOC, extract_source_filter, render_context

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "question",
    [
        "Show information for patients",
        "Use the patient_information corpus",
        "Mostra Informazioni_per_pazienti.md",
        "Usa le informazioni per pazienti",
        "Usa le informazioni per i pazienti",
        "Leggi l'informativa per i pazienti",
    ],
)
def test_extract_source_filter_recognizes_explicit_patient_corpus_aliases(question):
    """Verify supported English and Italian aliases select the patient corpus."""
    assert extract_source_filter(question) == {"source": PATIENT_DOC}


@pytest.mark.parametrize(
    "question",
    [
        "Quali informazioni servono ai pazienti?",
        "Queste sono disinformazioni per pazienti",
        "",
    ],
)
def test_extract_source_filter_avoids_incidental_or_embedded_phrases(question):
    """Verify incidental patient wording does not activate corpus filtering."""
    assert extract_source_filter(question) is None


def test_render_context_orders_corpora_and_recommendations_with_safe_defaults(tmp_path):
    """Verify context ordering, normalized sections, basenames, and metadata defaults."""
    scientific = Document(
        page_content="  studio ordinario  ",
        metadata={"source": str(tmp_path / "studio.md"), "section": "Discussione", "type": "text"},
    )
    metadata_free = Document(page_content="  senza metadati  ", metadata={})
    patient = Document(
        page_content="istruzioni ordinarie",
        metadata={"source": str(tmp_path / PATIENT_DOC), "section": "Preparazione", "type": "list"},
    )
    scientific_recommendation = Document(
        page_content="raccomandazione scientifica",
        metadata={"source": str(tmp_path / "linee_guida.md"), "section": "Raccomandazioni principali:"},
    )
    patient_recommendation = Document(
        page_content="raccomandazione paziente",
        metadata={"source": str(tmp_path / PATIENT_DOC), "section": "Raccomandazióni"},
    )
    documents = [scientific, metadata_free, patient, scientific_recommendation, patient_recommendation]

    rendered = render_context(documents)

    excerpts = [line.removeprefix("Excerpt: ") for line in rendered.splitlines() if line.startswith("Excerpt: ")]
    assert excerpts == [
        "raccomandazione paziente",
        "istruzioni ordinarie",
        "raccomandazione scientifica",
        "studio ordinario",
        "senza metadati",
    ]
    assert rendered.count("Corpus: Informazioni per i pazienti") == 2
    assert rendered.count("Corpus: Letteratura scientifica") == 3
    assert f"Source: {Path(PATIENT_DOC).name}" in rendered
    assert "Section: main\nSource: unknown\nType: text\nExcerpt: senza metadati" in rendered
    assert [document.page_content for document in documents] == [
        "  studio ordinario  ",
        "  senza metadati  ",
        "istruzioni ordinarie",
        "raccomandazione scientifica",
        "raccomandazione paziente",
    ]
