# tests/conftest.py
import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agentic_rag.config_schema import Config, load_config
from agentic_rag.graph import RAGContext

_ORIGINAL_LOGGING_DISABLE_LEVEL = logging.root.manager.disable
logging.disable(logging.CRITICAL)


def pytest_unconfigure(config) -> None:
    """Restore the process logging state after pytest finishes."""
    logging.disable(_ORIGINAL_LOGGING_DISABLE_LEVEL)


@pytest.fixture
def enabled_test_logging():
    """Temporarily enable logging for tests that verify handler output."""
    previous_level = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        logging.disable(previous_level)


# ---------------------------------------------------------------------------
# Fixtures config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the absolute path to config.yaml."""
    return project_root / "config.yaml"


@pytest.fixture(scope="session")
def config_data(config_path: Path) -> Config:
    return load_config(config_path)


# ---------------------------------------------------------------------------
# Fixtures build_Faiss
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_md(tmp_path: Path) -> Path:
    """
    Minimal Italian Markdown file covering the main structural elements:
    plain text, a numbered list inside a 'Raccomandazioni' section (no-split),
    a 'Riferimenti' section (skipped by SKIP_SECTIONS), and a Markdown table.
    """
    content = textwrap.dedent("""
        # Informazioni generali

        Questo documento descrive le informazioni generali per il paziente.
        Il medico curante ha fornito le seguenti indicazioni terapeutiche da seguire.

        ## Raccomandazioni

        Le raccomandazioni principali sono elencate nella sezione successiva.
        Si prega di seguire attentamente le istruzioni fornite di seguito.


        ## Lista di istruzioni

        Seguire le istruzioni riportate di seguito per garantire un corretto utilizzo del farmaco:

        1. Assumere il farmaco ogni mattina a digiuno senza eccezioni.
        2. Evitare alcolici durante l'intero periodo della terapia farmacologica.
        3. Contattare il medico immediatamente in caso di reazioni avverse.

        Se si verificano effetti collaterali seguire la procedura indicata di seguito.

        - Monitorare attentamente eventuali sintomi insoliti.
        - Annotare eventuali cambiamenti nel proprio stato di salute.


        ## Riferimenti

        - Smith J. et al., 2020. Studio clinico randomizzato controllato.

        ## Tabella dosaggi

        | Farmaco | Dose | Frequenza |
        |---------|------|-----------|
        | Aspirina | 100 mg | 1x/die |
        | Ibuprofene | 400 mg | 3x/die |


        ## Raccomandazioni

        È consigliabile seguire le raccomandazioni del medico curante per garantire l'efficacia della terapia
        e ridurre al minimo i rischi di effetti collaterali.
        Inoltre aggiungo alcune informazioni generali per il paziente, come la gestione degli effetti collaterali
        e le precauzioni da adottare durante il trattamento.
        Ad esempio, è importante monitorare eventuali sintomi insoliti e riferirli tempestivamente al medico.
        Qui una lista di raccomandazioni aggiuntive:
        - Mantenere uno stile di vita sano, con una dieta equilibrata e attività fisica regolare.
        - Evitare l'automedicazione e seguire scrupolosamente le indicazioni del medico.
        - Tenere un diario dei sintomi e degli effetti collaterali per facilitare la comunicazione con il medico.
        - Partecipare a eventuali programmi di supporto o gruppi di pazienti per condividere esperienze
        e ricevere consigli utili.
        - Informare il medico di eventuali cambiamenti nello stato di salute o nell'assunzione di altri farmaci.

        Per ulteriori informazioni, consultare le fonti ufficiali e le linee guida fornite dal medico curante.


        ## Referenze
        Testi bibliografici a caso

        ## Bibliografia
        1. Rossi L., 2019. Manuale di farmacologia clinica.
        2. Bianchi M., 2021. Linee guida per la terapia farmacologica.
    """).strip()
    md_file = tmp_path / "test_doc.md"
    md_file.write_text(content, encoding="utf-8")
    return md_file


@pytest.fixture()
def list_continuation_md(tmp_path: Path) -> Path:
    """
    Markdown whose numbered list is intentionally split across two separate
    paragraphs, triggering the merge_continuation_lists logic.
    """
    content = textwrap.dedent("""
        # Istruzioni operative

        Seguire attentamente i seguenti passi per completare la procedura:

        1. Primo passo: registrarsi al portale con credenziali valide.
        2. Secondo passo: completare il profilo inserendo tutti i dati richiesti.
        3. Terzo passo: inserire i dati clinici richiesti dal modulo online.

        Testo intermedio che separa visivamente i due blocchi della stessa lista.

        4. Quarto passo: verificare accuratamente i dati inseriti nel sistema.
        5. Quinto passo: inviare la richiesta e attendere conferma via email.
    """).strip()
    md_file = tmp_path / "list_continuation.md"
    md_file.write_text(content, encoding="utf-8")
    return md_file


# ---------------------------------------------------------------------------
# Fixtures general
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vectorstore():
    return MagicMock()


@pytest.fixture
def dummy_config():
    cfg = MagicMock()
    cfg.hf_home = "/tmp/hf_home"
    cfg.chat = False
    cfg.gradio = False
    return cfg


# ---------------------------------------------------------------------------
# Fixtures RAGContext
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=RAGContext)

    # Mock all chains to return basic mock values
    ctx.guardrail_chain = MagicMock()
    ctx.sanitizer_chain = MagicMock()
    ctx.sanitizer_chain.invoke.return_value = "Sanitized query"
    ctx.topic_continuity_classifier = MagicMock()
    ctx.topic_continuity_classifier.invoke.return_value = "SAME"
    ctx.pre_retrieval_question_rewriter = MagicMock()
    ctx.pre_retrieval_question_rewriter.invoke.return_value = "Rewritten query"
    ctx.question_transformer = MagicMock()
    ctx.question_transformer.invoke.return_value = "Transformed query"
    ctx.rag_chain = MagicMock()
    ctx.rag_chain.invoke.return_value = "Mocked RAG response"
    ctx.cleaner_chain = MagicMock()
    ctx.cleaner_chain.invoke.return_value = "Cleaned mocked RAG response"

    # Mock retriever
    ctx.retriever = MagicMock()
    mock_doc = Document(
        page_content="Istruzioni per pazienti esame Colon-TC.",
        metadata={"rerank_score": 0.9, "source": "Informazioni_per_pazienti.md"},
    )
    ctx.retriever.invoke.return_value = [mock_doc]

    # Mock synonyms
    ctx.synonyms = MagicMock()
    ctx.synonyms.find_matched_terms.return_value = []

    return ctx
