import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from scripts import performance_index

pytestmark = pytest.mark.cpu


def test_index_is_compatible_requires_files_and_matching_manifest(tmp_path: Path):
    index_dir = tmp_path / "index"
    settings = performance_index.index_settings("org/model", 100, 20)

    assert not performance_index.index_is_compatible(index_dir, settings)

    index_dir.mkdir()
    (index_dir / "index.faiss").touch()
    (index_dir / "index.pkl").touch()
    (index_dir / performance_index.INDEX_MANIFEST).write_text(json.dumps(settings), encoding="utf-8")

    assert performance_index.index_is_compatible(index_dir, settings)
    assert not performance_index.index_is_compatible(
        index_dir,
        performance_index.index_settings("different/model", 100, 20),
    )


def test_load_markdown_chunks_reads_only_top_level_markdown(tmp_path: Path):
    (tmp_path / "first.md").write_text("First document with enough words to split into chunks. " * 4, encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Not Markdown", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.md").write_text("Nested Markdown", encoding="utf-8")

    chunks = performance_index.load_markdown_chunks(str(tmp_path), chunk_size=60, chunk_overlap=10)

    assert len(chunks) > 1
    assert {chunk.metadata["source"] for chunk in chunks} == {str(tmp_path / "first.md")}


def test_embedding_model_location_resolves_local_model(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert performance_index.embedding_model_location(str(model_dir)) == model_dir.resolve()


def test_embedding_model_location_resolves_cached_hub_snapshot(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "hub"
    snapshot_dir = cache_dir / "models--org--model" / "snapshots" / "revision"
    snapshot_dir.mkdir(parents=True)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    location = performance_index.embedding_model_location("org/model", str(cache_dir))

    assert location == snapshot_dir.resolve()
    assert calls == [{"repo_id": "org/model", "cache_dir": str(cache_dir), "local_files_only": True}]


def test_get_or_create_vectorstore_builds_independent_index(monkeypatch, tmp_path: Path):
    index_dir = tmp_path / "performance-index"
    chunks = [Document(page_content="benchmark content", metadata={"source": "source.md"})]
    embedder = MagicMock()
    vectorstore = MagicMock()
    from_documents = MagicMock(return_value=vectorstore)
    monkeypatch.setattr(performance_index, "load_markdown_chunks", lambda *_args: chunks)
    monkeypatch.setattr(performance_index.FAISS, "from_documents", from_documents)

    result = performance_index.get_or_create_vectorstore(
        str(index_dir),
        str(tmp_path / "markdown"),
        "org/model",
        embedder,
        100,
        20,
    )

    assert result is vectorstore
    from_documents.assert_called_once_with(chunks, embedder)
    vectorstore.save_local.assert_called_once_with(str(index_dir.resolve()))
    manifest = json.loads((index_dir / performance_index.INDEX_MANIFEST).read_text(encoding="utf-8"))
    assert manifest == performance_index.index_settings("org/model", 100, 20)


def test_get_or_create_vectorstore_loads_compatible_index(monkeypatch, tmp_path: Path):
    index_dir = tmp_path / "performance-index"
    index_dir.mkdir()
    settings = performance_index.index_settings("org/model", 100, 20)
    (index_dir / "index.faiss").touch()
    (index_dir / "index.pkl").touch()
    (index_dir / performance_index.INDEX_MANIFEST).write_text(json.dumps(settings), encoding="utf-8")
    embedder = MagicMock()
    vectorstore = MagicMock()
    load_local = MagicMock(return_value=vectorstore)
    monkeypatch.setattr(performance_index.FAISS, "load_local", load_local)

    result = performance_index.get_or_create_vectorstore(
        str(index_dir),
        str(tmp_path / "markdown"),
        "org/model",
        embedder,
        100,
        20,
    )

    assert result is vectorstore
    load_local.assert_called_once_with(str(index_dir.resolve()), embedder, allow_dangerous_deserialization=True)
