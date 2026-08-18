import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu


def test_helpers_import_without_docling(monkeypatch):
    sys.modules.pop("scripts.build_markdown", None)
    real_import = builtins.__import__

    def reject_docling(name, *args, **kwargs):
        if name == "docling" or name.startswith("docling."):
            raise AssertionError("Docling must not be imported while loading pure helpers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_docling)
    module = importlib.import_module("scripts.build_markdown")

    assert module.clean_pdf_artifacts("A &amp; B") == "A & B"


def test_missing_docling_error_is_actionable(monkeypatch):
    module = importlib.import_module("scripts.build_markdown")
    real_import = builtins.__import__

    def reject_docling(name, *args, **kwargs):
        if name == "docling" or name.startswith("docling."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_docling)

    with pytest.raises(ImportError, match=r"uv sync --locked --extra markdown"):
        module._load_docling()


def test_conversion_validates_empty_input_before_loading_docling(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.build_markdown")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "nested" / "output"
    monkeypatch.setattr(module, "_load_docling", lambda: pytest.fail("Docling should not be loaded"))

    with pytest.raises(ValueError, match="No PDF or DOCX files"):
        module.create_markdown_from_pdf(input_dir, output_dir)

    assert not output_dir.exists()


def test_relative_input_prefers_input_directory_over_cwd(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.build_markdown")
    input_dir = tmp_path / "input"
    cwd = tmp_path / "cwd"
    input_dir.mkdir()
    cwd.mkdir()
    input_file = input_dir / "same.pdf"
    input_file.write_bytes(b"input")
    (cwd / "same.pdf").write_bytes(b"cwd")
    monkeypatch.chdir(cwd)

    assert module._normalize_input_files(input_dir, ["same.pdf"]) == [input_file]


@pytest.mark.parametrize(("device", "expected_device"), [(None, "auto"), ("cpu", "cpu")])
def test_conversion_normalizes_files_and_creates_nested_output(tmp_path, monkeypatch, device, expected_device):
    module = importlib.import_module("scripts.build_markdown")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf_file = input_dir / "a.PDF"
    docx_file = input_dir / "b.docx"
    pdf_file.write_bytes(b"pdf")
    docx_file.write_bytes(b"docx")
    output_dir = tmp_path / "nested" / "markdown"
    captured = {"converted": []}

    class FakePipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs
            self.table_structure_options = SimpleNamespace()

    class FakeAcceleratorOptions:
        def __init__(self, **kwargs):
            captured["accelerator_kwargs"] = kwargs

    class FakeConverter:
        def __init__(self, **kwargs):
            captured["converter_kwargs"] = kwargs

        def convert(self, path):
            captured["converted"].append(path)
            document = SimpleNamespace(export_to_markdown=lambda: "A &amp; B")
            return SimpleNamespace(document=document)

    fake_docling = SimpleNamespace(
        AcceleratorDevice=SimpleNamespace(AUTO="auto"),
        AcceleratorOptions=FakeAcceleratorOptions,
        DocumentConverter=FakeConverter,
        InputFormat=SimpleNamespace(PDF="pdf"),
        PdfFormatOption=lambda **kwargs: kwargs,
        PdfPipelineOptions=FakePipelineOptions,
        TableFormerMode=SimpleNamespace(ACCURATE="accurate"),
    )
    monkeypatch.setattr(module, "_load_docling", lambda: fake_docling)
    monkeypatch.setattr(module, "tqdm", lambda iterable, **kwargs: iterable)

    files = (path for path in (str(docx_file), Path(pdf_file)))
    module.create_markdown_from_pdf(input_dir, output_dir, files=files, device=device)

    assert output_dir.is_dir()
    assert (output_dir / "a.md").read_text(encoding="utf-8") == "A & B"
    assert (output_dir / "b.md").read_text(encoding="utf-8") == "A & B"
    assert all(isinstance(path, Path) for path in captured["converted"])
    assert captured["accelerator_kwargs"] == {"num_threads": 4, "device": expected_device}


def test_cli_forwards_selected_files_and_device(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.build_markdown")
    converter = MagicMock()
    monkeypatch.setattr(module, "create_markdown_from_pdf", converter)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    module.main(
        [
            str(input_dir),
            str(output_dir),
            "--files",
            "guide.pdf",
            "instructions.docx",
            "--device",
            "cpu",
        ]
    )

    converter.assert_called_once_with(
        input_dir,
        output_dir,
        files=["guide.pdf", "instructions.docx"],
        device="cpu",
    )
