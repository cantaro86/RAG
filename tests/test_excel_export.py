import json
import stat

import pytest
from openpyxl import load_workbook

from evaluation.cli import main as evaluation_main
from evaluation.export_excel import export_run_to_excel
from evaluation.tracing import write_gzip_json

pytestmark = pytest.mark.evaluation


def _trace(question_id, line_number, question, *, completed=True):
    if not completed:
        return {
            "schema_version": 1,
            "question_id": question_id,
            "question_line": line_number,
            "raw_question": question,
            "status": "failed",
            "error": {"type": "RuntimeError", "message": "graph failed"},
            "documents": {},
            "steps": [
                {
                    "sequence": 0,
                    "checkpoint_step": -1,
                    "nodes": ["__input__"],
                    "next_nodes": ["sanitize_question"],
                    "state": {"question": question},
                    "tasks": [],
                }
            ],
            "diagnostics": {
                "node_path": [],
                "question_versions": [{"sequence": -1, "nodes": ["__input__"], "question": question}],
                "retrieval_attempts": [],
                "reformulation_count": 0,
                "maximum_rewrite_count": 0,
                "terminal_node": None,
                "next_nodes": ["sanitize_question"],
                "failed_nodes": [],
                "guardrail_status": None,
                "retrieval_succeeded": False,
                "final_has_docs": None,
            },
        }

    document_id = "doc-123"
    sanitized = "Domanda sanitizzata"
    transformed = "Domanda trasformata"
    return {
        "schema_version": 1,
        "question_id": question_id,
        "question_line": line_number,
        "raw_question": question,
        "status": "completed",
        "documents": {
            document_id: {
                "page_content": "Contesto finale",
                "metadata": {"source": "source.md", "rerank_score": 0.91},
            }
        },
        "steps": [
            {
                "sequence": 0,
                "checkpoint_step": -1,
                "nodes": ["__input__"],
                "next_nodes": ["sanitize_question"],
                "state": {"question": question},
                "tasks": [],
            },
            {
                "sequence": 1,
                "checkpoint_step": 0,
                "nodes": ["sanitize_question"],
                "next_nodes": ["guardrail"],
                "state": {"question": sanitized, "guardrail_status": None},
                "tasks": [],
            },
            {
                "sequence": 2,
                "checkpoint_step": 1,
                "nodes": ["guardrail"],
                "next_nodes": ["transform_query"],
                "state": {"question": sanitized, "guardrail_status": "ON_TOPIC"},
                "tasks": [],
            },
            {
                "sequence": 3,
                "checkpoint_step": 2,
                "nodes": ["transform_query"],
                "next_nodes": ["retrieve_and_filter"],
                "state": {"question": transformed, "guardrail_status": "ON_TOPIC"},
                "tasks": [],
            },
            {
                "sequence": 4,
                "checkpoint_step": 3,
                "nodes": ["retrieve_and_filter"],
                "next_nodes": ["generate_with_docs"],
                "state": {
                    "question": transformed,
                    "guardrail_status": "ON_TOPIC",
                    "documents": [{"$document": document_id}],
                    "has_docs": True,
                    "rewrite_count": 1,
                },
                "tasks": [],
            },
        ],
        "diagnostics": {
            "node_path": ["sanitize_question", "guardrail", "transform_query", "retrieve_and_filter"],
            "question_versions": [
                {"sequence": -1, "nodes": ["__input__"], "question": question},
                {"sequence": 1, "nodes": ["sanitize_question"], "question": sanitized},
                {"sequence": 3, "nodes": ["transform_query"], "question": transformed},
            ],
            "retrieval_attempts": [
                {
                    "attempt": 1,
                    "sequence": 4,
                    "question": transformed,
                    "has_docs": True,
                    "rewrite_count": 1,
                    "source_filter": None,
                    "documents": [document_id],
                }
            ],
            "reformulation_count": 2,
            "maximum_rewrite_count": 1,
            "terminal_node": "retrieve_and_filter",
            "next_nodes": ["generate_with_docs"],
            "failed_nodes": [],
            "guardrail_status": "ON_TOPIC",
            "retrieval_succeeded": True,
            "final_has_docs": True,
        },
    }


def _write_run(run_dir):
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True)
    completed_id = "q0001-abcd1234"
    failed_id = "q0002-efgh5678"
    write_gzip_json(traces_dir / f"{completed_id}.json.gz", _trace(completed_id, 1, "Domanda originale"))
    write_gzip_json(
        traces_dir / f"{failed_id}.json.gz",
        _trace(failed_id, 2, "Domanda fallita", completed=False),
    )
    sample = {
        "schema_version": 1,
        "id": completed_id,
        "user_input": "Domanda originale",
        "response": "Risposta finale",
        "retrieved_contexts": ["Contesto finale"],
        "trace_path": f"traces/{completed_id}.json.gz",
    }
    score = {
        "id": completed_id,
        "metrics": {"faithfulness": 0.8, "context_utilization": 0.7, "answer_relevancy": 0.9},
        "errors": {},
        "skipped": {},
    }
    (run_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(json.dumps(score) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "test-run", "status": "scored", "question_count": 2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"collection": {"questions": 2, "completed": 1, "failed": 1}}) + "\n",
        encoding="utf-8",
    )


def test_excel_export_combines_artifacts_and_keeps_failed_questions(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    output_path = export_run_to_excel(run_dir)

    assert output_path == run_dir / "evaluation_results.xlsx"
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    workbook = load_workbook(output_path, read_only=False, data_only=True)
    try:
        assert workbook.sheetnames == ["Results", "Run Summary", "Legend"]
        worksheet = workbook["Results"]
        headers = [cell.value for cell in worksheet[1]]
        completed = dict(zip(headers, (cell.value for cell in worksheet[2]), strict=True))
        failed = dict(zip(headers, (cell.value for cell in worksheet[3]), strict=True))

        assert worksheet.freeze_panes == "A2"
        assert worksheet.tables["EvaluationResults"] is not None
        assert len(headers) == 18
        assert {
            "question_id",
            "scoring_status",
            "classification_raw",
            "question_versions",
            "contexts_truncated",
            "scoring_errors",
            "reformulation_count",
            "maximum_rewrite_count",
            "terminal_node",
            "collection_error",
            "trace_path",
        }.isdisjoint(headers)
        assert completed["question_line"] == 1
        assert completed["classification"] == "on_topic"
        assert completed["sanitized_question"] == "Domanda sanitizzata"
        assert completed["transformed_question"] == "Domanda trasformata"
        assert completed["final_response"] == "Risposta finale"
        assert completed["retrieved_contexts"] == "[Context 1]\nContesto finale"
        assert completed["retrieved_sources"] == "[1] source.md"
        assert completed["faithfulness"] == 0.8
        assert completed["context_utilization"] == 0.7
        assert completed["answer_relevancy"] == 0.9
        assert failed["question_line"] == 2
        assert failed["collection_status"] == "failed"
        assert failed["final_response"] is None
    finally:
        workbook.close()


def test_excel_export_requires_overwrite_for_existing_report(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    assert evaluation_main(["export-excel", str(run_dir)]) == 0
    output_path = run_dir / "evaluation_results.xlsx"

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        export_run_to_excel(run_dir)

    assert export_run_to_excel(run_dir, overwrite=True) == output_path


def test_excel_export_supports_collected_run_without_scores_or_summary(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "scores.jsonl").unlink()
    (run_dir / "summary.json").unlink()

    output_path = export_run_to_excel(run_dir, tmp_path / "collected.xlsx")

    workbook = load_workbook(output_path, read_only=True, data_only=True)
    try:
        rows = workbook["Results"].iter_rows(values_only=True)
        headers = next(rows)
        completed = dict(zip(headers, next(rows), strict=True))
        assert completed["faithfulness"] is None
        assert completed["context_utilization"] is None
        assert completed["answer_relevancy"] is None
    finally:
        workbook.close()
