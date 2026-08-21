import json
import stat
import sys
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock

import pytest
import yaml
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agentic_rag.graph import build_agent_graph
from evaluation.cli import _require_accelerator
from evaluation.cli import main as evaluation_main
from evaluation.ragas_scoring import LocalStructuredJudge, build_ragas_judge, score_dataset
from evaluation.tracing import (
    EvaluationConfig,
    EvaluationQuestion,
    collect_dataset,
    collect_question,
    load_evaluation_config,
    load_jsonl,
    load_questions,
    read_gzip_json,
    restore_trace_state,
    serialize_snapshots,
)

pytestmark = pytest.mark.evaluation


def evaluation_config_data() -> dict:
    return {
        "judge": {
            "model": "judge/model",
            "online": False,
            "quantization": False,
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_retries": 1,
        },
        "embeddings": {
            "model": "embedding/model",
            "online": False,
            "normalize_embeddings": True,
            "batch_size": 8,
        },
        "metrics": ["faithfulness", "context_utilization", "answer_relevancy"],
    }


def snapshot(step, state, next_nodes, *, source="loop"):
    return SimpleNamespace(
        metadata={"source": source, "step": step, "parents": {}},
        values=state,
        next=tuple(next_nodes),
        config={"configurable": {"thread_id": "ragas-q0001", "checkpoint_id": f"checkpoint-{step}"}},
        created_at=f"2026-01-01T00:00:{step + 1:02d}+00:00",
    )


def successful_history(document: Document) -> list[SimpleNamespace]:
    chronological = [
        snapshot(-1, {}, ["__start__"], source="input"),
        snapshot(0, {"question": "Domanda originale"}, ["sanitize_question"]),
        snapshot(
            1,
            {
                "question": "Domanda sanitizzata",
                "original_question": "Domanda sanitizzata",
                "documents": [],
                "rewrite_count": 0,
                "has_docs": False,
            },
            ["retrieve_and_filter"],
        ),
        snapshot(
            2,
            {
                "question": "Domanda sanitizzata",
                "documents": [],
                "rewrite_count": 1,
                "has_docs": False,
            },
            ["transform_query"],
        ),
        snapshot(
            3,
            {
                "question": "Domanda trasformata",
                "documents": [],
                "rewrite_count": 1,
                "has_docs": False,
            },
            ["retrieve_and_filter"],
        ),
        snapshot(
            4,
            {
                "question": "Domanda trasformata",
                "documents": [document],
                "rewrite_count": 0,
                "has_docs": True,
                "guardrail_status": "ON_TOPIC",
            },
            ["generate_with_docs"],
        ),
        snapshot(
            5,
            {
                "question": "Domanda trasformata",
                "documents": [document],
                "rewrite_count": 0,
                "has_docs": True,
                "guardrail_status": "ON_TOPIC",
                "generation": "Risposta finale",
            },
            [],
        ),
    ]
    return list(reversed(chronological))


class FakeAgent:
    def __init__(self, final_state, history, error=None):
        self.final_state = final_state
        self.history = history
        self.error = error
        self.checkpointer = SimpleNamespace(delete_thread=MagicMock())
        self.invocations = []

    def invoke(self, input_data, *, config):
        self.invocations.append((input_data, config))
        if self.error:
            raise self.error
        return self.final_state

    def get_state_history(self, config):
        return iter(self.history)


def test_load_questions_ignores_blanks_and_builds_stable_ids(tmp_path):
    questions_path = tmp_path / "questions.txt"
    questions_path.write_text("Prima domanda?\n\n  Seconda domanda?  \n", encoding="utf-8")

    questions = load_questions(questions_path)

    assert [question.line_number for question in questions] == [1, 3]
    assert [question.text for question in questions] == ["Prima domanda?", "Seconda domanda?"]
    assert questions[0].id.startswith("q0001-")
    assert questions == load_questions(questions_path)


def test_load_questions_rejects_duplicates(tmp_path):
    questions_path = tmp_path / "questions.txt"
    questions_path.write_text("Duplicata?\nDuplicata?\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate question at line 2"):
        load_questions(questions_path)


def test_load_evaluation_config_is_strict_and_deterministic(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(evaluation_config_data()), encoding="utf-8")

    config = load_evaluation_config(config_path)

    assert config.judge.temperature == 0.0
    invalid = evaluation_config_data()
    invalid["judge"]["temperature"] = 0.1
    config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature must be 0.0"):
        load_evaluation_config(config_path)


def test_validate_command_does_not_load_models(tmp_path, capsys):
    questions_path = tmp_path / "questions.txt"
    questions_path.write_text("Domanda?\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(evaluation_config_data()), encoding="utf-8")

    result = evaluation_main(["validate", "--questions", str(questions_path), "--evaluation-config", str(config_path)])

    assert result == 0
    assert "No application or evaluator model was loaded." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(False, True, "mps"), (True, False, "cuda"), (True, True, "mps")],
)
def test_evaluation_device_is_detected_automatically(monkeypatch, cuda_available, mps_available, expected):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _require_accelerator("score") == expected


def test_evaluation_device_rejects_cpu_only_nodes(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="requires a CUDA or MPS accelerator"):
        _require_accelerator("score")


def test_serialize_snapshots_is_chronological_and_deduplicates_documents():
    document = Document(page_content="Contesto", metadata={"source": "source.md", "rerank_score": 0.9})

    steps, documents = serialize_snapshots(successful_history(document))

    assert [step["checkpoint_step"] for step in steps] == [-1, 0, 1, 2, 3, 4, 5]
    assert steps[2]["nodes"] == ["sanitize_question"]
    assert len(documents) == 1
    restored = restore_trace_state(steps[-1]["state"], documents)
    assert restored["documents"] == [document]
    assert restored["generation"] == "Risposta finale"


def test_collect_question_records_reformulations_retrievals_and_cleanup():
    document = Document(page_content="Contesto", metadata={"source": "source.md", "rerank_score": 0.9})
    final_state = {"generation": "Risposta finale", "documents": [document], "has_docs": True}
    agent = FakeAgent(final_state, successful_history(document))
    question = EvaluationQuestion(id="q0001-abcd1234", line_number=1, text="Domanda originale")

    outcome = collect_question(agent, question, question_validator=lambda text: text)

    assert outcome.trace["status"] == "completed"
    assert outcome.sample["user_input"] == "Domanda originale"
    assert outcome.sample["retrieved_contexts"] == ["Contesto"]
    diagnostics = outcome.trace["diagnostics"]
    assert [item["question"] for item in diagnostics["question_versions"]] == [
        "Domanda originale",
        "Domanda sanitizzata",
        "Domanda trasformata",
    ]
    assert diagnostics["node_path"] == [
        "sanitize_question",
        "retrieve_and_filter",
        "transform_query",
        "retrieve_and_filter",
        "generate_with_docs",
    ]
    assert len(diagnostics["retrieval_attempts"]) == 2
    assert diagnostics["retrieval_succeeded"] is True
    assert diagnostics["maximum_rewrite_count"] == 1
    deleted_thread = agent.checkpointer.delete_thread.call_args.args[0]
    assert deleted_thread.startswith("ragas-q0001-abcd1234-")


def test_collect_question_keeps_partial_trace_after_failure():
    history = [
        snapshot(0, {"question": "Domanda originale", "guardrail_status": None}, ["guardrail"]),
        snapshot(-1, {}, ["__start__"], source="input"),
    ]
    agent = FakeAgent(None, history, error=RuntimeError("node failed"))
    question = EvaluationQuestion(id="q0001-abcd1234", line_number=1, text="Domanda originale")

    outcome = collect_question(agent, question)

    assert outcome.sample is None
    assert outcome.trace["status"] == "failed"
    assert outcome.trace["error"] == {"type": "RuntimeError", "message": "node failed"}
    assert len(outcome.trace["steps"]) == 2
    agent.checkpointer.delete_thread.assert_called_once()


def test_collect_dataset_writes_incremental_artifacts(tmp_path, capsys):
    document = Document(page_content="Contesto", metadata={"source": "source.md"})
    agent = FakeAgent(
        {"generation": "Risposta finale", "documents": [document], "has_docs": True},
        successful_history(document),
    )
    questions = [EvaluationQuestion(id="q0001-abcd1234", line_number=1, text="Domanda originale")]
    run_dir = tmp_path / "run"

    summary = collect_dataset(agent, questions, run_dir, run_metadata={"run_id": "test-run"})

    assert summary["collection"]["samples"] == 1
    samples = load_jsonl(run_dir / "samples.jsonl")
    assert samples[0]["trace_path"] == "traces/q0001-abcd1234.json.gz"
    trace = read_gzip_json(run_dir / samples[0]["trace_path"])
    assert trace["raw_question"] == "Domanda originale"
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "collected"
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((run_dir / "samples.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((run_dir / samples[0]["trace_path"]).stat().st_mode) == 0o600
    assert "Collecting questions" in capsys.readouterr().err
    with pytest.raises(FileExistsError):
        collect_dataset(agent, questions, run_dir)


class JudgeResponse(BaseModel):
    verdict: int


def test_local_structured_judge_extracts_json_and_retries():
    llm = MagicMock()
    llm.invoke.side_effect = [
        {"content": "not json"},
        {"content": 'Result:\n```json\n{"verdict": 1}\n```'},
    ]
    judge = LocalStructuredJudge(llm, max_retries=1)

    result = judge.generate("Evaluate", JudgeResponse)

    assert result == JudgeResponse(verdict=1)
    assert llm.invoke.call_count == 2
    retry_messages = llm.invoke.call_args_list[1].args[0]
    assert "failed validation" in retry_messages[-1]["content"]


def test_ragas_judge_implements_current_structured_interface():
    pytest.importorskip("ragas")
    from ragas.llms.base import InstructorBaseRagasLLM

    llm = MagicMock()
    llm.invoke.return_value = {"content": '{"verdict": 1}'}

    judge = build_ragas_judge(llm, max_retries=0)

    assert isinstance(judge, InstructorBaseRagasLLM)
    assert judge.generate("Evaluate", JudgeResponse) == JudgeResponse(verdict=1)


def test_collect_question_uses_real_graph_checkpoint_history(mock_ctx, config_data):
    document = Document(page_content="Contesto reale", metadata={"source": "source.md", "rerank_score": 0.9})
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"
    mock_ctx.retriever.invoke.side_effect = [[], [document]]
    agent = build_agent_graph(mock_ctx, config_data)
    question = EvaluationQuestion(id="q0001-abcd1234", line_number=1, text="Domanda originale")

    outcome = collect_question(agent, question)

    assert outcome.trace["status"] == "completed"
    assert [version["question"] for version in outcome.trace["diagnostics"]["question_versions"]] == [
        "Domanda originale",
        "Sanitized query",
        "Transformed query",
    ]
    assert outcome.trace["diagnostics"]["node_path"] == [
        "sanitize_question",
        "guardrail",
        "init_first_question",
        "retrieve_and_filter",
        "transform_query",
        "retrieve_and_filter",
        "generate_with_docs",
        "clean_answer",
    ]
    assert outcome.sample["response"] == "Mocked RAG response"
    assert outcome.sample["retrieved_contexts"] == ["Contesto reale"]


def test_collect_question_records_failed_langgraph_task():
    class State(TypedDict):
        question: str

    def fail(_state):
        raise RuntimeError("node failure")

    workflow = StateGraph(State)
    workflow.add_node("failing_node", fail)
    workflow.add_edge(START, "failing_node")
    workflow.add_edge("failing_node", END)
    agent = workflow.compile(checkpointer=InMemorySaver())
    question = EvaluationQuestion(id="q0001-abcd1234", line_number=1, text="Domanda originale")

    outcome = collect_question(agent, question)

    assert outcome.trace["status"] == "failed"
    failed_task = outcome.trace["steps"][-1]["tasks"][0]
    assert failed_task["name"] == "failing_node"
    assert failed_task["error"] == "RuntimeError('node failure')"
    assert outcome.trace["diagnostics"]["failed_nodes"] == ["failing_node"]


def test_score_dataset_uses_context_rules_and_updates_summary(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    samples = [
        {
            "id": "q1",
            "user_input": "Domanda",
            "response": "Risposta",
            "retrieved_contexts": ["Contesto"],
        },
        {
            "id": "q2",
            "user_input": "Domanda senza documenti",
            "response": "Nessun documento",
            "retrieved_contexts": [],
        },
    ]
    (run_dir / "samples.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    (run_dir / "summary.json").write_text('{"schema_version": 1, "collection": {}}\n', encoding="utf-8")
    (run_dir / "run.json").write_text('{"status": "collected"}\n', encoding="utf-8")
    config_data = evaluation_config_data()
    config_data["metrics"] = ["faithfulness", "answer_relevancy"]
    config = EvaluationConfig.model_validate(config_data)

    class Metric:
        def score(self, **_kwargs):
            return SimpleNamespace(value=0.75)

    monkeypatch.setattr("evaluation.ragas_scoring.build_ragas_judge", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "evaluation.ragas_scoring._build_metrics",
        lambda *_args, **_kwargs: {"faithfulness": Metric(), "answer_relevancy": Metric()},
    )

    summary = score_dataset(run_dir, config, MagicMock(), cache_folder="/tmp/cache", embedding_device="cpu")

    assert summary["scoring"]["metrics"]["faithfulness"] == {
        "mean": 0.75,
        "scored": 1,
        "errors": 0,
        "skipped": 1,
    }
    assert summary["scoring"]["metrics"]["answer_relevancy"]["scored"] == 2
    score_records = load_jsonl(run_dir / "scores.jsonl")
    assert score_records[1]["skipped"]["faithfulness"] == "No retrieved contexts"
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "scored"
    captured = capsys.readouterr()
    assert "RAGAS metrics ready" in captured.out
    assert "Scoring metrics" in captured.err


def test_score_dataset_fails_clearly_when_no_metric_can_be_scored(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample = {
        "id": "q1",
        "user_input": "Domanda",
        "response": "Nessun documento",
        "retrieved_contexts": [],
    }
    (run_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text('{"status": "collected"}\n', encoding="utf-8")
    config_data = evaluation_config_data()
    config_data["metrics"] = ["faithfulness"]
    config = EvaluationConfig.model_validate(config_data)
    monkeypatch.setattr("evaluation.ragas_scoring.build_ragas_judge", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "evaluation.ragas_scoring._build_metrics",
        lambda *_args, **_kwargs: {"faithfulness": MagicMock()},
    )

    with pytest.raises(RuntimeError, match="produced no scores"):
        score_dataset(run_dir, config, MagicMock(), cache_folder="/tmp/cache", embedding_device="cpu")

    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "scoring_failed"


def test_score_dataset_rejects_existing_scores_without_overwrite(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample = {"id": "q1", "user_input": "Domanda", "response": "Risposta", "retrieved_contexts": ["Testo"]}
    (run_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (run_dir / "scores.jsonl").write_text("existing\n", encoding="utf-8")
    config = EvaluationConfig.model_validate(evaluation_config_data())
    build_judge = MagicMock()
    monkeypatch.setattr("evaluation.ragas_scoring.build_ragas_judge", build_judge)

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        score_dataset(run_dir, config, MagicMock(), cache_folder="/tmp/cache", embedding_device="cpu")

    assert (run_dir / "scores.jsonl").read_text(encoding="utf-8") == "existing\n"
    build_judge.assert_not_called()


def test_score_dataset_rejects_non_finite_metric_results(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample = {"id": "q1", "user_input": "Domanda", "response": "Risposta", "retrieved_contexts": ["Testo"]}
    (run_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    config_data = evaluation_config_data()
    config_data["metrics"] = ["faithfulness"]
    config = EvaluationConfig.model_validate(config_data)

    class NonFiniteMetric:
        def score(self, **_kwargs):
            return SimpleNamespace(value=float("nan"))

    monkeypatch.setattr("evaluation.ragas_scoring.build_ragas_judge", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "evaluation.ragas_scoring._build_metrics",
        lambda *_args, **_kwargs: {"faithfulness": NonFiniteMetric()},
    )

    with pytest.raises(RuntimeError, match="produced no scores"):
        score_dataset(run_dir, config, MagicMock(), cache_folder="/tmp/cache", embedding_device="cpu")

    score = load_jsonl(run_dir / "scores.jsonl")[0]
    assert score["metrics"]["faithfulness"] is None
    assert score["errors"]["faithfulness"]["type"] == "ValueError"


def test_limited_score_run_is_marked_partial(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    samples = [
        {"id": "q1", "user_input": "Uno", "response": "Risposta", "retrieved_contexts": []},
        {"id": "q2", "user_input": "Due", "response": "Risposta", "retrieved_contexts": []},
    ]
    (run_dir / "samples.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    (run_dir / "run.json").write_text('{"status": "collected"}\n', encoding="utf-8")
    config_data = evaluation_config_data()
    config_data["metrics"] = ["answer_relevancy"]
    config = EvaluationConfig.model_validate(config_data)

    class Metric:
        def score(self, **_kwargs):
            return SimpleNamespace(value=0.5)

    monkeypatch.setattr("evaluation.ragas_scoring.build_ragas_judge", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "evaluation.ragas_scoring._build_metrics",
        lambda *_args, **_kwargs: {"answer_relevancy": Metric()},
    )

    summary = score_dataset(
        run_dir,
        config,
        MagicMock(),
        cache_folder="/tmp/cache",
        embedding_device="cpu",
        limit=1,
    )

    assert summary["scoring"]["samples"] == 1
    assert summary["scoring"]["available_samples"] == 2
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "scored_partial"
