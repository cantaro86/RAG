from __future__ import annotations

import gzip
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

TRACE_SCHEMA_VERSION = 1
SAMPLE_SCHEMA_VERSION = 1

MetricName = Literal["faithfulness", "context_utilization", "answer_relevancy"]


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    online: bool
    quantization: bool
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(gt=0)
    repetition_penalty: float = Field(ge=1.0)
    no_repeat_ngram_size: int = Field(ge=0)
    max_retries: int = Field(ge=0)

    @model_validator(mode="after")
    def require_deterministic_generation(self):
        if self.temperature != 0.0:
            raise ValueError("The RAGAS judge temperature must be 0.0 for reproducible scoring")
        return self


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    online: bool
    normalize_embeddings: bool
    batch_size: int = Field(gt=0)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: JudgeConfig
    embeddings: EmbeddingsConfig
    metrics: list[MetricName] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metrics(self):
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Evaluation metrics must not contain duplicates")
        return self


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    line_number: int
    text: str


@dataclass
class CollectionOutcome:
    sample: dict[str, Any] | None
    trace: dict[str, Any]


class TraceableAgent(Protocol):
    checkpointer: Any

    def invoke(self, input_data: Mapping[str, Any], *, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_state_history(self, config: Mapping[str, Any]) -> Iterable[Any]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_evaluation_config(path: Path) -> EvaluationConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation configuration file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if data is None:
        raise ValueError(f"Evaluation configuration file is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Evaluation configuration must contain a top-level mapping: {path}")
    return EvaluationConfig.model_validate(data)


def load_questions(path: Path) -> list[EvaluationQuestion]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions: list[EvaluationQuestion] = []
    first_occurrence: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        if text in first_occurrence:
            raise ValueError(
                f"Duplicate question at line {line_number}; first occurrence is line {first_occurrence[text]}: {text}"
            )
        first_occurrence[text] = line_number
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        questions.append(EvaluationQuestion(id=f"q{line_number:04d}-{digest}", line_number=line_number, text=text))

    if not questions:
        raise ValueError(f"Questions file contains no nonblank questions: {path}")
    return questions


def _json_value(value: Any, documents: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, Document):
        payload: dict[str, Any] = {
            "page_content": value.page_content,
            "metadata": _json_value(value.metadata, documents),
        }
        if value.id is not None:
            payload["id"] = value.id
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        document_id = f"doc-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        existing = documents.setdefault(document_id, payload)
        if existing != payload:
            raise ValueError(f"Document hash collision for {document_id}")
        return {"$document": document_id}
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Cannot serialize non-finite number: {number}")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Graph state mappings must use string keys, got {type(key).__name__}")
            result[key] = _json_value(item, documents)
        return result
    if isinstance(value, list | tuple):
        return [_json_value(item, documents) for item in value]
    raise TypeError(f"Unsupported graph state value: {type(value).__name__}")


def _snapshot_nodes(metadata: Mapping[str, Any], previous_next: list[str]) -> list[str]:
    writes = metadata.get("writes")
    if isinstance(writes, Mapping):
        return [str(name) for name in writes]
    if metadata.get("source") == "input":
        return ["__input__"]
    return previous_next


def _serialize_tasks(tasks: Iterable[Any], documents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for task in tasks:
        error = getattr(task, "error", None)
        result = getattr(task, "result", None)
        serialized.append(
            {
                "id": getattr(task, "id", None),
                "name": getattr(task, "name", None),
                "error": str(error) if error is not None else None,
                "result": _json_value(result, documents) if result is not None else None,
            }
        )
    return serialized


def serialize_snapshots(snapshots: Iterable[Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered = sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot.metadata.get("step", 0) if isinstance(snapshot.metadata, Mapping) else 0,
            snapshot.created_at or "",
        ),
    )
    documents: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    previous_next: list[str] = []

    for sequence, snapshot in enumerate(ordered):
        metadata = dict(snapshot.metadata) if isinstance(snapshot.metadata, Mapping) else {}
        next_nodes = [str(node) for node in snapshot.next]
        checkpoint_id = None
        configurable = snapshot.config.get("configurable", {}) if isinstance(snapshot.config, Mapping) else {}
        if isinstance(configurable, Mapping):
            checkpoint_id = configurable.get("checkpoint_id")

        step = {
            "sequence": sequence,
            "checkpoint_step": metadata.get("step"),
            "checkpoint_id": checkpoint_id,
            "created_at": snapshot.created_at,
            "nodes": _snapshot_nodes(metadata, previous_next),
            "next_nodes": next_nodes,
            "state": _json_value(snapshot.values, documents),
            "tasks": _serialize_tasks(getattr(snapshot, "tasks", ()), documents),
        }
        steps.append(step)
        previous_next = next_nodes

    return steps, documents


def _document_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item["$document"] for item in value if isinstance(item, dict) and isinstance(item.get("$document"), str)]


def derive_trace_diagnostics(raw_question: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    node_path: list[str] = []
    question_versions = [{"sequence": -1, "nodes": ["__input__"], "question": raw_question}]
    retrieval_attempts: list[dict[str, Any]] = []
    maximum_rewrite_count = 0
    last_question = raw_question

    for step in steps:
        nodes = [node for node in step["nodes"] if node not in {"__input__", "__start__"}]
        node_path.extend(nodes)
        state = step["state"]
        if not isinstance(state, dict):
            continue

        question = state.get("question")
        if isinstance(question, str) and question != last_question:
            question_versions.append(
                {
                    "sequence": step["sequence"],
                    "nodes": step["nodes"],
                    "question": question,
                }
            )
            last_question = question

        rewrite_count = state.get("rewrite_count")
        if isinstance(rewrite_count, int) and not isinstance(rewrite_count, bool):
            maximum_rewrite_count = max(maximum_rewrite_count, rewrite_count)

        if "retrieve_and_filter" in nodes:
            retrieval_attempts.append(
                {
                    "attempt": len(retrieval_attempts) + 1,
                    "sequence": step["sequence"],
                    "question": state.get("question"),
                    "has_docs": state.get("has_docs"),
                    "rewrite_count": state.get("rewrite_count"),
                    "source_filter": state.get("source_filter"),
                    "documents": _document_ids(state.get("documents")),
                }
            )

    final_state = steps[-1]["state"] if steps and isinstance(steps[-1]["state"], dict) else {}
    reformulation_nodes = {"sanitize_question", "pre_retrieval_rewriter", "transform_query"}
    reformulation_count = sum(bool(set(version["nodes"]) & reformulation_nodes) for version in question_versions[1:])
    failed_nodes = [
        task["name"]
        for step in steps
        for task in step["tasks"]
        if task["error"] is not None and isinstance(task["name"], str)
    ]
    return {
        "node_path": node_path,
        "question_versions": question_versions,
        "retrieval_attempts": retrieval_attempts,
        "reformulation_count": reformulation_count,
        "maximum_rewrite_count": maximum_rewrite_count,
        "terminal_node": node_path[-1] if node_path else None,
        "next_nodes": steps[-1]["next_nodes"] if steps else [],
        "failed_nodes": failed_nodes,
        "guardrail_status": final_state.get("guardrail_status"),
        "retrieval_succeeded": any(attempt["has_docs"] is True for attempt in retrieval_attempts),
        "final_has_docs": final_state.get("has_docs"),
    }


def _error_record(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def collect_question(
    agent: TraceableAgent,
    question: EvaluationQuestion,
    *,
    question_validator: Callable[[str], str] | None = None,
) -> CollectionOutcome:
    thread_id = f"ragas-{question.id}-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    started_at = utc_now()
    final_state: Mapping[str, Any] | None = None
    invocation_error: Exception | None = None
    checkpoint_error: Exception | None = None

    try:
        effective_input = question_validator(question.text) if question_validator else question.text
        final_state = agent.invoke({"question": effective_input}, config=config)
        if not isinstance(final_state, Mapping):
            raise TypeError(f"Agent returned {type(final_state).__name__}, expected a mapping")
    except Exception as error:
        invocation_error = error

    try:
        snapshots = list(agent.get_state_history(config))
        steps, documents = serialize_snapshots(snapshots)
    except Exception as error:
        checkpoint_error = error
        steps, documents = [], {}

    cleanup_error = None
    try:
        checkpointer = getattr(agent, "checkpointer", None)
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(thread_id)
    except Exception as error:
        cleanup_error = error

    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "question_id": question.id,
        "question_line": question.line_number,
        "raw_question": question.text,
        "thread_id": thread_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "failed" if invocation_error or checkpoint_error else "completed",
        "documents": documents,
        "steps": steps,
        "diagnostics": derive_trace_diagnostics(question.text, steps),
    }
    if invocation_error:
        trace["error"] = _error_record(invocation_error)
    if checkpoint_error:
        trace["checkpoint_error"] = _error_record(checkpoint_error)
    if cleanup_error:
        trace["cleanup_error"] = _error_record(cleanup_error)

    if invocation_error or checkpoint_error or final_state is None:
        return CollectionOutcome(sample=None, trace=trace)

    generation = final_state.get("generation")
    final_documents = final_state.get("documents", [])
    try:
        if not isinstance(generation, str):
            raise TypeError("Final graph state does not contain a string generation")
        if not isinstance(final_documents, list) or not all(
            isinstance(document, Document) for document in final_documents
        ):
            raise TypeError("Final graph state documents must be a list of Document objects")
    except Exception as error:
        trace["status"] = "failed"
        trace["error"] = _error_record(error)
        return CollectionOutcome(sample=None, trace=trace)

    sample = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "id": question.id,
        "user_input": question.text,
        "response": generation,
        "retrieved_contexts": [document.page_content for document in final_documents],
    }
    return CollectionOutcome(sample=sample, trace=trace)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"Trace must contain a JSON object: {path}")
    return value


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        output.flush()
    path.chmod(0o600)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def restore_trace_state(value: Any, documents: Mapping[str, Mapping[str, Any]]) -> Any:
    if isinstance(value, dict) and set(value) == {"$document"}:
        document_id = value["$document"]
        payload = documents.get(document_id)
        if payload is None:
            raise KeyError(f"Unknown trace document: {document_id}")
        return Document(
            id=payload.get("id"),
            page_content=str(payload["page_content"]),
            metadata=restore_trace_state(dict(payload.get("metadata", {})), documents),
        )
    if isinstance(value, dict):
        return {key: restore_trace_state(item, documents) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_trace_state(item, documents) for item in value]
    return value


def collect_dataset(
    agent: TraceableAgent,
    questions: list[EvaluationQuestion],
    run_dir: Path,
    *,
    question_validator: Callable[[str], str] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    traces_dir = run_dir / "traces"
    samples_path = run_dir / "samples.jsonl"
    if samples_path.exists():
        raise FileExistsError(f"Evaluation run already contains samples: {samples_path}")
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir.chmod(0o700)
    traces_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    traces_dir.chmod(0o700)
    samples_path.touch()
    samples_path.chmod(0o600)

    manifest = {
        "schema_version": 1,
        "status": "collecting",
        "started_at": utc_now(),
        "question_count": len(questions),
        **dict(run_metadata or {}),
    }
    write_json(run_dir / "run.json", manifest)

    completed_count = 0
    sample_count = 0
    retrieval_succeeded_count = 0
    reformulated_count = 0
    retrieval_attempt_count = 0
    try:
        for question in questions:
            outcome = collect_question(agent, question, question_validator=question_validator)
            trace_relative_path = Path("traces") / f"{question.id}.json.gz"
            write_gzip_json(run_dir / trace_relative_path, outcome.trace)
            if outcome.sample is not None:
                outcome.sample["trace_path"] = trace_relative_path.as_posix()
                append_jsonl(samples_path, outcome.sample)
                sample_count += 1
            if outcome.trace["status"] == "completed":
                completed_count += 1
                diagnostics = outcome.trace["diagnostics"]
                retrieval_succeeded_count += diagnostics["retrieval_succeeded"]
                reformulated_count += diagnostics["reformulation_count"] > 0
                retrieval_attempt_count += len(diagnostics["retrieval_attempts"])
    except Exception as error:
        manifest.update(
            {
                "status": "collection_failed",
                "failed_at": utc_now(),
                "error": _error_record(error),
            }
        )
        write_json(run_dir / "run.json", manifest)
        raise

    summary = {
        "schema_version": 1,
        "collection": {
            "questions": len(questions),
            "completed": completed_count,
            "failed": len(questions) - completed_count,
            "samples": sample_count,
            "retrieval_succeeded": retrieval_succeeded_count,
            "questions_reformulated": reformulated_count,
            "retrieval_attempts": retrieval_attempt_count,
        },
    }
    write_json(run_dir / "summary.json", summary)
    manifest.update({"status": "collected", "completed_at": utc_now()})
    write_json(run_dir / "run.json", manifest)
    return summary
