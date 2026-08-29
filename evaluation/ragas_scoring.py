from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from evaluation.tracing import (
    EVALUATION_SCHEMA_VERSION,
    RESPONSE_TYPES,
    EvaluationConfig,
    ResponseType,
    load_jsonl,
    read_gzip_json,
    require_schema_version,
    utc_now,
    validate_sample_trace,
    write_json,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

# Evaluation inputs are medical-domain data; disable RAGAS usage telemetry by default.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


class BindableLLM(Protocol):
    def invoke(self, input_data: Any) -> Mapping[str, str]: ...


class LocalStructuredJudge:
    """Generate validated structured responses with the project's local LLM."""

    def __init__(self, llm: BindableLLM, *, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._lock = Lock()

    @staticmethod
    def parse_response(content: str, response_model: type[ResponseModel]) -> ResponseModel:
        content = content.strip()
        try:
            return response_model.model_validate_json(content)
        except (ValidationError, ValueError) as direct_error:
            decoder = json.JSONDecoder()
            for position, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(content[position:])
                except json.JSONDecodeError:
                    continue
                try:
                    return response_model.model_validate(value)
                except ValidationError:
                    continue
            raise ValueError(f"Judge returned invalid {response_model.__name__} JSON") from direct_error

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent evaluator. Follow the evaluation instructions exactly and return only "
                    "one valid JSON object, with no Markdown or explanatory text."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nRequired JSON schema:\n{schema}",
            },
        ]

        with self._lock:
            for attempt in range(self._max_retries + 1):
                result = self._llm.invoke(messages)
                if not isinstance(result, Mapping) or not isinstance(result.get("content"), str):
                    raise TypeError("Judge LLM returned no string content")
                content = result["content"]
                try:
                    return self.parse_response(content, response_model)
                except ValueError as error:
                    if attempt == self._max_retries:
                        raise
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "The previous response failed validation. Return only one JSON object matching "
                                    f"this schema: {schema}. Validation error: {error}"
                                ),
                            },
                        ]
                    )
        raise RuntimeError("Unreachable structured generation state")

    async def agenerate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        return await asyncio.to_thread(self.generate, prompt, response_model)


def build_ragas_judge(llm: BindableLLM, *, max_retries: int):
    try:
        from ragas.llms.base import InstructorBaseRagasLLM
    except ImportError as error:
        raise RuntimeError(
            "RAGAS evaluation dependencies are unavailable. Install them with `uv sync --locked --extra evaluation`."
        ) from error

    delegate = LocalStructuredJudge(llm, max_retries=max_retries)

    class ProjectRagasLLM(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):
            return delegate.generate(prompt, response_model)

        async def agenerate(self, prompt, response_model):
            return await delegate.agenerate(prompt, response_model)

    return ProjectRagasLLM()


def _build_metrics(config: EvaluationConfig, judge, *, cache_folder: str, embedding_device: str) -> dict[str, Any]:
    try:
        from ragas.metrics.collections import AnswerRelevancy, ContextUtilization, Faithfulness
    except ImportError as error:
        raise RuntimeError(
            "RAGAS evaluation dependencies are unavailable. Install them with `uv sync --locked --extra evaluation`."
        ) from error

    metrics: dict[str, Any] = {}
    for metric_name in config.metrics:
        if metric_name == "faithfulness":
            metrics[metric_name] = Faithfulness(llm=judge)
        elif metric_name == "context_utilization":
            metrics[metric_name] = ContextUtilization(llm=judge)
        elif metric_name == "answer_relevancy":
            from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings

            print(
                f"Loading embedding model: {config.embeddings.model} on {embedding_device}",
                flush=True,
            )
            embeddings = HuggingFaceEmbeddings(
                model=config.embeddings.model,
                use_api=False,
                device=embedding_device,
                normalize_embeddings=config.embeddings.normalize_embeddings,
                batch_size=config.embeddings.batch_size,
                cache_folder=cache_folder,
                local_files_only=not config.embeddings.online,
            )
            metrics[metric_name] = AnswerRelevancy(llm=judge, embeddings=embeddings)
        else:
            raise ValueError(f"Unsupported RAGAS metric: {metric_name}")
    return metrics


def _validate_sample(sample: Mapping[str, Any]) -> tuple[str, ResponseType, str, str, list[str]]:
    require_schema_version(sample, "samples.jsonl record")
    sample_id = sample.get("id")
    response_type = sample.get("response_type")
    user_input = sample.get("user_input")
    response = sample.get("response")
    contexts = sample.get("retrieved_contexts")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Evaluation sample has no string id")
    if response_type not in RESPONSE_TYPES:
        raise ValueError(f"Evaluation sample {sample_id} has invalid response_type: {response_type!r}")
    if not isinstance(user_input, str) or not isinstance(response, str):
        raise ValueError(f"Evaluation sample {sample_id} has invalid input or response")
    if not isinstance(contexts, list) or not all(isinstance(context, str) for context in contexts):
        raise ValueError(f"Evaluation sample {sample_id} has invalid retrieved contexts")
    if response_type == "rag" and not contexts:
        raise ValueError(f"RAG evaluation sample {sample_id} has no retrieved contexts")
    if response_type == "non_rag" and contexts:
        raise ValueError(f"Non-RAG evaluation sample {sample_id} has retrieved contexts")
    return sample_id, cast(ResponseType, response_type), user_input, response, contexts


def validate_scoring_run(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    samples_path = run_dir / "samples.jsonl"
    samples = load_jsonl(samples_path)
    traces_dir = run_dir / "traces"
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"Trace directory not found: {traces_dir}")
    traces: dict[str, tuple[Path, dict[str, Any]]] = {}
    for trace_path in sorted(traces_dir.glob("*.json.gz")):
        trace = read_gzip_json(trace_path)
        require_schema_version(trace, trace_path)
        question_id = trace.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Trace has no string question_id: {trace_path}")
        if question_id in traces:
            raise ValueError(f"Duplicate trace question id: {question_id}")
        traces[question_id] = (trace_path, trace)

    sample_ids: set[str] = set()
    for sample in samples:
        sample_id, _response_type, _user_input, _response, _contexts = _validate_sample(sample)
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate evaluation sample id: {sample_id}")
        sample_ids.add(sample_id)
        trace_relative_path = sample.get("trace_path")
        expected_trace_path = f"traces/{sample_id}.json.gz"
        if trace_relative_path != expected_trace_path:
            raise ValueError(f"Evaluation sample {sample_id} has invalid trace_path: {trace_relative_path!r}")
        trace_entry = traces.get(sample_id)
        if trace_entry is None:
            raise FileNotFoundError(f"Trace not found for evaluation sample {sample_id}")
        trace_path, trace = trace_entry
        validate_sample_trace(sample, trace, trace_path)

    for name in ("run.json", "summary.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        require_schema_version(value, path)
    return samples


def _metric_summary(
    metric_names: Mapping[str, Any],
    values: Mapping[str, list[float]],
    errors: Mapping[str, int],
    skipped: Mapping[str, int],
) -> dict[str, dict[str, float | int | None]]:
    return {
        name: {
            "mean": sum(values[name]) / len(values[name]) if values[name] else None,
            "scored": len(values[name]),
            "errors": errors[name],
            "skipped": skipped[name],
        }
        for name in metric_names
    }


def score_dataset(
    run_dir: Path,
    config: EvaluationConfig,
    llm: BindableLLM,
    *,
    cache_folder: str,
    embedding_device: str,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists() and not overwrite:
        raise FileExistsError(f"Scores already exist: {scores_path}. Pass --overwrite to replace them.")

    all_samples = validate_scoring_run(run_dir)
    validated_all_samples = [_validate_sample(sample) for sample in all_samples]
    validated_samples = validated_all_samples
    if limit is not None:
        validated_samples = validated_samples[:limit]
    if not validated_samples:
        raise ValueError(f"No collected samples to score in {run_dir}")

    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {"schema_version": EVALUATION_SCHEMA_VERSION}
    )
    if not isinstance(summary, dict):
        raise TypeError(f"Expected a JSON object in {summary_path}")
    require_schema_version(summary, summary_path)

    run_path = run_dir / "run.json"
    if run_path.is_file():
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError(f"Expected a JSON object in {run_path}")
        require_schema_version(manifest, run_path)
        manifest.update(
            {
                "status": "scoring",
                "scoring_started_at": utc_now(),
                "judge_model": config.judge.model,
            }
        )
        write_json(run_path, manifest)

    judge = build_ragas_judge(llm, max_retries=config.judge.max_retries)
    metrics = _build_metrics(config, judge, cache_folder=cache_folder, embedding_device=embedding_device)
    print(f"RAGAS metrics ready: {', '.join(metrics)}", flush=True)
    scores_path.write_text("", encoding="utf-8")
    scores_path.chmod(0o600)

    values: dict[str, list[float]] = {name: [] for name in metrics}
    errors: dict[str, int] = {name: 0 for name in metrics}
    skipped: dict[str, int] = {name: 0 for name in metrics}
    values_by_response_type = {response_type: {name: [] for name in metrics} for response_type in RESPONSE_TYPES}
    errors_by_response_type = {response_type: {name: 0 for name in metrics} for response_type in RESPONSE_TYPES}
    skipped_by_response_type = {response_type: {name: 0 for name in metrics} for response_type in RESPONSE_TYPES}

    operation_count = len(validated_samples) * len(metrics)
    with tqdm(total=operation_count, desc="Scoring metrics", unit="metric") as progress:
        for sample_number, sample in enumerate(validated_samples, 1):
            sample_id, response_type, user_input, response, contexts = sample
            score_record: dict[str, Any] = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "id": sample_id,
                "response_type": response_type,
                "metrics": {},
                "errors": {},
                "skipped": {},
            }
            for metric_name, metric in metrics.items():
                progress.set_postfix(
                    sample=f"{sample_number}/{len(validated_samples)}",
                    question=sample_id,
                    metric=metric_name,
                )
                try:
                    if metric_name in {"faithfulness", "context_utilization"} and not contexts:
                        score_record["metrics"][metric_name] = None
                        score_record["skipped"][metric_name] = "No retrieved contexts"
                        skipped[metric_name] += 1
                        skipped_by_response_type[response_type][metric_name] += 1
                        continue
                    kwargs = {"user_input": user_input, "response": response}
                    if metric_name in {"faithfulness", "context_utilization"}:
                        kwargs["retrieved_contexts"] = contexts
                    result = metric.score(**kwargs)
                    value = float(result.value)
                    if not math.isfinite(value):
                        raise ValueError(f"Metric returned a non-finite score: {value}")
                    score_record["metrics"][metric_name] = value
                    values[metric_name].append(value)
                    values_by_response_type[response_type][metric_name].append(value)
                except Exception as error:
                    score_record["metrics"][metric_name] = None
                    score_record["errors"][metric_name] = {"type": type(error).__name__, "message": str(error)}
                    errors[metric_name] += 1
                    errors_by_response_type[response_type][metric_name] += 1
                finally:
                    progress.update()

            with scores_path.open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(score_record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                )
                output.flush()

    metric_summary = _metric_summary(metrics, values, errors, skipped)
    metrics_by_response_type = {
        response_type: _metric_summary(
            metrics,
            values_by_response_type[response_type],
            errors_by_response_type[response_type],
            skipped_by_response_type[response_type],
        )
        for response_type in RESPONSE_TYPES
    }
    samples_by_response_type = {
        response_type: sum(sample[1] == response_type for sample in validated_samples)
        for response_type in RESPONSE_TYPES
    }
    available_samples_by_response_type = {
        response_type: sum(sample[1] == response_type for sample in validated_all_samples)
        for response_type in RESPONSE_TYPES
    }
    summary["scoring"] = {
        "completed_at": utc_now(),
        "samples": len(validated_samples),
        "available_samples": len(all_samples),
        "samples_by_response_type": samples_by_response_type,
        "available_samples_by_response_type": available_samples_by_response_type,
        "judge_model": config.judge.model,
        "embedding_model": config.embeddings.model if "answer_relevancy" in metrics else None,
        "embedding_device": embedding_device if "answer_relevancy" in metrics else None,
        "metrics": metric_summary,
        "metrics_by_response_type": metrics_by_response_type,
    }
    write_json(summary_path, summary)

    total_scored = sum(len(metric_values) for metric_values in values.values())
    if run_path.is_file():
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        status = "scoring_failed" if not total_scored else "scored"
        if total_scored and len(validated_samples) < len(all_samples):
            status = "scored_partial"
        manifest.update({"status": status, "scored_at": utc_now()})
        write_json(run_path, manifest)
    if not total_scored:
        raise RuntimeError(f"RAGAS produced no scores; inspect {scores_path} for errors and skipped metrics")
    return summary
