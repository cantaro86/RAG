from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from evaluation.tracing import EvaluationConfig, load_jsonl, utc_now, write_json

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
            "RAGAS evaluation dependencies are unavailable. Install them with "
            "`uv sync --locked --extra evaluation`."
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
            "RAGAS evaluation dependencies are unavailable. Install them with "
            "`uv sync --locked --extra evaluation`."
        ) from error

    metrics: dict[str, Any] = {}
    for metric_name in config.metrics:
        if metric_name == "faithfulness":
            metrics[metric_name] = Faithfulness(llm=judge)
        elif metric_name == "context_utilization":
            metrics[metric_name] = ContextUtilization(llm=judge)
        elif metric_name == "answer_relevancy":
            from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings

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


def _validate_sample(sample: Mapping[str, Any]) -> tuple[str, str, str, list[str]]:
    sample_id = sample.get("id")
    user_input = sample.get("user_input")
    response = sample.get("response")
    contexts = sample.get("retrieved_contexts")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Evaluation sample has no string id")
    if not isinstance(user_input, str) or not isinstance(response, str):
        raise ValueError(f"Evaluation sample {sample_id} has invalid input or response")
    if not isinstance(contexts, list) or not all(isinstance(context, str) for context in contexts):
        raise ValueError(f"Evaluation sample {sample_id} has invalid retrieved contexts")
    return sample_id, user_input, response, contexts


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
    all_samples = load_jsonl(run_dir / "samples.jsonl")
    samples = all_samples
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No collected samples to score in {run_dir}")

    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists() and not overwrite:
        raise FileExistsError(f"Scores already exist: {scores_path}. Pass --overwrite to replace them.")

    run_path = run_dir / "run.json"
    if run_path.is_file():
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
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
    scores_path.write_text("", encoding="utf-8")
    scores_path.chmod(0o600)

    values: dict[str, list[float]] = {name: [] for name in metrics}
    errors: dict[str, int] = {name: 0 for name in metrics}
    skipped: dict[str, int] = {name: 0 for name in metrics}

    for sample in samples:
        sample_id, user_input, response, contexts = _validate_sample(sample)
        score_record: dict[str, Any] = {"id": sample_id, "metrics": {}, "errors": {}, "skipped": {}}
        for metric_name, metric in metrics.items():
            if metric_name in {"faithfulness", "context_utilization"} and not contexts:
                score_record["metrics"][metric_name] = None
                score_record["skipped"][metric_name] = "No retrieved contexts"
                skipped[metric_name] += 1
                continue
            kwargs = {"user_input": user_input, "response": response}
            if metric_name in {"faithfulness", "context_utilization"}:
                kwargs["retrieved_contexts"] = contexts
            try:
                result = metric.score(**kwargs)
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError(f"Metric returned a non-finite score: {value}")
                score_record["metrics"][metric_name] = value
                values[metric_name].append(value)
            except Exception as error:
                score_record["metrics"][metric_name] = None
                score_record["errors"][metric_name] = {"type": type(error).__name__, "message": str(error)}
                errors[metric_name] += 1

        with scores_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(score_record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            )
            output.flush()

    metric_summary = {
        name: {
            "mean": sum(metric_values) / len(metric_values) if metric_values else None,
            "scored": len(metric_values),
            "errors": errors[name],
            "skipped": skipped[name],
        }
        for name, metric_values in values.items()
    }
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"schema_version": 1}
    summary["scoring"] = {
        "completed_at": utc_now(),
        "samples": len(samples),
        "available_samples": len(all_samples),
        "judge_model": config.judge.model,
        "embedding_model": config.embeddings.model if "answer_relevancy" in metrics else None,
        "embedding_device": embedding_device if "answer_relevancy" in metrics else None,
        "metrics": metric_summary,
    }
    write_json(summary_path, summary)

    total_scored = sum(len(metric_values) for metric_values in values.values())
    if run_path.is_file():
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        status = "scoring_failed" if not total_scored else "scored"
        if total_scored and len(samples) < len(all_samples):
            status = "scored_partial"
        manifest.update({"status": status, "scored_at": utc_now()})
        write_json(run_path, manifest)
    if not total_scored:
        raise RuntimeError(f"RAGAS produced no scores; inspect {scores_path} for errors and skipped metrics")
    return summary
