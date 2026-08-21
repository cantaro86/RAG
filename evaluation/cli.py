"""Collect traced agent outputs and score them with a local RAGAS judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from evaluation.tracing import (
    collect_dataset,
    load_evaluation_config,
    load_jsonl,
    load_questions,
    utc_now,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = REPO_ROOT / "evaluation" / "questions.txt"
DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "evaluation" / "config.yaml"
DEFAULT_RESULTS = REPO_ROOT / "evaluation" / "results"
INDEX_FILENAMES = ("index.faiss", "index.pkl")


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate input files without loading any model")
    validate.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    validate.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)

    collect = subparsers.add_parser("collect", help="Run the RAG agent and save samples and graph traces")
    collect.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    collect.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    collect.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    collect.add_argument("--run-id")
    collect.add_argument("--limit", type=positive_integer)

    score = subparsers.add_parser("score", help="Score a collected run with the independent local judge")
    score.add_argument("run_dir", type=Path)
    score.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    score.add_argument("--limit", type=positive_integer)
    score.add_argument("--overwrite", action="store_true")

    export_excel = subparsers.add_parser(
        "export-excel",
        help="Export a collected run and its RAGAS scores to a formatted Excel workbook",
    )
    export_excel.add_argument("run_dir", type=Path)
    export_excel.add_argument("--output", type=Path)
    export_excel.add_argument("--overwrite", action="store_true")
    return parser


def _require_accelerator(command: str) -> str:
    import torch

    mps = getattr(torch.backends, "mps", None)
    cuda_available = torch.cuda.is_available()
    mps_available = mps is not None and mps.is_available()
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    raise RuntimeError(
        f"The `{command}` command requires a CUDA or MPS accelerator. Allocate a GPU node before running it."
    )


def _secure_runtime_environment() -> None:
    os.umask(0o077)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _validate_agent_resources(config) -> None:
    markdown_dir = Path(config.md_dir)
    if not markdown_dir.is_dir() or not any(path.is_file() for path in markdown_dir.glob("*.md")):
        raise FileNotFoundError(f"No top-level Markdown files found in {markdown_dir}")
    dictionary_path = Path(config.dizionario_path)
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
    index_dir = Path(config.index_dir)
    missing = [name for name in INDEX_FILENAMES if not (index_dir / name).is_file()]
    if not index_dir.is_dir() or missing:
        raise FileNotFoundError(f"FAISS index is incomplete at {index_dir}; missing: {', '.join(missing)}")


def _run_id(requested: str | None) -> str:
    if requested is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"
    if requested in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", requested) is None:
        raise ValueError("Run ID may contain only letters, numbers, dots, underscores, and hyphens")
    return requested


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _mark_run_failed(run_dir: Path, status: str, error: Exception) -> None:
    run_path = run_dir.expanduser().resolve() / "run.json"
    if not run_path.is_file():
        return
    try:
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": status,
                "failed_at": utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        write_json(run_path, manifest)
    except OSError, TypeError, ValueError:
        return


def validate_command(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    config = load_evaluation_config(args.evaluation_config)
    print(f"Validated {len(questions)} questions from {args.questions.resolve()}")
    print(f"Judge: {config.judge.model}")
    print(f"Metrics: {', '.join(config.metrics)}")
    print("No application or evaluator model was loaded.")
    return 0


def collect_command(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    load_evaluation_config(args.evaluation_config)
    if args.limit is not None:
        questions = questions[: args.limit]
    _secure_runtime_environment()
    accelerator = _require_accelerator("collect")

    from agentic_rag._load_env import CONFIG_PATH, cfg
    from agentic_rag.agent_factory import build_rag_agent
    from agentic_rag.detect_language import DetectLanguage

    _validate_agent_resources(cfg)
    run_id = _run_id(args.run_id)
    run_dir = args.results_dir.expanduser().resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"Evaluation run already exists: {run_dir}")

    print(
        f"Collection run {run_id}: {len(questions)} questions on {accelerator}; results: {run_dir}",
        flush=True,
    )
    print("Loading application models...", flush=True)
    agent = build_rag_agent(cfg)
    print("Application models ready. Starting question collection.", flush=True)

    def validate_italian(question: str) -> str:
        return DetectLanguage(question, online=cfg.online).text

    metadata = {
        "run_id": run_id,
        "questions_file": str(args.questions.expanduser().resolve()),
        "application_config": str(CONFIG_PATH),
        "generator_model": cfg.llm_model,
        "git_revision": _git_revision(),
        "generator_config": json.loads(cfg.model_dump_json()),
    }
    try:
        summary = collect_dataset(
            agent,
            questions,
            run_dir,
            question_validator=validate_italian,
            run_metadata=metadata,
        )
    except Exception as error:
        _mark_run_failed(run_dir, "collection_failed", error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Collected run: {run_dir}")
    if summary["collection"]["samples"] == 0:
        raise RuntimeError(f"Collection produced no scoreable samples; inspect traces in {run_dir}")
    return 0


def score_command(args: argparse.Namespace) -> int:
    config = load_evaluation_config(args.evaluation_config)
    samples = load_jsonl(args.run_dir.expanduser().resolve() / "samples.jsonl")
    if not samples:
        raise ValueError(f"No collected samples to score in {args.run_dir.expanduser().resolve()}")
    scores_path = args.run_dir.expanduser().resolve() / "scores.jsonl"
    if scores_path.exists() and not args.overwrite:
        raise FileExistsError(f"Scores already exist: {scores_path}. Pass --overwrite to replace them.")
    _secure_runtime_environment()
    embedding_device = _require_accelerator("score")
    sample_count = min(len(samples), args.limit) if args.limit is not None else len(samples)
    print(
        f"Scoring {sample_count} of {len(samples)} samples on {embedding_device}; run: {args.run_dir.resolve()}",
        flush=True,
    )
    print(f"Loading judge model: {config.judge.model}", flush=True)

    try:
        from agentic_rag._load_env import cfg, hf_hub_cache_for
        from agentic_rag.llm_build import build_llm_pipe
        from evaluation.ragas_scoring import score_dataset

        cache_folder = hf_hub_cache_for(cfg.hf_home)
        judge = build_llm_pipe(
            config.judge.model,
            config.judge.max_new_tokens,
            config.judge.temperature,
            config.judge.top_p,
            config.judge.top_k,
            config.judge.repetition_penalty,
            config.judge.no_repeat_ngram_size,
            quantization=config.judge.quantization,
            online=config.judge.online,
            cache_folder=cache_folder,
        ).bind(do_sample=False)
        print("Judge model ready. Initializing RAGAS metrics.", flush=True)
        summary = score_dataset(
            args.run_dir,
            config,
            judge,
            cache_folder=cache_folder,
            embedding_device=embedding_device,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except Exception as error:
        _mark_run_failed(args.run_dir, "scoring_failed", error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def export_excel_command(args: argparse.Namespace) -> int:
    from evaluation.export_excel import export_run_to_excel

    output_path = export_run_to_excel(args.run_dir, args.output, overwrite=args.overwrite)
    print(f"Excel evaluation report: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return validate_command(args)
    if args.command == "collect":
        return collect_command(args)
    if args.command == "score":
        return score_command(args)
    if args.command == "export-excel":
        return export_excel_command(args)
    raise ValueError(f"Unknown command: {args.command}")


def run(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
