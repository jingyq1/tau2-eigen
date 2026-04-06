"""Batch evaluation: run tau2 simulations + eigen postprocess for multiple (model, domain) pairs.

Usage:
    # Via CLI
    tau2 eval config.yaml
    tau2 eval config.yaml --only run
    tau2 eval config.yaml --only postprocess
    tau2 eval config.yaml --entry 0

    # Programmatic
    from tau2.eigen.batch_eval import run_batch_eval
    run_batch_eval("config.yaml")
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from tau2.utils.utils import DATA_DIR


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

EVAL_DEFAULTS = {
    "user_llm": "claude-opus-4-6",
    "num_trials": 1,
    "num_tasks": None,
    "eval_llm": "gpt-4.1",
    "agent": "llm_agent",
    "user": "user_simulator",
    "max_steps": 200,
    "max_concurrency": 5,
    "seed": 300,
    "max_retries": 3,
    "retry_delay": 1.0,
    "task_ids": None,
    "task_split_name": "base",
    "save_to": None,
}


@dataclass
class EvalEntry:
    """A single (domain, agent_llm) pair with resolved settings."""

    domain: str
    agent_llm: str
    user_llm: str = "openai/gpt-5.2"
    num_trials: int = 1
    num_tasks: int | None = None
    eval_llm: str = "gpt-4.1"
    agent: str = "llm_agent"
    user: str = "user_simulator"
    max_steps: int = 200
    max_concurrency: int = 5
    seed: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    task_ids: list[str] | None = None
    task_split_name: str = "base"
    save_to: str | None = None


@dataclass
class BatchEvalConfig:
    """Parsed batch eval config."""

    entries: list[EvalEntry] = field(default_factory=list)
    max_workers: int = 4


def load_config(config_path: str | Path) -> BatchEvalConfig:
    """Load a YAML config file and resolve defaults."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = {**EVAL_DEFAULTS, **(raw.get("defaults") or {})}
    max_workers = raw.get("max_workers", 4)

    entries = []
    for item in raw.get("evals", []):
        merged = {**defaults, **item}
        entries.append(EvalEntry(**{k: v for k, v in merged.items() if k in EvalEntry.__dataclass_fields__}))

    return BatchEvalConfig(entries=entries, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Single eval pipeline
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of a single eval run."""

    domain: str
    agent_llm: str
    results_dir: Path | None = None
    metrics: dict | None = None
    error: str | None = None


def _build_run_config(entry: EvalEntry):
    """Build a TextRunConfig from an EvalEntry."""
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain=entry.domain,
        agent=entry.agent,
        llm_agent=entry.agent_llm,
        llm_args_agent={"temperature": 0.0},
        user=entry.user,
        llm_user=entry.user_llm,
        llm_args_user={"temperature": 0.0},
        num_trials=entry.num_trials,
        num_tasks=entry.num_tasks,
        max_steps=entry.max_steps,
        max_errors=10,
        max_concurrency=entry.max_concurrency,
        seed=entry.seed,
        log_level="ERROR",
        max_retries=entry.max_retries,
        retry_delay=entry.retry_delay,
        task_ids=entry.task_ids,
        task_split_name=entry.task_split_name,
        save_to=entry.save_to,
    )


def run_single_eval(entry: EvalEntry, only: str | None = None) -> EvalResult:
    """Run the full eval pipeline for one (domain, agent_llm) pair.

    Args:
        entry: Resolved eval entry.
        only: If "run", only run simulation. If "postprocess", only postprocess.
               If None, run both.

    Returns:
        EvalResult with metrics or error.
    """
    label = f"[{entry.domain} / {entry.agent_llm}]"

    try:
        results_dir = None

        # Step 1: Run simulation
        if only != "postprocess":
            logger.info(f"{label} Starting simulation...")
            from tau2.run import run_domain

            config = _build_run_config(entry)
            run_domain(config)

            # Find the results dir (latest matching this domain + agent)
            results_dir = _find_latest_results_dir(entry.domain, entry.agent_llm)
            if results_dir is None:
                return EvalResult(
                    domain=entry.domain,
                    agent_llm=entry.agent_llm,
                    error="Simulation completed but results directory not found",
                )
            logger.info(f"{label} Simulation saved to {results_dir}")

        # Step 2: Postprocess
        if only != "run":
            if results_dir is None:
                results_dir = _find_latest_results_dir(entry.domain, entry.agent_llm)
            if results_dir is None:
                return EvalResult(
                    domain=entry.domain,
                    agent_llm=entry.agent_llm,
                    error="No results directory found for postprocessing",
                )

            results_path = results_dir / "results.json"
            scored_path = results_dir / "results_scored.json"

            logger.info(f"{label} Running postprocess on {results_path}...")
            from tau2.eigen.postprocess import postprocess

            postprocess(
                results_path=str(results_path),
                domain_name=entry.domain,
                output_path=str(scored_path),
                llm_model=entry.eval_llm,
            )

            # Load metrics
            metrics_path = results_dir / "metrics.json"
            metrics = None
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

            logger.info(f"{label} Done. Metrics: {metrics_path}")
            return EvalResult(
                domain=entry.domain,
                agent_llm=entry.agent_llm,
                results_dir=results_dir,
                metrics=metrics,
            )

        return EvalResult(
            domain=entry.domain,
            agent_llm=entry.agent_llm,
            results_dir=results_dir,
        )

    except Exception as e:
        logger.error(f"{label} Failed: {e}")
        return EvalResult(
            domain=entry.domain,
            agent_llm=entry.agent_llm,
            error=str(e),
        )


def _find_latest_results_dir(domain: str, agent_llm: str) -> Path | None:
    """Find the latest simulation directory matching a domain and agent model."""
    sim_root = DATA_DIR / "simulations"
    if not sim_root.exists():
        return None

    # Extract clean model name (e.g., "openai/gpt-5.3-codex" -> "gpt-5.3-codex")
    clean_model = [x for x in agent_llm.split("/") if x][-1]

    candidates = []
    for d in sim_root.iterdir():
        if d.is_dir() and domain in d.name and clean_model in d.name:
            results_file = d / "results.json"
            if results_file.exists():
                candidates.append(d)

    if not candidates:
        return None

    # Sort by name (timestamp prefix) descending → latest first
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def _run_single_eval_wrapper(args: tuple) -> EvalResult:
    """Wrapper for ProcessPoolExecutor (must be top-level picklable)."""
    entry, only = args
    return run_single_eval(entry, only=only)


def run_batch_eval(
    config_path: str | Path,
    only: str | None = None,
    entry_index: int | None = None,
    max_workers: int | None = None,
) -> list[EvalResult]:
    """Run batch evaluation from a config file.

    Args:
        config_path: Path to YAML config file.
        only: If "run" or "postprocess", only run that step.
        entry_index: If set, only run the Nth entry from the config.
        max_workers: Override max_workers from config.

    Returns:
        List of EvalResult objects.
    """
    config = load_config(config_path)
    entries = config.entries
    workers = max_workers or config.max_workers

    if entry_index is not None:
        if entry_index < 0 or entry_index >= len(entries):
            raise ValueError(
                f"--entry {entry_index} out of range (config has {len(entries)} entries)"
            )
        entries = [entries[entry_index]]

    logger.info(
        f"Batch eval: {len(entries)} pair(s), max_workers={workers}, only={only or 'all'}"
    )
    for i, e in enumerate(entries):
        logger.info(f"  [{i}] {e.domain} / {e.agent_llm}")

    # Run concurrently
    results: list[EvalResult] = []
    if len(entries) == 1:
        # Single entry — run in-process to avoid subprocess overhead
        results.append(run_single_eval(entries[0], only=only))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(entries))) as pool:
            future_to_entry = {
                pool.submit(_run_single_eval_wrapper, (entry, only)): entry
                for entry in entries
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = EvalResult(
                        domain=entry.domain,
                        agent_llm=entry.agent_llm,
                        error=str(e),
                    )
                results.append(result)

    # Print summary
    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Summary table (mirrors scripts/print_tau2_metrics_tables.py)
# ---------------------------------------------------------------------------

HEADERS = [
    "Model",
    "Config Match",
    "Key Func.",
    "LLM Judge",
    "Config + Func",
    "All Three",
]

UPPERCASE_TOKENS = {
    "api": "API",
    "glm": "GLM",
    "gpt": "GPT",
    "llm": "LLM",
    "ui": "UI",
}


def _format_percentage(value: float) -> str:
    return f"{value:.1f}"


def _as_percentage(value: int | float, total: int) -> float:
    if total <= 0:
        return 0.0
    return (float(value) / float(total)) * 100.0


def _normalize_domain_name(raw_domain: str) -> str:
    domain = re.sub(r"^eigendata[-_]", "", raw_domain)
    return domain.replace("-", " ").replace("_", " ").title()


def _normalize_model_name(raw_model: str) -> str:
    # Strip provider prefix (e.g., "openai/gpt-5.3-codex" -> "gpt-5.3-codex")
    clean = [x for x in raw_model.split("/") if x][-1]
    parts = []
    for token in clean.split("-"):
        normalized = UPPERCASE_TOKENS.get(token.lower())
        if normalized:
            parts.append(normalized)
            continue
        parts.append(token if any(ch.isdigit() for ch in token) else token.capitalize())
    return "-".join(parts)


def _metrics_to_row(agent_llm: str, metrics: dict) -> dict[str, str]:
    """Convert a metrics dict to a table row."""
    total = int(metrics.get("total", 0))
    sample_results = metrics.get("sample_results") or []
    config_and_func = sum(
        1
        for sample in sample_results
        if sample.get("config") and sample.get("func")
    )

    return {
        "Model": _normalize_model_name(agent_llm),
        "Config Match": _format_percentage(_as_percentage(metrics.get("db_match", 0), total)),
        "Key Func.": _format_percentage(_as_percentage(metrics.get("func_match", 0), total)),
        "LLM Judge": _format_percentage(_as_percentage(metrics.get("llm_judge", 0), total)),
        "Config + Func": _format_percentage(_as_percentage(config_and_func, total)),
        "All Three": _format_percentage(_as_percentage(metrics.get("overall_pass", 0), total)),
    }


def _render_table(rows: list[dict[str, str]]) -> str:
    """Render a list of row dicts as an aligned ASCII table."""
    table_rows = [[row[header] for header in HEADERS] for row in rows]
    widths = []
    for index, header in enumerate(HEADERS):
        widths.append(max(len(header), *(len(row[index]) for row in table_rows)))

    def _format_row(values: list[str]) -> str:
        formatted = []
        for index, value in enumerate(values):
            if index == 0:
                formatted.append(value.ljust(widths[index]))
            else:
                formatted.append(value.rjust(widths[index]))
        return " | ".join(formatted)

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(HEADERS), separator]
    lines.extend(_format_row(row) for row in table_rows)
    return "\n".join(lines)


def print_summary(results: list[EvalResult]) -> None:
    """Print per-domain summary tables from eval results."""
    tables_by_domain: dict[str, list[dict[str, str]]] = {}

    for r in results:
        if r.error:
            logger.warning(f"FAILED: {r.domain} / {r.agent_llm} — {r.error}")
            continue
        if r.metrics is None:
            continue

        domain_label = _normalize_domain_name(r.domain)
        row = _metrics_to_row(r.agent_llm, r.metrics)
        tables_by_domain.setdefault(domain_label, []).append(row)

    if not tables_by_domain:
        logger.info("No metrics to display.")
        return

    print("\n" + "=" * 60)
    print("BATCH EVAL SUMMARY")
    print("=" * 60)
    for domain in sorted(tables_by_domain):
        rows = sorted(
            tables_by_domain[domain],
            key=lambda row: (-float(row["Config Match"]), row["Model"]),
        )
        print(f"\nDomain: {domain}")
        print(_render_table(rows))
    print()
