"""Post-process tau2 results with our custom evaluator.

Usage:
    python -m tau2.eigen.postprocess \
        data/simulations/<run_name>/results.json \
        --name airline_eigen \
        -o data/simulations/<run_name>/results_scored.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

from loguru import logger

from tau2.data_model.simulation import (
    DBCheck,
    RewardInfo,
    Results,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType
from tau2.eigen.config import EigenDomainConfig
from tau2.eigen.output_adapter import build_eval_payload
from tau2.metrics.agent_metrics import compute_metrics
from tau2.utils.utils import DATA_DIR


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper providing ``.generate(prompt) -> str`` interface via LiteLLM."""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model

    def generate(self, prompt: str) -> str:
        from litellm import completion

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


def _load_sample_evaluator(
    domain_dir: Path,
    source_dir: str | Path,
    sample_index: str,
):
    """Load the per-sample evaluator module from the registered domain."""
    evaluator_path = domain_dir / "evaluators" / f"evaluator_{sample_index}.py"
    if not evaluator_path.exists():
        evaluator_path = Path(source_dir) / "evaluators" / f"evaluator_{sample_index}.py"
    if not evaluator_path.exists():
        raise FileNotFoundError(
            f"Evaluator not found for sample {sample_index}: {evaluator_path}"
        )

    module_name = f"tau2_eigen_eval_{domain_dir.name}_{sample_index}"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluator module from {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    evaluate_submission = getattr(module, "evaluate_submission", None)
    if evaluate_submission is None:
        raise AttributeError(
            f"Evaluator {evaluator_path} does not define evaluate_submission()"
        )

    return evaluate_submission


# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------


def eval_result_to_reward_info(eval_result: dict) -> RewardInfo:
    """Map our evaluator output to tau2 RewardInfo."""
    config_match = eval_result.get("config_match", False)
    function_match = eval_result.get("function_call_match", False)
    llm_passed = eval_result.get("llm_judgment", {}).get("passed", False)
    overall_pass = eval_result.get("overall_pass", False)

    return RewardInfo(
        reward=1.0 if overall_pass else 0.0,
        db_check=DBCheck(
            db_match=config_match,
            db_reward=1.0 if config_match else 0.0,
        ),
        reward_basis=[RewardType.DB, RewardType.ACTION, RewardType.NL_ASSERTION],
        reward_breakdown={
            RewardType.DB: 1.0 if config_match else 0.0,
            RewardType.ACTION: 1.0 if function_match else 0.0,
            RewardType.NL_ASSERTION: 1.0 if llm_passed else 0.0,
        },
        info={"eigen_eval": eval_result},
    )


def _build_sample_status_rows(results: Results) -> list[dict]:
    """Return per-sample evaluation status rows for metrics output."""
    rows: list[dict] = []
    for sim in results.simulations:
        eigen = (sim.reward_info.info or {}).get("eigen_eval", {})
        if not eigen:
            continue
        rows.append(
            {
                "sample": sim.task_id,
                "status": "PASS" if eigen.get("overall_pass", False) else "FAIL",
                "config": bool(eigen.get("config_match", False)),
                "func": bool(eigen.get("function_call_match", False)),
                "llm": bool((eigen.get("llm_judgment") or {}).get("passed", False)),
            }
        )
    return rows


def _build_sample_status_markdown(rows: list[dict]) -> str:
    """Render per-sample evaluation status as a markdown table."""
    lines = [
        "| sample | status | config | func | llm |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample']} | {row['status']} | {row['config']} | {row['func']} | {row['llm']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main post-processing
# ---------------------------------------------------------------------------


def postprocess(
    results_path: str,
    domain_name: str,
    output_path: str | None = None,
    llm_model: str = "gpt-4.1",
) -> Results:
    """Load tau2 results, run our evaluator, inject scores, save.

    Args:
        results_path: Path to tau2's results.json.
        domain_name: Eigen domain name (e.g., "airline_eigen").
        output_path: Where to save scored results. If None, overwrites input.
        llm_model: LLM model for evaluator judgments.

    Returns:
        Updated Results object.
    """
    # Load config
    config_path = DATA_DIR / "tau2" / "domains" / domain_name / "domain_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {config_path}. "
            f"Run 'python -m tau2.eigen.register' first."
        )
    config = EigenDomainConfig.load(config_path)

    # Load results
    results = Results.load(Path(results_path))
    logger.info(f"Loaded {len(results.simulations)} simulations from {results_path}")

    # Load initial DB for eval_payload construction
    from tau2.eigen.domain_factory import load_db_for_domain

    initial_db = load_db_for_domain(config.base_domain, config.db_path).model_dump()

    # Initialize LLM client
    llm_client = LLMClient(model=llm_model)

    # Process each simulation
    domain_dir = config_path.parent
    scored = 0
    skipped = 0

    for sim in results.simulations:
        # Skip failed simulations
        if sim.termination_reason not in {
            TerminationReason.AGENT_STOP,
            TerminationReason.USER_STOP,
        }:
            sim.reward_info = RewardInfo(
                reward=0.0,
                reward_basis=[RewardType.DB, RewardType.ACTION, RewardType.NL_ASSERTION],
                info={"note": f"Skipped: termination_reason={sim.termination_reason.value}"},
            )
            skipped += 1
            continue

        # Extract sample index from task_id ("eigen_000001" → "000001")
        sample_index = sim.task_id.replace("eigen_", "")

        # Load reference_payload (prefer domain dir, fall back to source_dir)
        ref_path = domain_dir / "reference_payloads" / f"reference_payload_{sample_index}.json"
        if not ref_path.exists():
            ref_path = Path(config.source_dir) / "reference_payloads" / f"reference_payload_{sample_index}.json"
        if not ref_path.exists():
            logger.warning(f"Reference payload not found: {ref_path}, skipping")
            skipped += 1
            continue

        with open(ref_path) as f:
            ref_payload = json.load(f)

        # Build eval_payload from simulation
        eval_payload = build_eval_payload(sim, config, initial_db)

        # Load and run the sample-specific evaluator.
        evaluate_submission = _load_sample_evaluator(
            domain_dir, config.source_dir, sample_index
        )
        logger.info(f"Evaluating {sim.task_id} (trial {sim.trial})...")
        eval_result = evaluate_submission(ref_payload, eval_payload, llm_client)

        # Inject score
        sim.reward_info = eval_result_to_reward_info(eval_result)
        scored += 1

        status = "PASS" if eval_result["overall_pass"] else "FAIL"
        logger.info(
            f"  {sim.task_id}: {status}  "
            f"config={eval_result['config_match']}  "
            f"func={eval_result['function_call_match']}  "
            f"llm={eval_result.get('llm_judgment', {}).get('passed', False)}"
        )

    logger.info(f"Scored {scored} simulations, skipped {skipped}")

    # Save
    save_path = Path(output_path) if output_path else Path(results_path)
    results.save(save_path)
    logger.info(f"Saved scored results to {save_path}")

    # Print metrics
    metrics = compute_metrics(results)
    func_pass = llm_pass = 0
    for sim in results.simulations:
        eigen = (sim.reward_info.info or {}).get("eigen_eval", {})
        if not eigen:
            continue
        if eigen.get("function_call_match"):
            func_pass += 1
        if eigen.get("llm_judgment", {}).get("passed"):
            llm_pass += 1
    logger.info(
        f"Metrics: avg_reward={metrics.avg_reward:.3f}, "
        f"pass^1={metrics.pass_hat_ks.get(1, 0):.3f}, "
        f"db_match={metrics.db_match_count}/{scored}, "
        f"func_match={func_pass}/{scored}, "
        f"llm_judge={llm_pass}/{scored}"
    )

    # Save metrics summary
    overall_pass = sum(
        1 for sim in results.simulations
        if (sim.reward_info.info or {}).get("eigen_eval", {}).get("overall_pass")
    )
    sample_rows = _build_sample_status_rows(results)
    metrics_summary = {
        "avg_reward": round(metrics.avg_reward, 3),
        "pass_at_1": round(metrics.pass_hat_ks.get(1, 0), 3),
        "total": scored,
        "skipped": skipped,
        "db_match": metrics.db_match_count,
        "func_match": func_pass,
        "llm_judge": llm_pass,
        "overall_pass": overall_pass,
        "sample_results": sample_rows,
        "sample_results_markdown": _build_sample_status_markdown(sample_rows),
    }
    metrics_path = save_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Post-process tau2 results with eigen evaluator."
    )
    parser.add_argument("results_path", help="Path to tau2 results.json")
    parser.add_argument(
        "--name", required=True, help="Eigen domain name (e.g., 'airline_eigen')"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output path for scored results. Defaults to overwriting input."
    )
    parser.add_argument(
        "--llm-model", default="gpt-4.1", help="LLM model for evaluator judgments (default: gpt-4.1)"
    )
    args = parser.parse_args()
    postprocess(
        results_path=args.results_path,
        domain_name=args.name,
        output_path=args.output,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
