"""Convert intent/*.json files to tau2 tasks.json format."""

import json
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import (
    EvaluationCriteria,
    Task,
    UserScenario,
    StructuredUserInstructions,
)


def _build_task_instructions(intent: dict) -> str:
    """Combine motivations and constraints into task_instructions text."""
    lines = []
    motivations = intent.get("motivations") or []
    if motivations:
        lines.append("Motivations:")
        for m in motivations:
            lines.append(f"- {m}")
    constraints = intent.get("constraints") or []
    if constraints:
        if lines:
            lines.append("")
        lines.append("Constraints:")
        for c in constraints:
            lines.append(f"- {c}")
    return "\n".join(lines) if lines else "Follow the goal described above."


def _build_known_info(intent: dict) -> Optional[str]:
    """Build known_info, prepending scenario_time if present."""
    parts = []
    scenario_time = intent.get("scenario_time")
    if scenario_time:
        parts.append(f"Current time: {scenario_time}")
    profile = intent.get("profile")
    if profile:
        parts.append(profile)
    return "\n".join(parts) if parts else None


def intent_to_task(
    sample_id: str, intent: dict, domain: str
) -> dict:
    """Convert a single intent file to a tau2 Task dict.

    Returns a dict (not a Task object) so the caller can serialise
    without Pydantic model_dump round-trips.
    """
    task = Task(
        id=f"eigen_{sample_id}",
        user_scenario=UserScenario(
            persona=intent.get("persona"),
            instructions=StructuredUserInstructions(
                domain=domain,
                reason_for_call=intent.get("goal", ""),
                known_info=_build_known_info(intent),
                unknown_info=None,
                task_instructions=_build_task_instructions(intent),
            ),
        ),
        initial_state=None,
        evaluation_criteria=EvaluationCriteria(
            actions=None,
            reward_basis=[],
        ),
    )
    # Validate round-trip
    Task.model_validate(task.model_dump())
    return task.model_dump()


def convert_intents_to_tasks(
    intent_dir: str | Path,
    domain: str,
    sample_ids: list[str],
    output_path: str | Path,
) -> list[dict]:
    """Convert all intent files to a single tasks.json.

    Args:
        intent_dir: Path to intent/ directory.
        domain: Base domain name (e.g., "airline").
        sample_ids: Validated list of sample IDs to convert.
        output_path: Where to write the tasks.json file.

    Returns:
        List of task dicts that were written.
    """
    intent_dir = Path(intent_dir)
    output_path = Path(output_path)
    tasks = []
    for sid in sample_ids:
        intent_file = intent_dir / f"{sid}.json"
        with open(intent_file) as f:
            intent = json.load(f)
        task_dict = intent_to_task(sid, intent, domain)
        tasks.append(task_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    return tasks
