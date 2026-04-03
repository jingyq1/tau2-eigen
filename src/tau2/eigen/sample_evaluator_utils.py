import json
from typing import Any, List


def _parse_wrapper_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return arguments
    return arguments if arguments is not None else {}


def extract_function_calls_from_trajectory(trajectory: dict) -> List[dict]:
    """Extract function calls and unwrap discoverable-tool wrappers."""
    calls = []
    if not trajectory:
        return calls

    conversation = trajectory.get("conversation", {})
    turns = conversation.get("turns", trajectory.get("turns", []))

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        steps = turn.get("assistant_steps", turn.get("steps", []))
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "function_call":
                continue

            name = step.get("function_name") or step.get("tool_name")
            arguments = step.get("parameters") or step.get("arguments", {})
            if not name:
                continue

            if name == "call_discoverable_agent_tool":
                tool_name = arguments.get("agent_tool_name") if isinstance(arguments, dict) else None
                if tool_name:
                    calls.append(
                        {
                            "name": tool_name,
                            "arguments": _parse_wrapper_arguments(arguments.get("arguments")),
                        }
                    )
                continue

            if name == "call_discoverable_user_tool":
                tool_name = None
                if isinstance(arguments, dict):
                    tool_name = arguments.get("discoverable_tool_name") or arguments.get("tool_name")
                if tool_name:
                    calls.append(
                        {
                            "name": tool_name,
                            "arguments": _parse_wrapper_arguments(arguments.get("arguments")),
                        }
                    )
                continue

            calls.append({"name": name, "arguments": arguments or {}})

    return calls
