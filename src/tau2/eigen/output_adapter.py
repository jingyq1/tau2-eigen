"""Convert tau2 SimulationRun → evaluator's eval_payload format.

Handles:
- Messages → trajectory turns
- DB replay for final state
"""

import json
from typing import Optional

from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.eigen.config import EigenDomainConfig
from tau2.eigen.domain_factory import load_db_for_domain


# ---------------------------------------------------------------------------
# Messages → trajectory
# ---------------------------------------------------------------------------


def messages_to_trajectory(messages: list) -> dict:
    """Convert tau2's flat message list into our evaluator's turn-based trajectory."""
    turns = []
    current_turn: Optional[dict] = None

    for msg in messages:
        if isinstance(msg, UserMessage):
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "user_message": msg.content or "",
                "assistant_steps": [],
            }
        elif isinstance(msg, AssistantMessage):
            if current_turn is None:
                current_turn = {"user_message": "", "assistant_steps": []}
            if msg.is_tool_call() and msg.tool_calls:
                for tc in msg.tool_calls:
                    current_turn["assistant_steps"].append({
                        "type": "function_call",
                        "function_name": tc.name,
                        "parameters": tc.arguments or {},
                    })
            elif msg.content:
                current_turn["assistant_steps"].append({
                    "type": "assistant_message",
                    "content": msg.content,
                })
        elif isinstance(msg, ToolMessage):
            if current_turn is None:
                current_turn = {"user_message": "", "assistant_steps": []}
            try:
                output = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                output = msg.content
            current_turn["assistant_steps"].append({
                "type": "function_output",
                "output": output,
            })

    if current_turn is not None:
        turns.append(current_turn)

    return {
        "conversation": {
            "turns": turns,
            "metadata": {"total_turns": len(turns)},
        }
    }


# ---------------------------------------------------------------------------
# DB replay for final state
# ---------------------------------------------------------------------------


def replay_and_get_final_db(simulation: SimulationRun, config: EigenDomainConfig) -> dict:
    """Replay simulation's mutating tool calls on a fresh environment to get final DB state."""
    from tau2.eigen.domain_factory import load_db_for_domain
    from tau2.registry import registry

    base_env_constructor = registry.get_env_constructor(config.base_domain)
    db = load_db_for_domain(config.base_domain, config.db_path)
    env = base_env_constructor(db=db)

    messages = simulation.get_messages() if simulation.messages is None else simulation.messages
    if not messages:
        return db.model_dump()

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call() and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    env.make_tool_call(tc.name, **tc.arguments)
                except Exception as e:
                    logger.debug(f"Tool call replay failed for {tc.name}: {e}")

    return env.tools.db.model_dump()


# ---------------------------------------------------------------------------
# Build eval_payload
# ---------------------------------------------------------------------------


def build_eval_payload(
    simulation: SimulationRun,
    config: EigenDomainConfig,
    initial_db: dict,
) -> dict:
    """Build eval_payload from a tau2 SimulationRun in nested format.

    Produces the same structure as reference_payloads on disk::

        environment_snapshots[env_name] = {
            "initial_state": {"database": initial_db, "state": {"loaded": True}},
            "final_state":   {"database": final_db, "state": {"loaded": True}},
        }

    This matches the evaluator's ``.get("final_state", {})`` access pattern.
    """
    final_db = replay_and_get_final_db(simulation, config)
    messages = simulation.get_messages() if simulation.messages is None else simulation.messages
    trajectory = messages_to_trajectory(messages or [])

    return {
        "environment_snapshots": {
            config.env_name_in_payload: {
                "initial_state": {"database": initial_db, "state": {"database_loaded": True, "scenario": None}},
                "final_state": {"database": final_db, "state": {"database_loaded": True, "scenario": None}},
            }
        },
        "trajectory": trajectory,
    }
