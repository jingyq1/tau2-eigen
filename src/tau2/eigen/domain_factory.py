"""Generic domain factory for eigen-registered domains.

All functions receive ``registry`` as an explicit parameter to avoid
circular imports (registry.py → domain_factory.py → registry.py).
"""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from tau2.data_model.tasks import Task
from tau2.eigen.config import EigenDomainConfig
from tau2.utils import load_file
from tau2.utils.utils import DATA_DIR

if TYPE_CHECKING:
    from tau2.registry import Registry


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def load_db_for_domain(base_domain: str, db_path: str):
    """Load a DB from *file path* via the domain's Pydantic model."""
    if base_domain == "airline":
        from tau2.domains.airline.data_model import FlightDB

        return FlightDB.load(db_path)
    elif base_domain == "retail":
        from tau2.domains.retail.data_model import RetailDB

        return RetailDB.load(db_path)
    elif base_domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.data_model import TransactionalDB

        return TransactionalDB.load(str(db_path))
    else:
        raise ValueError(f"Unknown base domain: {base_domain}")


def validate_db_for_domain(base_domain: str, data: dict):
    """Construct a DB from a *dict* via the domain's ``model_validate()``."""
    if base_domain == "airline":
        from tau2.domains.airline.data_model import FlightDB

        return FlightDB.model_validate(data)
    elif base_domain == "retail":
        from tau2.domains.retail.data_model import RetailDB

        return RetailDB.model_validate(data)
    elif base_domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.data_model import TransactionalDB

        return TransactionalDB.model_validate(data)
    else:
        raise ValueError(f"Unknown base domain: {base_domain}")


# ---------------------------------------------------------------------------
# Domain factory
# ---------------------------------------------------------------------------


def make_domain_functions(config: EigenDomainConfig, registry: "Registry"):
    """Return ``(get_environment, get_tasks)`` callables for the registry.

    ``registry`` is resolved eagerly so the returned closures carry no
    reference to the global singleton.
    """
    base_env_constructor = registry.get_env_constructor(config.base_domain)

    def get_environment(**kwargs):
        if "db" not in kwargs or kwargs["db"] is None:
            kwargs["db"] = load_db_for_domain(config.base_domain, config.db_path)
        env = base_env_constructor(**kwargs)
        env.domain_name = config.eigen_domain_name
        if config.custom_policy_path == "__empty__":
            env.policy = ""
        elif config.custom_policy_path:
            with open(config.custom_policy_path) as f:
                env.policy = f.read()
        return env

    def get_tasks(task_split_name=None):
        if task_split_name is not None and task_split_name != "base":
            raise ValueError(
                f"Eigen domain '{config.eigen_domain_name}' does not support "
                f"task splits. Got --task-split-name '{task_split_name}'. "
                f"Use None or 'base' to load all tasks."
            )
        tasks = load_file(config.tasks_path)
        return [Task.model_validate(t) for t in tasks]

    return get_environment, get_tasks


# ---------------------------------------------------------------------------
# Auto-discovery at tau2 startup
# ---------------------------------------------------------------------------


def discover_and_register_eigen_domains(registry: "Registry") -> None:
    """Scan ``data/tau2/domains/*/domain_config.json`` and register each.

    Called from ``registry.py`` with the registry instance as parameter.
    """
    config_dir = DATA_DIR / "tau2" / "domains"
    if not config_dir.exists():
        return
    for config_path in sorted(config_dir.glob("*/domain_config.json")):
        try:
            config = EigenDomainConfig.load(config_path)
            get_env, get_tasks = make_domain_functions(config, registry)
            registry.register_domain(get_env, config.eigen_domain_name)
            registry.register_tasks(get_tasks, config.eigen_domain_name)
            logger.debug(
                f"Registered eigen domain '{config.eigen_domain_name}' "
                f"(base: {config.base_domain})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to register eigen domain from {config_path}: {e}"
            )
