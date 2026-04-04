"""EigenDomainConfig — configuration for an eigen-registered domain."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class EigenDomainConfig:
    eigen_domain_name: str  # e.g., "airline_eigen"
    base_domain: str  # e.g., "airline" (must exist in tau2 registry)
    db_path: str  # path to our custom db.json
    tasks_path: str  # path to generated tasks.json
    env_name_in_payload: str  # e.g., "mcp_8001" (key in reference_payloads)
    source_dir: str  # original data dir for post-processing
    custom_policy_path: Optional[str] = None
    state_modifying_prefixes: Optional[dict[str, list[str]]] = field(
        default=None
    )  # per-sample prefix map {"000001": [...], ...}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EigenDomainConfig":
        with open(path) as f:
            data = json.load(f)
        config = cls(**data)
        # Resolve relative paths against DATA_DIR so the repo is portable.
        from tau2.utils.utils import DATA_DIR

        for attr in ("db_path", "tasks_path"):
            p = getattr(config, attr)
            if p and not Path(p).is_absolute():
                setattr(config, attr, str(DATA_DIR / p))
        return config
