"""CLI to register a new eigen domain from a source data directory.

Usage:
    python -m tau2.eigen.register \
        --source "Archive 2/tau2-airline" \
        --base-domain airline \
        --name airline_eigen \
        --db mcp_deploy/airline_mcp/data/db_updated.json
"""

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

from loguru import logger

from tau2.eigen.config import EigenDomainConfig
from tau2.eigen.convert import convert_intents_to_tasks
from tau2.eigen.domain_factory import load_db_for_domain, validate_db_for_domain
from tau2.utils.utils import DATA_DIR


# ---------------------------------------------------------------------------
# Default DB paths per base domain
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATHS = {
    "airline": DATA_DIR / "tau2" / "domains" / "airline" / "db.json",
    "retail": DATA_DIR / "tau2" / "domains" / "retail" / "db.json",
    "banking_knowledge": DATA_DIR / "tau2" / "domains" / "banking_knowledge" / "db.json",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_sample_alignment(source_dir: str | Path) -> list[str]:
    """Verify intent/, reference_payloads/, evaluators/ have identical sample IDs.

    Returns sorted list of validated sample IDs.
    """
    source_dir = Path(source_dir)
    intent_ids = {f.stem for f in (source_dir / "intent").glob("*.json")}
    ref_ids = {
        f.stem.replace("reference_payload_", "")
        for f in (source_dir / "reference_payloads").glob("reference_payload_*.json")
    }
    eval_ids = {
        f.stem.replace("evaluator_", "")
        for f in (source_dir / "evaluators").glob("evaluator_*.py")
    }

    if intent_ids == ref_ids == eval_ids:
        return sorted(intent_ids)

    all_ids = intent_ids | ref_ids | eval_ids
    lines = []
    for label, ids in [
        ("intent", intent_ids),
        ("reference_payloads", ref_ids),
        ("evaluators", eval_ids),
    ]:
        missing = all_ids - ids
        if missing:
            lines.append(f"  {label}: missing {sorted(missing)}")
    raise ValueError(
        "Sample ID mismatch across subdirectories:\n" + "\n".join(lines)
    )


def validate_source_dir(
    db_path: str | Path, ref_payload_dir: str | Path, base_domain: str
) -> tuple[str, object, bool]:
    """Validate ALL reference_payloads and return env_name plus resolved DB.

    Three-phase check:
    1. Collect all env_names and init DB hashes (enforce single-env per payload).
    2. Internal consistency (all env_names same, all init DB hashes same).
    3. External consistency (prefer selected --db when its hash matches;
       otherwise fall back to the payload initial_state DB).
    """
    ref_payload_dir = Path(ref_payload_dir)
    ref_files = sorted(ref_payload_dir.glob("reference_payload_*.json"))
    if not ref_files:
        raise ValueError(f"No reference_payload files found in {ref_payload_dir}")

    all_env_names: list[str] = []
    all_init_hashes: list[str] = []
    payload_db = None

    for ref_file in ref_files:
        with open(ref_file) as f:
            ref_data = json.load(f)
        snapshots = ref_data["environment_snapshots"]
        if len(snapshots) != 1:
            raise ValueError(
                f"{ref_file.name} has {len(snapshots)} environment snapshots "
                f"({list(snapshots.keys())}). Eigen pipeline requires exactly one "
                f"environment per payload."
            )
        env_name = next(iter(snapshots))
        all_env_names.append(env_name)
        init_db_dict = snapshots[env_name]["initial_state"]["database"]
        init_db = validate_db_for_domain(base_domain, init_db_dict)
        if payload_db is None:
            payload_db = init_db
        all_init_hashes.append(init_db.get_hash())

    # Phase 2a: env_names
    unique_env_names = set(all_env_names)
    if len(unique_env_names) > 1:
        raise ValueError(
            f"Mixed env_names in source directory: {unique_env_names}. "
            f"A single source directory must use one consistent environment name."
        )

    # Phase 2b: init DB hashes
    unique_init_hashes = set(all_init_hashes)
    if len(unique_init_hashes) > 1:
        raise ValueError(
            f"Source directory contains {len(unique_init_hashes)} distinct initial DBs! "
            f"A single source directory must use one consistent initial DB."
        )

    # Phase 3: external consistency
    db = load_db_for_domain(base_domain, str(db_path))
    db_hash = db.get_hash()
    init_hash = all_init_hashes[0]
    if db_hash != init_hash:
        logger.warning(
            f"DB mismatch! Selected DB hash ({db_hash[:16]}...) != "
            f"reference_payload initial_state hash ({init_hash[:16]}...). "
            "Falling back to the initial_state database embedded in the "
            "reference payloads for registration."
        )
        return all_env_names[0], payload_db, True

    return all_env_names[0], db, False


# ---------------------------------------------------------------------------
# Prefix extraction
# ---------------------------------------------------------------------------


def _parse_prefixes_from_source(eval_file: Path) -> list[str]:
    """Extract the modifying_prefixes list from an evaluator .py file."""
    source = eval_file.read_text()
    # Find the assignment: modifying_prefixes = [...]
    match = re.search(
        r"modifying_prefixes\s*=\s*\[([^\]]*)\]", source, re.DOTALL
    )
    if not match:
        return []
    list_str = "[" + match.group(1) + "]"
    try:
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        return []


def extract_per_sample_prefixes(
    evaluator_dir: str | Path, sample_ids: list[str]
) -> dict[str, list[str]]:
    """Extract state_modifying_prefixes per sample from evaluator files."""
    evaluator_dir = Path(evaluator_dir)
    result = {}
    for sid in sample_ids:
        eval_file = evaluator_dir / f"evaluator_{sid}.py"
        prefixes = _parse_prefixes_from_source(eval_file)
        result[sid] = sorted(prefixes)
    return result


# ---------------------------------------------------------------------------
# Main register flow
# ---------------------------------------------------------------------------


def register(
    source: str,
    base_domain: str,
    name: str,
    db: str | None = None,
) -> EigenDomainConfig:
    """Register a new eigen domain.

    Returns the saved EigenDomainConfig.
    """
    source_dir = Path(source)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Step 1: sample-set alignment
    logger.info("Validating sample alignment...")
    sample_ids = validate_sample_alignment(source_dir)
    logger.info(f"Found {len(sample_ids)} aligned samples: {sample_ids}")

    # Step 2: resolve DB path
    if db is None:
        if base_domain not in _DEFAULT_DB_PATHS:
            raise ValueError(
                f"No default DB path for base domain '{base_domain}'. "
                f"Please provide --db explicitly."
            )
        db_path = _DEFAULT_DB_PATHS[base_domain]
        logger.info(f"No --db provided, using base domain default: {db_path}")
    else:
        db_path = Path(db)
        if not db_path.is_file():
            raise ValueError(f"DB file does not exist: {db_path}")

    # Step 3: full-scan validation (env_name, DB consistency)
    logger.info("Validating source directory (full scan)...")
    env_name, resolved_db, used_payload_db = validate_source_dir(
        db_path, source_dir / "reference_payloads", base_domain
    )
    logger.info(f"Validated: env_name={env_name}, DB consistent across all payloads")

    # Step 4: create output directory
    domain_dir = DATA_DIR / "tau2" / "domains" / name
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: copy DB, reference_payloads, and evaluators
    dest_db = domain_dir / "db.json"
    resolved_db.dump(str(dest_db))
    if used_payload_db:
        logger.info(f"Wrote payload-derived DB to {dest_db}")
    else:
        logger.info(f"Copied validated DB to {dest_db}")

    for subdir in ("reference_payloads", "evaluators"):
        src = source_dir / subdir
        dest = domain_dir / subdir
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(src), str(dest))
            logger.info(f"Copied {subdir}/ to {dest}")

    # Step 6: extract per-sample prefixes
    logger.info("Extracting per-sample state_modifying_prefixes...")
    prefixes = extract_per_sample_prefixes(source_dir / "evaluators", sample_ids)

    # Step 7: convert intents → tasks.json
    tasks_path = domain_dir / "tasks.json"
    logger.info("Converting intents to tasks.json...")
    tasks = convert_intents_to_tasks(
        source_dir / "intent", base_domain, sample_ids, tasks_path
    )
    logger.info(f"Wrote {len(tasks)} tasks to {tasks_path}")

    # Step 8: write domain_config.json
    config = EigenDomainConfig(
        eigen_domain_name=name,
        base_domain=base_domain,
        db_path=str(dest_db),
        tasks_path=str(tasks_path),
        env_name_in_payload=env_name,
        source_dir=str(source_dir),
        custom_policy_path=None,
        state_modifying_prefixes=prefixes,
    )
    config_path = domain_dir / "domain_config.json"
    config.save(config_path)
    logger.info(f"Wrote config to {config_path}")

    logger.info(
        f"Domain '{name}' registered successfully. "
        f"Run: tau2 run --domain {name} --agent-llm <model> --user-llm <model>"
    )
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Register a new eigen domain from a source data directory."
    )
    parser.add_argument(
        "--source", required=True, help="Path to source data directory (e.g., 'Archive 2/tau2-airline')"
    )
    parser.add_argument(
        "--base-domain", required=True, help="Base tau2 domain name (e.g., 'airline', 'retail', 'banking_knowledge')"
    )
    parser.add_argument(
        "--name", required=True, help="Name for the new eigen domain (e.g., 'airline_eigen')"
    )
    parser.add_argument(
        "--db", default=None, help="Path to custom DB file. If omitted, uses base domain's default."
    )
    args = parser.parse_args()
    register(source=args.source, base_domain=args.base_domain, name=args.name, db=args.db)


if __name__ == "__main__":
    main()
