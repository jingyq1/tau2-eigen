# tau2-eigen

Simulation and evaluation of EigenData domains using the [tau2-bench](https://github.com/sierra-research/tau2-bench) framework.

## Installation

This project is built on top of tau2-bench. Installation and general usage follow the same instructions as the upstream project. See [tau-bench.md](tau-bench.md) for details.

## Usage

### 1. Register an EigenData domain

Convert EigenData source data into a tau2-bench domain:

```bash
python -m tau2.eigen.register \
    --source <path-to-eigendata-source> \
    --base-domain <base-domain-name> \
    --name <domain-name>
```

The database is automatically extracted from the `initial_state.database` in the reference payloads (validated for consistency across all samples).

**Example:**

```bash
python -m tau2.eigen.register \
    --source "eigendataDB/tau2-airline" \
    --base-domain airline \
    --name eigendata-airline
```

### 2. Run simulation

Run a tau2 simulation against the registered domain:

```bash
tau2 run --domain <domain-name> \
    --agent-llm <agent-model> \
    --user-llm <user-model> \
    --num-trials <n> --num-tasks <n>
```

**Example:**

```bash
tau2 run --domain eigendata-airline \
    --agent-llm openai/gpt-5.3-codex \
    --user-llm openai/gpt-5.2 \
    --num-trials 1 --num-tasks 5
```

### 3. View simulation results

Find the latest simulation output for a domain:

```bash
ls -td tau2-bench/data/simulations/*<domain-name>* | head -1
```

### 4. Run evaluation

Score the simulation results using the EigenData evaluator:

```bash
python -m tau2.eigen.postprocess \
    <path-to-results-json> \
    --name <domain-name> \
    -o <path-to-output-json>
```

**Example:**

```bash
python -m tau2.eigen.postprocess \
    tau2-bench/data/simulations/xxxxxxxx_xxxxxx_eigendata-airline_.../results.json \
    --name eigendata-airline \
    -o tau2-bench/data/simulations/xxxxxxxx_xxxxxx_eigendata-airline_.../results_scored.json
```

### 5. Batch evaluation

Run simulation + postprocess for multiple (model, domain) pairs from a single YAML config:

```bash
tau2 eval config.yaml
```

**Config file format:**

```yaml
defaults:
  user_llm: claude-opus-4-6
  num_trials: 1
  eval_llm: gpt-4.1
  max_concurrency: 5

max_workers: 4  # parallel eval processes

evals:
  - domain: eigendata-airline
    agent_llm: openai/gpt-5.3-codex
  - domain: eigendata-airline
    agent_llm: anthropic/claude-sonnet-4-6
  - domain: eigendata-retail
    agent_llm: openai/gpt-5.3-codex
    num_tasks: 10  # per-entry override
```

**Options:**

```bash
# Run only the simulation step (skip postprocess)
tau2 eval config.yaml --only run

# Run only the postprocess step (skip simulation)
tau2 eval config.yaml --only postprocess

# Run a single entry by index (0-indexed)
tau2 eval config.yaml --entry 0

# Override max parallel workers
tau2 eval config.yaml --max-workers 2
```

After completion, a summary table is printed with per-domain metrics (Config Match, Key Func., LLM Judge, etc.).
