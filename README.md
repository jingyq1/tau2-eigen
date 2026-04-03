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
    --name <domain-name> \
    --db <path-to-db-json>
```

**Example:**

```bash
python -m tau2.eigen.register \
    --source "eigendataDB/tau2-airline" \
    --base-domain airline \
    --name eigendata-airline \
    --db eigendataDB//data-airline/db.json
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
