# AI-Assisted Modification Setup

QuantForge is local-first. AI-assisted modification planning uses a local
Ollama model selected with `QUANTFORGE_LOCAL_LLM_MODEL`.

The AI planner does not generate executable strategy code. The workflow is:

```text
user prompt -> local LLM -> validated JSON spec -> deterministic builder -> auditable variant
```

QuantForge evaluates strategy behavior and tradeoffs; it does not recommend
trades or optimize strategies.

## Setup

1. Install Ollama from the official Ollama distribution for your system.
2. Pull a local model:

```bash
ollama pull <model-name>
```

3. Make sure Ollama is running:

```bash
ollama list
```

4. Set the model name for QuantForge:

```bash
export QUANTFORGE_LOCAL_LLM_MODEL=<model-name>
```

5. Run an AI-assisted modification:

```bash
quantforge modify strategy.qf.json --ai "add a momentum confirmation filter"
```

## Expected Failures

If Ollama is not running, the model is unavailable, or
`QUANTFORGE_LOCAL_LLM_MODEL` is not set, the planner should fail clearly and no
AI-assisted variant should be created.

Non-AI workflows still work without Ollama:

```bash
quantforge create
quantforge analyze
quantforge modify strategy.qf.json "Add a volatility filter"
quantforge compare
```

## Boundary

AI assistance only proposes a structured modification specification. QuantForge
then validates the JSON spec and uses deterministic builders to create visible,
auditable variant artifacts. This keeps assumptions, parameters, warnings, and
diffs inspectable.

AI-assisted changes are experiments on user-provided hypotheses. They are not
trading advice, buy/sell recommendations, or performance guarantees.
