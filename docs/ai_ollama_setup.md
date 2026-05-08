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

1. Install Ollama:

```bash
brew install ollama
```

2. Start Ollama:

```bash
ollama serve
```

3. Pull the demo model:

```bash
ollama pull llama3
```

4. Make sure Ollama is running:

```bash
ollama list
```

5. Set the model name for QuantForge:

```bash
export QUANTFORGE_LOCAL_LLM_MODEL=llama3
```

6. Run the curated demo modification:

```bash
quantforge modify strategy.qf.json --ai "Add a momentum filter with a short lookback and low threshold"
```

## Expected Failures

If Ollama is not running, the model is unavailable, or
`QUANTFORGE_LOCAL_LLM_MODEL` is not set, the planner should fail clearly and no
AI-assisted variant should be created.

Expected missing-model message:

```text
ERROR: Local AI planner unavailable. Configure a local model before using AI-assisted modifications.
```

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
trading advice, directional trade instructions, or performance guarantees.
