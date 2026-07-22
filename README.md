# LLM Agent Factory

**Generate, retrieve, and evaluate structured AI agents with semantic search and RAG.**

[![CI](https://github.com/frontier-ai-next/LLM-Agent-Factory/actions/workflows/ci.yml/badge.svg)](https://github.com/frontier-ai-next/LLM-Agent-Factory/actions/workflows/ci.yml)
[![GitHub tag](https://img.shields.io/github/v/tag/frontier-ai-next/LLM-Agent-Factory?display_name=tag&sort=semver)](https://github.com/frontier-ai-next/LLM-Agent-Factory/tags)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/frontier-ai/llm-agent-factory)

[Hugging Face artifacts](https://huggingface.co/frontier-ai/llm-agent-factory) · [Data guide](DATASETS.md) · [Extended quick start](QUICK_START.md) · [gMAS](https://github.com/frontier-ai-next/gMAS)

LLM Agent Factory turns a natural-language request into a reusable agent specification. It can search an existing agent library, retrieve the closest examples, and ask an OpenAI-compatible LLM to generate one or more adapted agents.

Each agent includes a stable identifier, display name, persona, description, role, domain, and tool set.

> **Repository split:** GitHub contains the source code, tests, configurations, and documentation. The large agent datasets, training outputs, and model artifacts remain on Hugging Face and are downloaded only when needed.

## Core capabilities

| Area | What it provides |
| --- | --- |
| Retrieval | Semantic agent search with BGE, MiniLM, MPNet, E5, or a TF-IDF fallback |
| Reranking | Optional cross-encoder reranking for higher-quality results |
| Generation | RAG-based agent creation through OpenAI-compatible endpoints |
| Agent tooling | Scripts for generation, curation, deduplication, task creation, and SFT preparation |
| Evaluation | Reproducible experiment runner, checkpoints, metrics, and benchmark scenarios |
| Multi-agent runtime | The bundled `gmas-main` framework used by the experiment harness |

## Install

LLM Agent Factory requires Python 3.12 or newer.

```bash
git clone https://github.com/frontier-ai-next/LLM-Agent-Factory.git
cd LLM-Agent-Factory
python -m pip install -e ".[rag]"
```

Copy the environment template when generation through an LLM is required:

```bash
cp env.example .env
```

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4
```

## Download the agent dataset

The default retriever expects `task-agents_database/agents_eng.jsonl`. Download it directly from the canonical Hugging Face release:

```bash
python -m pip install -U huggingface_hub
hf download frontier-ai/llm-agent-factory task-agents_database/agents_eng.jsonl --local-dir .
```

The directory is intentionally ignored by Git. See [DATASETS.md](DATASETS.md) for the complete artifact inventory, sizes, optional downloads, and provenance.

## Quick start

Search the existing agent library:

```bash
agent-search -q "Python code review specialist" -k 5
```

Generate a new agent with retrieval-augmented context:

```bash
agent-generate "Create an agent that reviews Python APIs for security issues"
```

Generate several variants:

```bash
agent-generate --agents 3 --format pretty "Customer support specialist"
```

Programmatic usage:

```python
from retrieval import quick_generate, quick_search

results = quick_search("machine learning research assistant", top_k=3)
for result in results:
    print(result.agent.display_name, result.score)

agents = quick_generate(
    "A Python code-review assistant",
    api_key="your-api-key",
)
print(agents[0]["display_name"])
```

## How it works

1. The loader reads the downloaded agent records.
2. An embedding model builds or restores a local semantic index.
3. Retrieval selects the closest agent examples and can rerank them.
4. The RAG layer combines those examples with the request.
5. The configured LLM returns a validated, structured agent specification.

The first retrieval run builds an index in `retrieval/.cache/`; later runs reuse it.

## Repository layout

```text
LLM-Agent-Factory/
├── config/          # Small role, domain, and tool vocabularies
├── experiments/     # Evaluation runner, metrics, and checkpoints
├── gmas-main/       # Multi-agent runtime used by experiments
├── retrieval/       # Search, embeddings, reranking, RAG, CLI, and tests
├── script/          # Dataset generation, curation, deduplication, and training scripts
├── DATASETS.md      # External artifact manifest and download commands
├── examples.py      # Programmatic examples
└── pyproject.toml   # Package metadata and dependencies
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest retrieval/tests -v
```

Large or generated artifacts must stay outside Git history. CI checks this boundary on every push and pull request.

## Release provenance

This source release was migrated from Hugging Face revision [`505aa098`](https://huggingface.co/frontier-ai/llm-agent-factory/commit/505aa09857889bc679f2b914e2c33527051c37a8). The canonical heavy artifacts remain attached to that Hugging Face repository.

The source release metadata declares the project under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
