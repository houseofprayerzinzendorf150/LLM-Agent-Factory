# Data and model artifacts

The GitHub repository is intentionally code-only. Large agent datasets, derived databases, and model checkpoints are hosted in the canonical [Hugging Face repository](https://huggingface.co/frontier-ai/llm-agent-factory/tree/main).

This keeps clones small, avoids duplicating LFS/Xet storage, and gives every consumer one authoritative artifact location.

## Artifact manifest

| Hugging Face path | Contents | Approximate size | Included on GitHub |
| --- | --- | ---: | :---: |
| [`task-agents_database/agents_eng.jsonl`](https://huggingface.co/frontier-ai/llm-agent-factory/blob/main/task-agents_database/agents_eng.jsonl) | Retrieval-ready English agent records | 157 MiB | No |
| [`agents_database/`](https://huggingface.co/frontier-ai/llm-agent-factory/tree/main/agents_database) | 692 domain-level JSON agent files | 9.8 MiB | No |
| [`agents_sort_database/`](https://huggingface.co/frontier-ai/llm-agent-factory/tree/main/agents_sort_database) | Derived and sorted agent JSON files | 11.7 MiB | No |
| [`alr-model/`](https://huggingface.co/frontier-ai/llm-agent-factory/tree/main/alr-model) | LoRA adapter, tokenizer, optimizer, and training state | 804 MiB | No |
| `config/` | Small role, domain, and tool vocabularies required by scripts | 16 KiB | Yes |

Sizes reflect Hugging Face revision [`505aa098`](https://huggingface.co/frontier-ai/llm-agent-factory/commit/505aa09857889bc679f2b914e2c33527051c37a8).

## Recommended: retrieval dataset only

Install the Hugging Face CLI and download the one file used by the default retrieval configuration:

```bash
python -m pip install -U huggingface_hub
hf download frontier-ai/llm-agent-factory task-agents_database/agents_eng.jsonl --local-dir .
```

The resulting path is:

```text
task-agents_database/agents_eng.jsonl
```

## Optional: source JSON databases

The curation and generation scripts work with the domain-level JSON database:

```bash
hf download frontier-ai/llm-agent-factory --include "agents_database/*.json" --local-dir .
```

Download the derived sorted database only when a workflow explicitly needs it:

```bash
hf download frontier-ai/llm-agent-factory --include "agents_sort_database/*.json" --local-dir .
```

## Optional: trained adapter

The training output is independent of the retrieval CLI and is not needed for ordinary search or RAG generation:

```bash
hf download frontier-ai/llm-agent-factory --include "alr-model/*" --local-dir .
```

Use `--dry-run` before a large download to inspect the selected files and total transfer size.

## Repository policy

The following paths are ignored by Git:

```text
agents_database/
agents_sort_database/
task-agents_database/
alr-model/
dataset/
train/
output/
experiments/checkpoints/
experiments/results/
```

Do not force-add these paths. Publish updated data or checkpoints to Hugging Face and update this manifest with their revision and expected layout.

