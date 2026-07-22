# Agent Retrieval & RAG System

Semantic search and RAG generation of agents for LLM Agent Factory. The system uses embedding-based retrieval with optional cross-encoder reranking to find the most suitable agents, as well as RAG to generate new agents based on queries.

> The large agent databases are hosted on Hugging Face and are not committed to GitHub. See the repository-level [data guide](../DATASETS.md) for download commands and artifact provenance.

## Features

- **Pydantic models** for validation and serialization
- **PyTorch tensors** for efficient work with embeddings and GPU
- **Multiple embedding models** (BGE, MiniLM, E5, multilingual)
- **Cross-encoder reranking** for improved accuracy (two-stage retrieval)
- **1 dataset** (eng) or all together
- **Index caching** for fast subsequent runs
- **RAG generation** of new agents using LLM

## Installation

### Basic Installation (TF-IDF fallback)
```bash
pip install -e ".[retrieval-light]"
```

### Full Installation (Sentence Transformers - recommended)
```bash
pip install -e ".[retrieval]"
```

### For Development
```bash
pip install -e ".[retrieval,dev]"
```

### With RAG (agent generation via LLM)
```bash
pip install -e ".[rag]"
```

## Usage

### Interactive Mode

```bash
# Run with English dataset (default)
python -m retrieval.cli

# With English dataset (default)
python -m retrieval.cli -d eng

# Use all datasets together
python -m retrieval.cli -d all
```

### Single Query

```bash
# Text output
python -m retrieval.cli -q "I need help with coding" -k 3

# JSON output
python -m retrieval.cli -q "math tutor" --format json

# Verbose output
python -m retrieval.cli -q "writing assistant" -v

# With reranking for better accuracy
python -m retrieval.cli -q "python programming" --rerank
```

### Choosing Embedding Model

```bash
# BGE small (default, recommended)
python -m retrieval.cli --model bge-small -q "coding"

# BGE large (more accurate, but slower)
python -m retrieval.cli --model bge-large -q "coding"

# For multilingual (English)
python -m retrieval.cli --model bge-m3 -d all -q "programming"
```

### Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `/switch <dataset>` | Switch dataset (eng, all) |
| `/topk <n>` | Change number of results |
| `/rerank` | Enable/disable reranking |
| `/stats` | Show dataset statistics |
| `/verbose` | Toggle verbose output |
| `/help` | Show help |
| `/quit` | Exit program |

## Embedding Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| `minilm` | 384 | Fast, basic quality |
| `mpnet` | 768 | Medium speed/quality |
| `bge-small` | 384 | **Recommended** - good balance |
| `bge-base` | 768 | High quality |
| `bge-large` | 1024 | Maximum quality |
| `bge-m3` | 1024 | Multilingual (100+ languages) |
| `e5-multi` | 1024 | Multilingual |

## Reranker Models

| Model | Description |
|-------|-------------|
| `bge-reranker-base` | Default, good balance |
| `bge-reranker-large` | More accurate |
| `bge-reranker-v2-m3` | Multilingual |
| `msmarco-minilm` | MS MARCO trained |

> **Note:** Agent database is located in `agents_database/` folder at project root.

## Programmatic Usage

```python
from retrieval import AgentRetriever, RetrievalConfig, DatasetType, EmbeddingModel

# Create configuration
config = RetrievalConfig(
    dataset_type=DatasetType.ENG,
    embedding_model=EmbeddingModel.BGE_SMALL.value,
    top_k=5,
    use_reranker=True,  # Enable two-stage retrieval
    device="auto",  # auto, cuda, cpu, mps
)

# Create retriever
retriever = AgentRetriever(config)

# Search agents
results = retriever.search("I need help with Python programming")

for result in results:
    print(f"{result.rank}. {result.agent.display_name}")
    print(f"   Score: {result.score:.4f}")
    if result.rerank_score:
        print(f"   Rerank: {result.rerank_score:.4f}")
    print(f"   {result.agent.description}")
```

### Working with Pydantic Models

```python
from retrieval import AgentSpec, RetrievalResult

# AgentSpec - agent specification
agent = AgentSpec(
    agent_id="my_agent",
    display_name="My Agent",
    persona="A helpful assistant",
    description="Helps with various tasks",
    tools=["web_search", "calculator"],
)

# Serialize to JSON
agent_json = agent.model_dump_json()

# Get text for indexing
indexable_text = agent.get_indexable_text()
```

### Switching Datasets

```python
# Switch to another dataset
retriever.switch_dataset(DatasetType.ALL)
retriever.initialize()

# Search in new dataset
results = retriever.search("help with math")
```

## Architecture

```
retrieval/
├── __init__.py          # Public API
├── cli.py               # CLI for retrieval
├── config.py            # Retrieval configuration (Pydantic)
├── models.py            # Pydantic models (AgentSpec, RetrievalResult)
├── data_loader.py       # JSONL data loading
├── embedder.py          # Embeddings + reranker (PyTorch)
├── retriever.py         # AgentRetriever (semantic search)
├── rag.py               # AgentRAG (generation via LLM)
├── rag_config.py        # RAG configuration (LLMConfig, RAGConfig)
├── rag_cli.py           # CLI for RAG
├── README.md            # Documentation
└── tests/
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_retriever.py
    └── test_rag.py      # RAG system tests
```

## Two-Stage Retrieval

The system supports two-stage retrieval:

1. **Stage 1: Bi-encoder retrieval** (fast)
   - Uses sentence-transformer model
   - All documents are indexed in advance
   - Query is encoded and compared with index
   - Returns top-N candidates

2. **Stage 2: Cross-encoder reranking** (accurate)
   - Uses cross-encoder model
   - Each (query, document) pair is evaluated separately
   - Reranks candidates by accuracy
   - Returns final top-K results

---

## RAG (Retrieval-Augmented Generation)

RAG system allows generating new agents based on user queries, using found similar agents as context for LLM.

### How RAG Works

1. **Retrieval**: Search for similar agents in database (using embedding-based search)
2. **Augmentation**: Add found agents as examples to prompt
3. **Generation**: LLM generates new agent based on query and examples

### Quick Start with RAG

#### Interactive Mode

```bash
# Launch interactive mode (English dataset by default)
python -m retrieval.rag_cli

# With English dataset (default)
python -m retrieval.rag_cli --dataset eng

# With all datasets
python -m retrieval.rag_cli --dataset all
```

#### Single Query

```bash
# Generate one agent
python -m retrieval.rag_cli "I need a code review assistant"

# Generate multiple agents
python -m retrieval.rag_cli --agents 3 "customer support agent"

# With pretty formatting instead of JSON
python -m retrieval.rag_cli --format pretty "data analysis helper"

# With English dataset (default)
python -m retrieval.rag_cli --dataset eng "help me write poetry"
```

### RAG Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `generate <query>` | Generate agent by query |
| `search <query>` | Only search for similar agents (without generation) |
| `dataset <name>` | Switch dataset (eng, all) |
| `agents <N>` | Set number of agents to generate (1-10) |
| `examples <N>` | Set number of examples for context (1-20) |
| `format <type>` | Output format (json, pretty) |
| `stats` | Show current configuration |
| `help` | Show help |
| `quit` | Exit |

> **Tip**: Any text without a command is automatically processed as `generate`.

### RAG CLI Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset (eng, all) | eng |
| `--agents` | `-n` | Number of agents to generate | 1 |
| `--examples` | `-e` | Number of examples for context | 5 |
| `--format` | `-f` | Output format (json, pretty) | json |
| `--model` | | LLM model | gpt-oss |
| `--url` | | LLM API URL | (configured) |
| `--api-key` | | API key | (configured) |
| `--temperature` | `-t` | Generation temperature | 0.7 |

### Programmatic RAG Usage

```python
from retrieval import AgentRAG, RAGConfig, LLMConfig, DatasetType

# Configure LLM
llm_config = LLMConfig(
    model="gpt-oss",
    base_url="https://your-llm-api.com/v1",
    api_key="your-api-key",
    temperature=0.7,
)

# Create RAG configuration
config = RAGConfig.with_dataset(
    dataset_type=DatasetType.ENG,
    llm=llm_config,
    num_agents_to_return=1,      # How many agents to generate
    num_retrieved_for_context=5,  # How many examples to use
)

# Create RAG
rag = AgentRAG(config)

# Generate agent
agents = rag.generate("I need a code review assistant")
print(agents[0])
# {
#   "agent_id": "code_review_assistant",
#   "display_name": "Code Review Assistant",
#   "persona": "A meticulous senior developer...",
#   "description": "Analyzes submitted code...",
#   "tools": []
# }
```

### Search Only (Without Generation)

```python
# Search for similar agents without calling LLM
results = rag.search_only("Python programming tutor", top_k=5)

for result in results:
    print(f"{result.rank}. {result.agent.display_name} (score: {result.score:.4f})")
```

### Runtime Dataset Switching

```python
from retrieval import DatasetType

# Switch to all datasets
rag.switch_dataset(DatasetType.ALL)

# Generate agents
agents = rag.generate("I need a programming assistant")
```

### Generating Multiple Agents

```python
# Generate 3 different agents
agents = rag.generate("customer support agent", num_agents=3)

for i, agent in enumerate(agents, 1):
    print(f"Agent {i}: {agent['display_name']}")
```

### Output Formatting

```python
from retrieval import format_agent_output

agents = rag.generate("data analyst")

# JSON format
json_output = format_agent_output(agents, "json")
print(json_output)

# Pretty format
pretty_output = format_agent_output(agents, "pretty")
print(pretty_output)
```

### RAG Statistics

```python
# Get statistics
stats = rag.stats
print(f"Dataset: {stats['dataset_type']}")
print(f"Total records: {stats['total_records']}")
print(f"Unique agents: {stats['unique_agents']}")
print(f"LLM model: {stats['llm_model']}")
```

---

## Testing

```bash
# Run all tests
pytest retrieval/tests/ -v

# Only retrieval tests
pytest retrieval/tests/test_retriever.py -v

# Only RAG tests
pytest retrieval/tests/test_rag.py -v

# With coverage
pytest retrieval/tests/ --cov=retrieval --cov-report=term-missing
```

## FAQ

### Retrieval

**Q: Is chunking needed?**

A: No, agent descriptions are short (< 200 tokens), chunking is needed for long documents.

**Q: Which embedding model should I choose?**

A: Start with `bge-small`. For production use `bge-base` or `bge-large`. For multilingual use `bge-m3`.

**Q: Should I use reranker?**

A: Yes, if accuracy is important. Reranking increases response time but improves quality.

**Q: Which device to use?**

A: `device="auto"` automatically selects GPU (CUDA/MPS) if available, otherwise CPU.

### RAG

**Q: What's the difference between retrieval and RAG?**

A: Retrieval (`python -m retrieval.cli`) searches for existing agents in database. RAG (`python -m retrieval.rag_cli`) generates new agents using LLM, using found ones as examples.

**Q: Which dataset is better for RAG?**

A: Use `eng` for English queries (default), `all` for all datasets (same as eng).

**Q: How many examples to use?**

A: Default is 5 examples. More examples = better context, but longer response and more tokens.

**Q: How to configure my own LLM?**

A: Pass parameters via CLI (`--model`, `--url`, `--api-key`) or create `LLMConfig` in code.

**Q: What to do if first run is slow?**

A: On first run, embedding index is built (~2 minutes). It's cached in `retrieval/.cache/` and subsequent runs are fast.
