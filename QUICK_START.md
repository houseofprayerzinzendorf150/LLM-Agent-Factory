# Quick Start Guide

Quick guide to using LLM Agent Factory.

> The large agent databases are hosted on Hugging Face and are not committed to GitHub. Download the required files first by following [DATASETS.md](DATASETS.md).

## Installation in 2 Minutes

```bash
# 1. Clone repository
git clone <repository-url>
cd LLM-Agent-Factory

# 2. Install dependencies
pip install -e ".[rag]"
```

## Usage in 30 Seconds

### 🔍 Agent Search (Retrieval)

Retrieval is semantic search for existing agents in the database.

```bash
# Launch interactive search
agent-search

# Or single query
agent-search -q "Python programming help"
```

### 🤖 Agent Generation (RAG)

RAG is generation of a new unique agent using LLM based on similar agents.

```bash
# Launch interactive generation
agent-generate

# Or generate immediately
agent-generate "I need a code review assistant"
```

## 📊 Datasets

The system has 1 agent dataset available:

| Dataset | Language | Agents | Description |
|---------|----------|--------|-------------|
| `eng` | English | ~18K | Main English dataset (default) |
| `all` | English | ~18K | All datasets together (same as eng) |

```bash
# Using different datasets
agent-search -d eng -q "query"      # English (default)
agent-search -d all -q "query"      # All together (same as eng)
```

## 🎯 Query Examples

### For Search (agent-search)

```bash
# English dataset
agent-search -q "machine learning expert"
agent-search -q "data visualization specialist"
agent-search -q "API documentation writer"

# English dataset (default)
agent-search -d eng -q "programming assistant"
agent-search -d eng -q "text translator"

# All datasets with multilingual model
agent-search -d all --model bge-m3 -q "programming"

# With reranking for better quality
agent-search --rerank -q "senior software architect"

# More results
agent-search -q "Python expert" -k 10
```

### For Generation (agent-generate)

```bash
# One agent
agent-generate "customer support specialist for e-commerce"

# Multiple variants
agent-generate --agents 3 "content marketing manager"

# Pretty output instead of JSON
agent-generate --format pretty "data analyst for healthcare"

# In English (default)
agent-generate -d eng "data analysis agent"

# More examples for context = better quality
agent-generate --examples 10 "blockchain developer"

# Configure LLM temperature (higher = more creative)
agent-generate --temperature 0.9 "creative writing assistant"
```

## ⚙️ Settings and Flags

### Common Flags for Both Commands

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset (eng, all) | `eng` |
| `--model` | | Embedding model | `bge-small` |
| `--rerank` | | Enable reranking | `false` |
| `--help` | `-h` | Show help | |

### Flags for agent-search

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--query` | `-q` | Search query | (interactive mode) |
| `--topk` | `-k` | Number of results | `5` |
| `--format` | | Output format (text, json) | `text` |
| `--verbose` | `-v` | Verbose output | `false` |
| `--rerank-model` | | Reranker model | `bge-reranker-base` |

### Flags for agent-generate

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--agents` | `-n` | Number of agents to generate | `1` |
| `--examples` | `-e` | Examples for context | `5` |
| `--format` | `-f` | Output format (json, pretty) | `json` |
| `--model` | | LLM model | `gpt-oss` |
| `--url` | | LLM API URL | (configured) |
| `--api-key` | | API key | (configured) |
| `--temperature` | `-t` | Generation temperature (0.0-1.0) | `0.7` |

## 🎮 Interactive Mode

### agent-search (Interactive)

Run without parameters for interactive mode:

```bash
agent-search
```

**Commands in interactive mode:**

| Command | Description |
|---------|-------------|
| `<text>` | Search agents (just enter query) |
| `/switch <dataset>` | Switch dataset (eng, all) |
| `/topk <N>` | Change number of results |
| `/rerank` | Enable/disable reranking |
| `/stats` | Show dataset statistics |
| `/verbose` | Toggle verbose output |
| `/help` | Show help |
| `/quit` | Exit |

**Example session:**

```
> Python expert
[Shows 5 results]

> /topk 10
✓ Set top_k to 10

> /rerank
✓ Reranking enabled

> machine learning specialist
[Shows 10 results with reranking]

> /switch all
✓ Switched to dataset: all

> data specialist
[Shows results from all datasets]

> /quit
```

### agent-generate (Interactive)

Run without parameters for interactive mode:

```bash
agent-generate
```

**Commands in interactive mode:**

| Command | Description |
|---------|-------------|
| `<text>` | Generate agent (just enter description) |
| `generate <query>` | Explicitly generate agent |
| `search <query>` | Only search for similar (without generation) |
| `dataset <name>` | Switch dataset (eng, all) |
| `agents <N>` | Number of agents to generate (1-10) |
| `examples <N>` | Examples for context (1-20) |
| `format <type>` | Output format (json, pretty) |
| `stats` | Show current configuration |
| `help` | Show help |
| `quit` | Exit |

**Example session:**

```
> code review assistant
[Generates agent in JSON]

> format pretty
✓ Set output format: pretty

> agents 3
✓ Set agents to generate: 3

> customer support specialist
[Generates 3 agent variants in pretty format]

> search Python expert
[Shows similar agents without generation]

> dataset all
✓ Switched to dataset: all

> programming assistant
[Generates agent based on all datasets]

> quit
```

## 💻 Programmatic Usage

### Simple Search (Quick API)

```python
from retrieval import quick_search

# Simplest way
results = quick_search("Python programming expert")

for result in results:
    print(f"{result.agent.display_name}: {result.agent.description}")
```

### Simple Generation (Quick API)

```python
from retrieval import quick_generate

# Simplest way (requires API key)
agents = quick_generate(
    "code review assistant for Python",
    api_key="your-api-key"
)

print(agents[0]["display_name"])
print(agents[0]["description"])
```

### Advanced Search

```python
from retrieval import AgentRetriever, RetrievalConfig, DatasetType

# Create configuration
config = RetrievalConfig(
    dataset_type=DatasetType.ENG,
    top_k=5,
    use_reranker=True,  # Two-stage search
)

# Create retriever
retriever = AgentRetriever(config)
retriever.initialize()

# Search
results = retriever.search("Python expert")

for result in results:
    print(f"{result.agent.display_name}: {result.score:.4f}")
```

### Advanced Generation

```python
from retrieval import AgentRAG, RAGConfig, LLMConfig, DatasetType

# Configure LLM
llm = LLMConfig(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    temperature=0.7,
)

# Create RAG configuration
config = RAGConfig.with_dataset(
    dataset_type=DatasetType.ENG,
    llm=llm,
    num_agents_to_return=1,
    num_retrieved_for_context=5,
)

# Create RAG
rag = AgentRAG(config)
rag.initialize()

# Generation
agents = rag.generate("code review assistant")
print(agents[0])
```

## 🎨 Embedding Models

| Model | Dimensions | When to Use |
|-------|------------|-------------|
| `bge-small` | 384 | **Default** - balance of speed and quality |
| `bge-base` | 768 | High quality |
| `bge-large` | 1024 | Maximum quality |
| `bge-m3` | 1024 | **Multilingual** (100+ languages) |
| `minilm` | 384 | Fast, basic quality |
| `mpnet` | 768 | Medium speed/quality |

```bash
# Usage
agent-search --model bge-large -q "query"
agent-search --model bge-m3 -d all -q "multilingual query"
```

## 🔄 Reranking (Two-stage Retrieval)

Reranking improves search quality but is slower:

1. **Stage 1:** Fast candidate search (bi-encoder)
2. **Stage 2:** Accurate ranking (cross-encoder)

```bash
# Enable reranking
agent-search --rerank -q "complex query"

# With reranker model selection
agent-search --rerank --rerank-model bge-reranker-large -q "query"
```

## 🔧 LLM Configuration

For agent generation, an LLM with OpenAI-compatible API is needed.

### Via Environment Variables

Create `.env` file:

```bash
LLM_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
```

### Via Command Line Parameters

```bash
agent-generate \
  --model "gpt-4" \
  --url "https://api.openai.com/v1" \
  --api-key "your-key" \
  --temperature 0.7 \
  "your query"
```

### In Code

```python
from retrieval import quick_generate

agents = quick_generate(
    "code review assistant",
    model="gpt-4",
    api_key="your-key",
    base_url="https://api.openai.com/v1"
)
```

## 💡 Tips

1. **First run is slow** (~1-2 min) — embedding index is built, cached for subsequent runs

2. **For multilingual** — use `--model bge-m3` and `-d all`

3. **For better search quality** — enable `--rerank`

4. **More examples = better generation** — use `--examples 10-15` for RAG

5. **Different agent variants** — increase temperature: `--temperature 0.9`

6. **Use all datasets** — use `-d all` to search across all available datasets

7. **Use Quick API** — for quick start in code:
   ```python
   from retrieval import quick_search, quick_generate
   ```

## ❓ Problems?

```bash
# Error "Dataset not found"
# → Make sure agents_database/ folder exists with files:
#   - agents_eng.jsonl

# Slow performance
# → Use lighter model:
agent-search --model minilm -q "query"

# Poor search quality
# → Try reranking or more powerful model:
agent-search --rerank --model bge-large -q "query"

# Generation not working
# → Check LLM settings (model, url, api-key)
agent-generate --help

# CUDA out of memory
# → Use CPU:
agent-search --device cpu -q "query"
```

## 📚 What's Next?

- **Full documentation:** [README.md](README.md)
- **Code examples:** [examples.py](examples.py)
- **Installation:** [INSTALLATION.md](INSTALLATION.md)
- **Project structure:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **API documentation:** [retrieval/README.md](retrieval/README.md)

---

**Done!** Now you can search and generate agents 🚀

**Key concepts:**
- **Retrieval** = Search for existing agents in database
- **RAG** = Generate new agents using LLM
- **Reranking** = Two-stage search for better quality
- **Quick API** = Simple way to use in code
