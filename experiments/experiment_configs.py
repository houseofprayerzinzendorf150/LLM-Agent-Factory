"""
Experiment configurations: 4-5 retrieval configs + 4-5 RAG configs.

Each config defines a unique combination of embedding model, reranker,
dataset, and other parameters for systematic benchmarking.
"""

from dataclasses import dataclass
from enum import Enum


class ExperimentMode(str, Enum):
    RETRIEVAL = "retrieval"
    RAG = "rag"
    AUTOGEN = "autogen"
    BASELINE = "baseline"


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    config_id: str
    mode: ExperimentMode
    description: str

    # Retrieval parameters
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    dataset_type: str = "agents_eng"
    top_k: int = 1
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 20

    # RAG parameters (only for RAG mode)
    num_retrieved_for_context: int = 5
    llm_temperature: float = 0.7
    include_examples_in_prompt: bool = True

    # LLM parameters for agent execution
    agent_model: str = "gpt-4"
    agent_base_url: str = "https://api.openai.com/v1"
    agent_api_key: str = ""
    agent_temperature: float = 0.1
    agent_max_tokens: int = 256


# ── Retrieval Configurations ──────────────────────────────────────────────────

RETRIEVAL_CONFIGS = [
    ExperimentConfig(
        config_id="ret_bge_small",
        mode=ExperimentMode.RETRIEVAL,
        description="Retrieval: BGE-small, no reranker",
        embedding_model="BAAI/bge-small-en-v1.5",
        dataset_type="agents_eng",
        top_k=1,
        use_reranker=False,
    ),
    ExperimentConfig(
        config_id="ret_bge_base",
        mode=ExperimentMode.RETRIEVAL,
        description="Retrieval: BGE-base, no reranker",
        embedding_model="BAAI/bge-base-en-v1.5",
        dataset_type="agents_eng",
        top_k=1,
        use_reranker=False,
    ),
    ExperimentConfig(
        config_id="ret_minilm",
        mode=ExperimentMode.RETRIEVAL,
        description="Retrieval: MiniLM, no reranker",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        dataset_type="agents_eng",
        top_k=1,
        use_reranker=False,
    ),
    ExperimentConfig(
        config_id="ret_bge_small_rerank",
        mode=ExperimentMode.RETRIEVAL,
        description="Retrieval: BGE-small + BGE reranker",
        embedding_model="BAAI/bge-small-en-v1.5",
        dataset_type="agents_eng",
        top_k=1,
        use_reranker=True,
        reranker_model="BAAI/bge-reranker-base",
        rerank_top_k=20,
    ),
    ExperimentConfig(
        config_id="ret_mpnet",
        mode=ExperimentMode.RETRIEVAL,
        description="Retrieval: MPNet, no reranker",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        dataset_type="agents_eng",
        top_k=1,
        use_reranker=False,
    ),
]

# ── RAG Configurations ────────────────────────────────────────────────────────

RAG_CONFIGS = [
    ExperimentConfig(
        config_id="rag_bge_small_ctx5",
        mode=ExperimentMode.RAG,
        description="RAG: BGE-small, 5 examples, temp=0.7",
        embedding_model="BAAI/bge-small-en-v1.5",
        dataset_type="agents_eng",
        num_retrieved_for_context=5,
        llm_temperature=0.7,
        include_examples_in_prompt=True,
    ),
    ExperimentConfig(
        config_id="rag_bge_base_ctx5",
        mode=ExperimentMode.RAG,
        description="RAG: BGE-base, 5 examples, temp=0.7",
        embedding_model="BAAI/bge-base-en-v1.5",
        dataset_type="agents_eng",
        num_retrieved_for_context=5,
        llm_temperature=0.7,
        include_examples_in_prompt=True,
    ),
    ExperimentConfig(
        config_id="rag_bge_small_ctx10",
        mode=ExperimentMode.RAG,
        description="RAG: BGE-small, 10 examples, temp=0.7",
        embedding_model="BAAI/bge-small-en-v1.5",
        dataset_type="agents_eng",
        num_retrieved_for_context=10,
        llm_temperature=0.7,
        include_examples_in_prompt=True,
    ),
    ExperimentConfig(
        config_id="rag_bge_small_ctx5_t03",
        mode=ExperimentMode.RAG,
        description="RAG: BGE-small, 5 examples, temp=0.3",
        embedding_model="BAAI/bge-small-en-v1.5",
        dataset_type="agents_eng",
        num_retrieved_for_context=5,
        llm_temperature=0.3,
        include_examples_in_prompt=True,
    ),
    ExperimentConfig(
        config_id="rag_minilm_ctx5",
        mode=ExperimentMode.RAG,
        description="RAG: MiniLM, 5 examples, temp=0.7",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        dataset_type="agents_eng",
        num_retrieved_for_context=5,
        llm_temperature=0.7,
        include_examples_in_prompt=True,
    ),
]

# ── AutoGen Configuration ─────────────────────────────────────────────────────

AUTOGEN_CONFIG = ExperimentConfig(
    config_id="autogen_baseline",
    mode=ExperimentMode.AUTOGEN,
    description="AutoGen: AgentBuilder baseline",
)

# ── Baseline Configuration (pure model, no retrieval/RAG/autogen) ────────────

BASELINE_CONFIG = ExperimentConfig(
    config_id="baseline_pure_model",
    mode=ExperimentMode.BASELINE,
    description="Baseline: pure model with generic system prompt, no agent generation",
)


ALL_CONFIGS = RETRIEVAL_CONFIGS + RAG_CONFIGS + [AUTOGEN_CONFIG, BASELINE_CONFIG]


def get_config_by_id(config_id: str) -> ExperimentConfig | None:
    """Get a config by its ID."""
    for cfg in ALL_CONFIGS:
        if cfg.config_id == config_id:
            return cfg
    return None
