"""Configuration for the RAG (Retrieval-Augmented Generation) system."""

import os
from typing import Literal

from pydantic import BaseModel, Field

from retrieval.config import DatasetType, RetrievalConfig


class LLMConfig(BaseModel):
    """Configuration for the LLM backend."""

    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-oss"), description="Model name to use")
    base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        description="Base URL for the API",
    )
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""), description="API key for authentication")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens to generate")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")


class RAGConfig(BaseModel):
    """Configuration for the RAG system."""

    # Retrieval configuration
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig, description="Configuration for the retrieval system"
    )

    # LLM configuration
    llm: LLMConfig = Field(default_factory=LLMConfig, description="Configuration for the LLM")

    # RAG behavior
    num_agents_to_return: int = Field(default=1, ge=1, le=10, description="Number of agents to generate/return")
    num_retrieved_for_context: int = Field(
        default=5, ge=1, le=20, description="Number of similar agents to retrieve for context"
    )
    include_examples_in_prompt: bool = Field(
        default=True, description="Whether to include retrieved examples in the prompt"
    )

    # Output format
    output_format: Literal["json", "pretty"] = Field(default="json", description="Output format for generated agents")

    @classmethod
    def with_dataset(cls, dataset_type: DatasetType, **kwargs) -> "RAGConfig":
        """Create a config with a specific dataset."""
        retrieval_config = RetrievalConfig(dataset_type=dataset_type)
        return cls(retrieval=retrieval_config, **kwargs)
