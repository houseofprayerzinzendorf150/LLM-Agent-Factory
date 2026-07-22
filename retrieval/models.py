"""Pydantic models for the retrieval system."""

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    """Agent specification from the dataset."""

    model_config = ConfigDict(frozen=True)  # Immutable for hashing

    agent_id: str = Field(description="Unique identifier for the agent")
    display_name: str = Field(description="Human-readable name")
    persona: str = Field(default="", description="Agent's persona/character")
    description: str = Field(default="", description="What the agent does")
    tools: list[str] = Field(default_factory=list, description="Available tools")

    def get_indexable_text(self) -> str:
        """Get text representation for embedding/indexing."""
        return f"{self.display_name}: {self.description} {self.persona}"


class AgentRecord(BaseModel):
    """A single agent record from the dataset with source info."""

    input_text: str = Field(description="Original user query")
    agent: AgentSpec = Field(description="Agent specification")
    source_file: str = Field(description="Source dataset file")

    @classmethod
    def from_json(cls, data: dict, source_file: str) -> "AgentRecord":
        """Create an AgentRecord from a JSON object."""
        output = data.get("output", {})
        agent = AgentSpec(
            agent_id=output.get("agent_id", ""),
            display_name=output.get("display_name", ""),
            persona=output.get("persona", ""),
            description=output.get("description", ""),
            tools=output.get("tools", []),
        )
        return cls(
            input_text=data.get("input", ""),
            agent=agent,
            source_file=source_file,
        )


class RetrievalResult(BaseModel):
    """A single retrieval result."""

    agent: AgentSpec = Field(description="The retrieved agent")
    score: float = Field(description="Similarity/relevance score")
    rank: int = Field(description="Result rank (1-indexed)")
    rerank_score: float | None = Field(default=None, description="Score after reranking")

    def format_output(self, verbose: bool = False, full: bool = True) -> str:
        """
        Format the result for display.

        Args:
            verbose: Show additional metadata (scores, etc.)
            full: Show full agent spec as in dataset (default True)

        """
        import json

        lines = []

        # Header with rank and score
        score_str = f"score: {self.score:.4f}"
        if self.rerank_score is not None:
            score_str = f"rerank: {self.rerank_score:.4f}, retrieval: {self.score:.4f}"

        lines.append(f"[{self.rank}] ({score_str})")

        if full:
            # Full agent output as in dataset
            agent_dict = {
                "agent_id": self.agent.agent_id,
                "display_name": self.agent.display_name,
                "persona": self.agent.persona,
                "description": self.agent.description,
                "tools": self.agent.tools,
            }
            lines.append(json.dumps(agent_dict, indent=2, ensure_ascii=False))
        else:
            # Compact format
            lines.append(f"    {self.agent.display_name}")
            lines.append(f"    ID: {self.agent.agent_id}")
            if verbose:
                lines.append(f"    Description: {self.agent.description}")
                lines.append(f"    Persona: {self.agent.persona}")
            if self.agent.tools:
                lines.append(f"    Tools: {', '.join(self.agent.tools)}")

        return "\n".join(lines)


class SearchQuery(BaseModel):
    """Search query with parameters."""

    text: str = Field(description="Query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Min similarity")
    use_reranker: bool = Field(default=False, description="Apply reranking")
