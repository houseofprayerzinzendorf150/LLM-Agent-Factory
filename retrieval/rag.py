"""RAG (Retrieval-Augmented Generation) system for agent generation."""

import json

from openai import OpenAI

from retrieval.config import DatasetType
from retrieval.models import RetrievalResult
from retrieval.rag_config import RAGConfig
from retrieval.retriever import AgentRetriever

SYSTEM_PROMPT = """You are an AI Agent Factory - a specialized system that creates AI agent specifications based on user requests.

Your task is to generate agent specifications in a specific JSON format. Each agent specification must include:
- agent_id: A unique snake_case identifier for the agent
- display_name: A human-readable name for the agent
- persona: A description of the agent's personality and approach
- description: What the agent does and how it helps users
- tools: A list of tools/capabilities the agent has access to (can be empty)

You will be provided with examples of similar agents for reference. Use them to understand the format and style, but create a unique agent tailored to the user's specific request.

IMPORTANT:
- Output ONLY valid JSON, no additional text
- If asked for multiple agents, output a JSON array
- Match the style and quality of the provided examples
- Be creative but practical in your agent designs"""


def build_prompt(
    query: str,
    examples: list[RetrievalResult],
    num_agents: int = 1,
) -> str:
    """Build the prompt for the LLM with retrieved examples."""
    parts = []

    # Add examples section
    if examples:
        parts.append("Here are examples of similar agents for reference:\n")
        for i, result in enumerate(examples, 1):
            agent = result.agent
            example = {
                "agent_id": agent.agent_id,
                "display_name": agent.display_name,
                "persona": agent.persona,
                "description": agent.description,
                "tools": agent.tools,
            }
            parts.append(f"Example {i}:")
            parts.append(json.dumps(example, indent=2, ensure_ascii=False))
            parts.append("")

    # Add user request
    parts.append("---")
    parts.append(f"User request: {query}")
    parts.append("")

    if num_agents == 1:
        parts.append("Generate ONE agent specification in JSON format that best matches this request:")
    else:
        parts.append(f"Generate {num_agents} different agent specifications as a JSON array that match this request:")

    return "\n".join(parts)


def parse_agent_response(response_text: str, num_agents: int = 1) -> list[dict]:
    """Parse the LLM response into agent specifications."""
    # Clean up the response
    text = response_text.strip()

    # Try to find JSON in the response
    # Sometimes LLMs wrap JSON in markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    try:
        data = json.loads(text)

        # Normalize to list
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        msg = f"Unexpected response type: {type(data)}"
        raise ValueError(msg)

    except json.JSONDecodeError as e:
        msg = f"Failed to parse LLM response as JSON: {e}\nResponse: {text[:500]}"
        raise ValueError(msg)


class AgentRAG:
    """
    RAG system for agent generation.

    Combines retrieval of similar agents with LLM generation
    to create new agent specifications based on user queries.
    """

    def __init__(self, config: RAGConfig | None = None):
        """
        Initialize the RAG system.

        Args:
            config: RAG configuration. Uses defaults if not provided.

        """
        self.config = config or RAGConfig()
        self._retriever: AgentRetriever | None = None
        self._client: OpenAI | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the RAG system (retriever and LLM client)."""
        if self._initialized:
            return

        # Initialize retriever
        self._retriever = AgentRetriever(self.config.retrieval)
        self._retriever.initialize()

        # Initialize OpenAI client
        self._client = OpenAI(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            timeout=self.config.llm.timeout,
        )

        self._initialized = True

    def generate(
        self,
        query: str,
        num_agents: int | None = None,
        num_examples: int | None = None,
    ) -> list[dict]:
        """
        Generate agent specifications based on user query.

        Args:
            query: User's request for an agent
            num_agents: Number of agents to generate (overrides config)
            num_examples: Number of examples to retrieve (overrides config)

        Returns:
            List of generated agent specifications as dicts

        """
        if not self._initialized:
            self.initialize()

        n_agents = num_agents or self.config.num_agents_to_return
        n_examples = num_examples or self.config.num_retrieved_for_context

        # Stage 1: Retrieve similar agents
        assert self._retriever is not None, "Retriever not initialized"
        examples = self._retriever.search(query, top_k=n_examples)

        if not examples:
            pass
        else:
            pass

        # Stage 2: Build prompt with examples
        prompt = build_prompt(
            query=query,
            examples=examples if self.config.include_examples_in_prompt else [],
            num_agents=n_agents,
        )

        # Stage 3: Generate with LLM
        assert self._client is not None, "LLM client not initialized"
        response = self._client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        response_text = response.choices[0].message.content

        # Stage 4: Parse response
        return parse_agent_response(response_text, n_agents)

    def search_only(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for similar agents without generation.

        Useful for exploring the dataset or when you don't need new agents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results

        """
        if not self._initialized:
            self.initialize()

        assert self._retriever is not None, "Retriever not initialized"
        k = top_k or self.config.num_retrieved_for_context
        return self._retriever.search(query, top_k=k)

    def switch_dataset(self, dataset_type: DatasetType) -> None:
        """
        Switch to a different dataset.

        Args:
            dataset_type: The dataset to switch to

        """
        if self._retriever:
            self._retriever.switch_dataset(dataset_type)
        else:
            # Convert enum to string to match Pydantic's use_enum_values=True behavior
            dataset_type_str = dataset_type.value if isinstance(dataset_type, DatasetType) else dataset_type
            self.config.retrieval.dataset_type = dataset_type_str

        # Need to reinitialize to reload the index
        if self._initialized:
            self._initialized = False
            self.initialize()

    @property
    def available_datasets(self) -> list[str]:
        """Get list of available datasets."""
        return [dt.value for dt in DatasetType]

    @property
    def current_dataset(self) -> str:
        """Get the current dataset name."""
        return self.config.retrieval.dataset_type

    @property
    def stats(self) -> dict:
        """Get statistics about the current state."""
        if not self._initialized:
            return {
                "initialized": False,
                "dataset": self.config.retrieval.dataset_type,
            }

        assert self._retriever is not None, "Retriever not initialized"
        return {
            "initialized": True,
            **self._retriever.dataset_stats,
            "llm_model": self.config.llm.model,
            "llm_base_url": self.config.llm.base_url,
        }


def format_agent_output(agents: list[dict], format_type: str = "json") -> str:
    """Format agent output for display."""
    if format_type == "json":
        if len(agents) == 1:
            return json.dumps(agents[0], indent=2, ensure_ascii=False)
        return json.dumps(agents, indent=2, ensure_ascii=False)

    # Pretty format
    lines = []
    for i, agent in enumerate(agents, 1):
        if len(agents) > 1:
            lines.append(f"═══ Agent {i} ═══")
        lines.append(f"ID: {agent.get('agent_id', 'N/A')}")
        lines.append(f"Name: {agent.get('display_name', 'N/A')}")
        lines.append(f"Description: {agent.get('description', 'N/A')}")
        lines.append(f"Persona: {agent.get('persona', 'N/A')}")
        tools = agent.get("tools", [])
        if tools:
            lines.append(f"Tools: {', '.join(tools)}")
        else:
            lines.append("Tools: None")
        lines.append("")

    return "\n".join(lines)
