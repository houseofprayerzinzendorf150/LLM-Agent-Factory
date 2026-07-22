"""Interactive CLI for the RAG-based Agent Generator."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from retrieval.config import DatasetType, EmbeddingModel, RerankerModel, RetrievalConfig
from retrieval.rag import AgentRAG, format_agent_output
from retrieval.rag_config import LLMConfig, RAGConfig

console = Console()


def print_welcome():
    """Print welcome message."""
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Agent RAG Generator[/bold cyan]\n\n"
            "Generate AI agent specifications using RAG\n"
            "(Retrieval-Augmented Generation)\n\n"
            "[dim]Type 'help' for commands, 'quit' to exit[/dim]",
            border_style="cyan",
        )
    )


def print_help():
    """Print help message."""
    table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan")
    table.add_column("Description")

    table.add_row("generate <query>", "Generate agent(s) based on your query")
    table.add_row("search <query>", "Search for similar existing agents (no generation)")
    table.add_row("dataset <name>", "Switch dataset (eng, all)")
    table.add_row("agents <N>", "Set number of agents to generate (1-10)")
    table.add_row("examples <N>", "Set number of examples for context (1-20)")
    table.add_row("format <type>", "Set output format (json, pretty)")
    table.add_row("stats", "Show current configuration and stats")
    table.add_row("help", "Show this help message")
    table.add_row("quit / exit", "Exit the application")
    table.add_row("", "")
    table.add_row("[dim]<any text>[/dim]", "[dim]Treated as generate query[/dim]")

    console.print(table)


def print_stats(rag: AgentRAG, config: RAGConfig):
    """Print current stats and configuration."""
    stats = rag.stats

    table = Table(title="Current Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Initialized", "✓" if stats.get("initialized") else "✗")
    table.add_row("Dataset", stats.get("dataset_type", config.retrieval.dataset_type))
    table.add_row("Total Records", str(stats.get("total_records", "N/A")))
    table.add_row("Unique Agents", str(stats.get("unique_agents", "N/A")))
    table.add_row("Embedding Model", stats.get("embedding_model", config.retrieval.embedding_model))
    table.add_row("LLM Model", config.llm.model)
    table.add_row("LLM URL", config.llm.base_url)
    table.add_row("Agents to Generate", str(config.num_agents_to_return))
    table.add_row("Examples for Context", str(config.num_retrieved_for_context))
    table.add_row("Output Format", config.output_format)

    console.print(table)


def handle_generate(rag: AgentRAG, query: str, config: RAGConfig):
    """Handle generate command."""
    if not query.strip():
        console.print("[red]Please provide a query for agent generation[/red]")
        return

    try:
        with console.status("[bold cyan]Generating agent(s)...", spinner="dots"):
            agents = rag.generate(query)

        output = format_agent_output(agents, config.output_format)

        if config.output_format == "json":
            syntax = Syntax(output, "json", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[green]Generated Agent(s)[/green]", border_style="green"))
        else:
            console.print(Panel(output, title="[green]Generated Agent(s)[/green]", border_style="green"))

        console.print(f"[dim]Generated {len(agents)} agent(s)[/dim]")

    except Exception as e:
        console.print(f"[red]Error generating agent: {e}[/red]")


def handle_search(rag: AgentRAG, query: str, config: RAGConfig):
    """Handle search command."""
    if not query.strip():
        console.print("[red]Please provide a search query[/red]")
        return

    try:
        with console.status("[bold cyan]Searching...", spinner="dots"):
            results = rag.search_only(query)

        if not results:
            console.print("[yellow]No matching agents found[/yellow]")
            return

        console.print(f"\n[bold]Found {len(results)} matching agents:[/bold]\n")

        for result in results:
            score_str = f"score: {result.score:.4f}"
            if result.rerank_score is not None:
                score_str = f"rerank: {result.rerank_score:.4f}"

            agent_data = {
                "agent_id": result.agent.agent_id,
                "display_name": result.agent.display_name,
                "persona": result.agent.persona,
                "description": result.agent.description,
                "tools": result.agent.tools,
            }

            import json

            syntax = Syntax(json.dumps(agent_data, indent=2, ensure_ascii=False), "json", theme="monokai")
            console.print(
                Panel(
                    syntax,
                    title=f"[cyan][{result.rank}] {result.agent.display_name}[/cyan] ({score_str})",
                    border_style="blue",
                )
            )

    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")


def handle_dataset(rag: AgentRAG, dataset_name: str, config: RAGConfig):
    """Handle dataset switch command."""
    name = dataset_name.strip().lower()

    # Map short names to DatasetType
    dataset_map = {
        "eng": DatasetType.ENG,
        "all": DatasetType.ALL,
    }

    if name not in dataset_map:
        console.print(f"[red]Unknown dataset: {name}[/red]")
        console.print("[yellow]Available: eng, all[/yellow]")
        return

    try:
        with console.status(f"[bold cyan]Switching to {name}...", spinner="dots"):
            rag.switch_dataset(dataset_map[name])
        console.print(f"[green]Switched to dataset: {name}[/green]")
    except Exception as e:
        console.print(f"[red]Error switching dataset: {e}[/red]")


def interactive_mode(config: RAGConfig):
    """Run interactive mode."""
    print_welcome()

    # Initialize RAG
    console.print("\n[dim]Initializing RAG system...[/dim]")
    rag = AgentRAG(config)

    try:
        with console.status("[bold cyan]Loading models and index...", spinner="dots"):
            rag.initialize()
        console.print("[green]✓ RAG system ready![/green]\n")
    except Exception as e:
        console.print(f"[red]Error initializing RAG: {e}[/red]")
        return 1

    while True:
        try:
            # Get input
            user_input = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Handle commands
            if cmd in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if cmd == "help":
                print_help()

            elif cmd == "stats":
                print_stats(rag, config)

            elif cmd == "generate":
                handle_generate(rag, arg, config)

            elif cmd == "search":
                handle_search(rag, arg, config)

            elif cmd == "dataset":
                if not arg:
                    console.print(f"[cyan]Current dataset: {rag.current_dataset}[/cyan]")
                    console.print(f"[dim]Available: {', '.join(rag.available_datasets)}[/dim]")
                else:
                    handle_dataset(rag, arg, config)

            elif cmd == "agents":
                if not arg:
                    console.print(f"[cyan]Agents to generate: {config.num_agents_to_return}[/cyan]")
                else:
                    try:
                        n = int(arg)
                        if 1 <= n <= 10:
                            config.num_agents_to_return = n
                            console.print(f"[green]Set agents to generate: {n}[/green]")
                        else:
                            console.print("[red]Must be between 1 and 10[/red]")
                    except ValueError:
                        console.print("[red]Please provide a number[/red]")

            elif cmd == "examples":
                if not arg:
                    console.print(f"[cyan]Examples for context: {config.num_retrieved_for_context}[/cyan]")
                else:
                    try:
                        n = int(arg)
                        if 1 <= n <= 20:
                            config.num_retrieved_for_context = n
                            console.print(f"[green]Set examples: {n}[/green]")
                        else:
                            console.print("[red]Must be between 1 and 20[/red]")
                    except ValueError:
                        console.print("[red]Please provide a number[/red]")

            elif cmd == "format":
                if not arg:
                    console.print(f"[cyan]Output format: {config.output_format}[/cyan]")
                elif arg.lower() in ("json", "pretty"):
                    config.output_format = arg.lower()
                    console.print(f"[green]Set output format: {arg.lower()}[/green]")
                else:
                    console.print("[red]Format must be 'json' or 'pretty'[/red]")

            else:
                # Treat as generate query
                handle_generate(rag, user_input, config)

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

    return 0


def single_query_mode(config: RAGConfig, query: str):
    """Run single query mode."""
    rag = AgentRAG(config)

    try:
        rag.initialize()
        agents = rag.generate(query)

        format_agent_output(agents, config.output_format)

        return 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def main():
    """Main entry point for RAG CLI."""
    parser = argparse.ArgumentParser(
        description="RAG-based Agent Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  agent-rag

  # Single query
  agent-rag "I need an agent that helps with code review"

  # Use specific dataset
  agent-rag --dataset eng "I need a poetry writing agent"

  # Generate multiple agents
  agent-rag --agents 3 "customer support agent"
        """,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query for agent generation (interactive mode if not provided)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        choices=["eng", "all"],
        default="eng",
        help="Dataset to use (default: eng)",
    )

    parser.add_argument(
        "--agents",
        "-n",
        type=int,
        default=1,
        help="Number of agents to generate (default: 1)",
    )

    parser.add_argument(
        "--examples",
        "-e",
        type=int,
        default=5,
        help="Number of examples for context (default: 5)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "pretty"],
        default="json",
        help="Output format (default: json)",
    )

    import os

    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "gpt-4"),
        help="LLM model to use",
    )

    parser.add_argument(
        "--url",
        default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        help="LLM API base URL",
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="LLM API key",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )

    parser.add_argument(
        "--embedding",
        "--embed",
        choices=["bge-small", "bge-base", "bge-large", "bge-m3", "minilm", "mpnet", "e5-multi"],
        default="bge-small",
        help="Embedding model (default: bge-small, use bge-m3 for multilingual)",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking for better accuracy",
    )

    parser.add_argument(
        "--rerank-model",
        choices=["bge-reranker-base", "bge-reranker-large", "bge-reranker-v2-m3", "msmarco-minilm"],
        default="bge-reranker-base",
        help="Reranker model (default: bge-reranker-base)",
    )

    args = parser.parse_args()

    # Map dataset name to enum
    dataset_map = {
        "eng": DatasetType.ENG,
        "all": DatasetType.ALL,
    }

    # Map embedding model name to enum value
    embedding_map = {
        "bge-small": EmbeddingModel.BGE_SMALL.value,
        "bge-base": EmbeddingModel.BGE_BASE.value,
        "bge-large": EmbeddingModel.BGE_LARGE.value,
        "bge-m3": EmbeddingModel.BGE_M3.value,
        "minilm": EmbeddingModel.MINILM.value,
        "mpnet": EmbeddingModel.MPNET.value,
        "e5-multi": EmbeddingModel.MULTILINGUAL_E5.value,
    }

    # Map reranker model name to enum value
    reranker_map = {
        "bge-reranker-base": RerankerModel.BGE_RERANKER_BASE.value,
        "bge-reranker-large": RerankerModel.BGE_RERANKER_LARGE.value,
        "bge-reranker-v2-m3": RerankerModel.BGE_RERANKER_V2_M3.value,
        "msmarco-minilm": RerankerModel.MSMARCO_MINILM.value,
    }

    # Build config
    llm_config = LLMConfig(
        model=args.model,
        base_url=args.url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    # Build retrieval config with embedding and reranker settings
    retrieval_config = RetrievalConfig(
        dataset_type=dataset_map[args.dataset],
        embedding_model=embedding_map[args.embedding],
        use_reranker=args.rerank,
        reranker_model=reranker_map[args.rerank_model],
    )

    config = RAGConfig(
        retrieval=retrieval_config,
        llm=llm_config,
        num_agents_to_return=args.agents,
        num_retrieved_for_context=args.examples,
        output_format=args.format,
    )

    # Run appropriate mode
    if args.query:
        return single_query_mode(config, args.query)
    return interactive_mode(config)


if __name__ == "__main__":
    sys.exit(main())
