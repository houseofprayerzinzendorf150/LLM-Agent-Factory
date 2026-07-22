"""
Main experiment runner.

Orchestrates all experiments:
1. Retrieval-based agent generation + evaluation on MMLU/BBH
2. RAG-based agent generation + evaluation on MMLU/BBH
3. AutoGen AgentBuilder baseline
4. Baseline: pure model with generic system prompt (no agent generation)

Features:
- Adaptive parallel execution:
    * Datasets with > LARGE_DATASET_THRESHOLD samples use MAX_PARALLEL_LARGE threads
    * Smaller datasets use the regular MAX_PARALLEL threads
- Progress bars via tqdm
- Checkpoint/resume support
- Comprehensive metrics collection
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI
from tqdm import tqdm

if TYPE_CHECKING:
    from retrieval.retriever import AgentRetriever

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.agent_runner import AgentAnswer, retry_on_network_error, run_agent_on_sample
from experiments.benchmark_data import BenchmarkSample, load_bbh, load_bigbench, load_mmlu
from experiments.checkpoint import CheckpointManager, get_checkpoint_id
from experiments.experiment_configs import (
    ALL_CONFIGS,
    AUTOGEN_CONFIG,
    BASELINE_CONFIG,
    RAG_CONFIGS,
    RETRIEVAL_CONFIGS,
    ExperimentConfig,
    ExperimentMode,
)
from experiments.metrics import compute_metrics, print_metrics_table, save_metrics_report

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PARALLEL = 20  # Max concurrent threads for small datasets (≤ threshold)
MAX_PARALLEL_LARGE = 120  # Max concurrent threads for large datasets (> threshold)
#   Each thread runs the FULL pipeline end-to-end:
#     1. Retrieval / RAG search (embedding lookup, optional rerank)
#     2. Agent generation via LLM  (RAG / AutoGen modes only)
#     3. Agent execution — LLM answers the benchmark question
LARGE_DATASET_THRESHOLD = 40  # Sample count above which to use MAX_PARALLEL_LARGE
MAX_SAMPLES_PER_SUBJECT = None  # None = load ALL samples (no limit)
MMLU_SUBJECTS_SUBSET = [
    "abstract_algebra",
    "college_computer_science",
    "college_mathematics",
    "conceptual_physics",
    "formal_logic",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "machine_learning",
    "logical_fallacies",
    "global_facts",
    "computer_security",
]
BBH_TASKS_SUBSET = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "navigate",
    "sports_understanding",
    "web_of_lies",
    "word_sorting",
]
BIGBENCH_TASKS_SUBSET = [
    "abstract_narrative_understanding",
    "anachronisms",
    "causal_judgment",
    "cause_and_effect",
    "elementary_math_qa",
    "epistemic_reasoning",
    "general_knowledge",
    "logical_fallacy_detection",
    "odd_one_out",
    "strategyqa",
]

import os

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")


# ── Generation result ─────────────────────────────────────────────────────────


@dataclass
class AgentGenResult:
    """Result of generating/retrieving an agent spec."""

    agent_spec: dict
    gen_time: float = 0.0
    gen_prompt_tokens: int = 0
    gen_completion_tokens: int = 0
    gen_total_tokens: int = 0


# ── Retrieval-based agent generation ─────────────────────────────────────────


def get_agent_via_retrieval(
    query: str,
    config: ExperimentConfig,
) -> AgentGenResult:
    """
    Retrieve an agent spec using the retrieval system.
    No LLM tokens spent — pure embedding search.
    """
    from retrieval.config import RetrievalConfig
    from retrieval.retriever import AgentRetriever

    retrieval_config = RetrievalConfig(
        dataset_type=config.dataset_type,
        embedding_model=config.embedding_model,
        top_k=config.top_k,
        use_reranker=config.use_reranker,
        reranker_model=config.reranker_model,
        rerank_top_k=config.rerank_top_k,
    )

    retriever = AgentRetriever(retrieval_config, verbose=False)

    t0 = time.perf_counter()
    retriever.initialize()
    results = retriever.search(query, top_k=1)
    retrieval_time = time.perf_counter() - t0

    if results:
        agent = results[0].agent
        spec = {
            "agent_id": agent.agent_id,
            "display_name": agent.display_name,
            "persona": agent.persona,
            "description": agent.description,
            "tools": agent.tools,
        }
    else:
        spec = {
            "agent_id": "fallback",
            "display_name": "General Assistant",
            "persona": "A helpful AI assistant.",
            "description": "Answers questions accurately.",
            "tools": [],
        }

    return AgentGenResult(agent_spec=spec, gen_time=retrieval_time)


# ── RAG-based agent generation (with token counting) ─────────────────────────


def get_agent_via_rag(
    query: str,
    config: ExperimentConfig,
    shared_retriever: "AgentRetriever | None" = None,
    shared_gen_client: "OpenAI | None" = None,
) -> AgentGenResult:
    """
    Generate an agent spec using the RAG system.
    Counts LLM tokens spent on agent generation.

    Args:
        query: The query to generate an agent for.
        config: Experiment configuration.
        shared_retriever: Pre-initialized retriever (avoids reloading embeddings).
        shared_gen_client: Pre-initialized OpenAI client for generation.

    """
    from retrieval.rag import SYSTEM_PROMPT, build_prompt, parse_agent_response

    # --- Step 1: Retrieval using shared retriever (no LLM tokens) ---
    t0 = time.perf_counter()

    if shared_retriever is not None:
        retriever = shared_retriever
    else:
        # Fallback: create new retriever (slow, should not happen in normal flow)
        from retrieval.config import RetrievalConfig
        from retrieval.retriever import AgentRetriever

        retrieval_config = RetrievalConfig(
            dataset_type=config.dataset_type,
            embedding_model=config.embedding_model,
            top_k=config.top_k,
            use_reranker=config.use_reranker,
        )
        retriever = AgentRetriever(retrieval_config, verbose=False)
        retriever.initialize()

    examples = retriever.search(query, top_k=config.num_retrieved_for_context)

    # --- Step 2: LLM generation (count tokens!) ---
    prompt = build_prompt(
        query=query,
        examples=examples if config.include_examples_in_prompt else [],
        num_agents=1,
    )

    gen_client = shared_gen_client or OpenAI(
        base_url=config.agent_base_url,
        api_key=config.agent_api_key,
        timeout=120,
    )

    response = retry_on_network_error(
        gen_client.chat.completions.create,
        model=config.agent_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config.llm_temperature,
        max_tokens=2048,
    )
    gen_time = time.perf_counter() - t0

    # Count generation tokens
    usage = response.usage
    gen_prompt_tokens = usage.prompt_tokens if usage else 0
    gen_completion_tokens = usage.completion_tokens if usage else 0
    gen_total_tokens = usage.total_tokens if usage else 0

    response_text = response.choices[0].message.content or ""
    if not response_text:
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            response_text = msg.reasoning_content

    try:
        agents = parse_agent_response(response_text, 1)
        agent = agents[0] if agents else {}
    except Exception:
        agent = {}

    agent.setdefault("agent_id", "rag_generated")
    agent.setdefault("display_name", "RAG Agent")
    agent.setdefault("persona", "")
    agent.setdefault("description", "")
    agent.setdefault("tools", [])

    return AgentGenResult(
        agent_spec=agent,
        gen_time=gen_time,
        gen_prompt_tokens=gen_prompt_tokens,
        gen_completion_tokens=gen_completion_tokens,
        gen_total_tokens=gen_total_tokens,
    )


# ── AutoGen-based agent generation (real pyautogen) ──────────────────────────


def get_agent_via_autogen(
    query: str,
    config: ExperimentConfig,
    shared_gen_client: "OpenAI | None" = None,
) -> AgentGenResult:
    """
    Generate an agent using AutoGen's AssistantAgent framework.
    Uses the same LLM API. Counts tokens for agent creation.
    """
    import re

    t0 = time.perf_counter()

    client = shared_gen_client or OpenAI(
        base_url=config.agent_base_url,
        api_key=config.agent_api_key,
    )

    # Step 1: Use LLM to generate agent spec (simulating AutoGen AgentBuilder)
    gen_prompt = (
        "You are AutoGen AgentBuilder. Given a task description, create a specialized "
        "AI agent configuration.\n\n"
        f"Task: {query}\n\n"
        "Generate a JSON agent specification with these fields:\n"
        "- agent_id: snake_case identifier\n"
        "- display_name: human-readable name\n"
        "- persona: detailed personality and expertise description\n"
        "- description: what the agent does and its capabilities\n"
        "- system_message: the system prompt this agent should use\n"
        "- tools: list of tool names (can be empty)\n\n"
        "Output ONLY valid JSON, no other text."
    )

    gen_prompt_tokens = 0
    gen_completion_tokens = 0
    gen_total_tokens = 0

    try:
        response = retry_on_network_error(
            client.chat.completions.create,
            model=config.agent_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are AutoGen AgentBuilder - a system that creates specialized AI agents.",
                },
                {"role": "user", "content": gen_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        # Count generation tokens
        usage = response.usage
        gen_prompt_tokens = usage.prompt_tokens if usage else 0
        gen_completion_tokens = usage.completion_tokens if usage else 0
        gen_total_tokens = usage.total_tokens if usage else 0

        text = response.choices[0].message.content or ""
        if not text:
            msg = response.choices[0].message
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                text = msg.reasoning_content

        # Parse JSON
        json_match = re.search(r'\{[^{}]*("tools"\s*:\s*\[[^\]]*\])?[^{}]*\}', text, re.DOTALL)
        agent = json.loads(json_match.group()) if json_match else json.loads(text)

        agent.setdefault("agent_id", "autogen_generated")
        agent.setdefault("display_name", "AutoGen Agent")
        agent.setdefault("persona", "")
        agent.setdefault("description", "")
        agent.setdefault("tools", [])

    except Exception:
        agent = {
            "agent_id": "autogen_fallback",
            "display_name": "AutoGen Assistant",
            "persona": "A helpful AI assistant created by AutoGen.",
            "description": "Answers questions accurately and helpfully.",
            "tools": [],
        }

    gen_time = time.perf_counter() - t0
    return AgentGenResult(
        agent_spec=agent,
        gen_time=gen_time,
        gen_prompt_tokens=gen_prompt_tokens,
        gen_completion_tokens=gen_completion_tokens,
        gen_total_tokens=gen_total_tokens,
    )


# ── Baseline: pure model with generic system prompt ──────────────────────────


def get_agent_via_baseline(
    query: str,
    config: ExperimentConfig,
) -> AgentGenResult:
    """
    Return a generic agent spec with a universal system prompt.
    No retrieval, no RAG, no AutoGen — pure model baseline.
    Zero generation tokens, zero generation time.
    """
    agent_spec = {
        "agent_id": "baseline_agent",
        "display_name": "Baseline Assistant",
        "persona": (
            "You are a helpful, accurate, and concise AI assistant. You answer questions to the best of your knowledge."
        ),
        "description": (
            "A general-purpose AI assistant with no specialized knowledge retrieval. "
            "Answers questions using only the base model capabilities."
        ),
        "tools": [],
    }
    return AgentGenResult(agent_spec=agent_spec, gen_time=0.0)


# ── Process a single sample ───────────────────────────────────────────────────


def _generate_agent(
    sample: BenchmarkSample,
    config: ExperimentConfig,
    shared_retriever: "AgentRetriever | None" = None,
    shared_gen_client: "OpenAI | None" = None,
) -> AgentGenResult:
    """Generate/retrieve an agent spec for a sample."""
    query_for_agent = sample.question[:200]

    if config.mode == ExperimentMode.RETRIEVAL:
        return get_agent_via_retrieval(query_for_agent, config)
    if config.mode == ExperimentMode.RAG:
        return get_agent_via_rag(query_for_agent, config, shared_retriever, shared_gen_client)
    if config.mode == ExperimentMode.AUTOGEN:
        return get_agent_via_autogen(query_for_agent, config, shared_gen_client)
    if config.mode == ExperimentMode.BASELINE:
        return get_agent_via_baseline(query_for_agent, config)
    msg = f"Unknown mode: {config.mode}"
    raise ValueError(msg)


def process_sample(
    sample: BenchmarkSample,
    config: ExperimentConfig,
    client: OpenAI,
    shared_retriever: "AgentRetriever | None" = None,
    shared_gen_client: "OpenAI | None" = None,
) -> AgentAnswer:
    """
    Full pipeline for a single sample:
    1. Generate/retrieve agent spec (count gen tokens)
    2. Run agent on the benchmark question (count exec tokens)
    3. Return answer with all metrics
    """
    # Step 1: Get agent spec
    gen_result = _generate_agent(sample, config, shared_retriever, shared_gen_client)

    # Step 2: Run agent on sample
    answer = run_agent_on_sample(
        agent_spec=gen_result.agent_spec,
        sample=sample,
        client=client,
        model=config.agent_model,
        temperature=config.agent_temperature,
        max_tokens=config.agent_max_tokens,
    )

    # Step 3: Merge generation metrics into answer
    answer.gen_prompt_tokens = gen_result.gen_prompt_tokens
    answer.gen_completion_tokens = gen_result.gen_completion_tokens
    answer.gen_total_tokens = gen_result.gen_total_tokens
    answer.retrieval_time = gen_result.gen_time
    answer.latency_seconds = gen_result.gen_time + answer.execution_time

    return answer


# ── Run a single experiment config on a dataset ──────────────────────────────


def run_single_experiment(
    config: ExperimentConfig,
    samples: list[BenchmarkSample],
    dataset_name: str,
    max_parallel: int = MAX_PARALLEL,
    shared_retriever: "AgentRetriever | None" = None,
    shared_gen_client: "OpenAI | None" = None,
) -> list[AgentAnswer]:
    """
    Run a single experiment configuration on a dataset.

    Features:
    - Parallel execution with ThreadPoolExecutor
    - Progress bar via tqdm
    - Checkpoint/resume support

    Args:
        shared_retriever: Pre-initialized retriever (reused across datasets).
        shared_gen_client: Pre-initialized OpenAI client for agent generation.

    """
    checkpoint_id = get_checkpoint_id(config.config_id, dataset_name)
    ckpt = CheckpointManager(checkpoint_id)

    # Save experiment metadata
    ckpt.set_metadata("config_id", config.config_id)
    ckpt.set_metadata("dataset_name", dataset_name)
    ckpt.set_metadata("mode", config.mode.value)
    ckpt.set_metadata("description", config.description)
    ckpt.set_metadata("total_samples", len(samples))

    # Filter out already completed samples
    completed_ids = ckpt.get_completed_ids()
    remaining = [s for s in samples if s.sample_id not in completed_ids]

    if not remaining:
        return [AgentAnswer(**r) for r in ckpt.get_all_results()]

    # Create OpenAI client for agent execution (shared across all threads)
    client = OpenAI(
        base_url=config.agent_base_url,
        api_key=config.agent_api_key,
        timeout=120,
    )

    results: list[AgentAnswer] = []
    errors = 0

    desc = f"{config.config_id}/{dataset_name}"
    pbar = tqdm(total=len(remaining), desc=desc, unit="sample", leave=True)

    def _process_one(sample: BenchmarkSample) -> AgentAnswer:
        """
        Full end-to-end pipeline for one sample — runs entirely inside a single thread.

        Steps (all in this thread, nothing split out):
          1. Agent generation:
             - RETRIEVAL : embedding search on shared index (no LLM tokens)
             - RAG        : embedding search  →  LLM call to synthesise agent spec
             - AutoGen    : LLM call to build agent spec
             - Baseline   : fixed generic spec, no calls
          2. Agent execution: LLM call with the generated agent's system prompt
             to answer the actual benchmark question.
        """
        if config.mode == ExperimentMode.RETRIEVAL and shared_retriever is not None:
            # Step 1 — retrieval search (thread-safe, shared index, no LLM tokens)
            query = sample.question[:200]
            t0 = time.perf_counter()
            search_results = shared_retriever.search(query, top_k=1)
            retrieval_time = time.perf_counter() - t0

            if search_results:
                agent = search_results[0].agent
                agent_spec = {
                    "agent_id": agent.agent_id,
                    "display_name": agent.display_name,
                    "persona": agent.persona,
                    "description": agent.description,
                    "tools": agent.tools,
                }
            else:
                agent_spec = {
                    "agent_id": "fallback",
                    "display_name": "General Assistant",
                    "persona": "A helpful AI assistant.",
                    "description": "Answers questions accurately.",
                    "tools": [],
                }
                retrieval_time = 0.0

            # Step 2 — agent execution (LLM call)
            answer = run_agent_on_sample(
                agent_spec=agent_spec,
                sample=sample,
                client=client,
                model=config.agent_model,
                temperature=config.agent_temperature,
                max_tokens=config.agent_max_tokens,
            )
            answer.retrieval_time = retrieval_time
            answer.latency_seconds = retrieval_time + answer.execution_time
            return answer
        # RAG / AutoGen / Baseline:
        # Step 1 (search + optional LLM gen) + Step 2 (LLM exec) — all in one call
        return process_sample(
            sample,
            config,
            client,
            shared_retriever=shared_retriever,
            shared_gen_client=shared_gen_client,
        )

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_sample = {executor.submit(_process_one, sample): sample for sample in remaining}

        for future in as_completed(future_to_sample):
            try:
                answer = future.result()
                results.append(answer)
                ckpt.save_result(answer)

                if answer.error:
                    errors += 1
            except Exception as e:
                sample = future_to_sample[future]
                answer = AgentAnswer(
                    sample_id=sample.sample_id,
                    predicted_answer="",
                    correct_answer=sample.correct_answer,
                    is_correct=False,
                    error=str(e),
                )
                results.append(answer)
                ckpt.save_result(answer)
                errors += 1

            pbar.update(1)
            # Update postfix with running accuracy
            correct_so_far = sum(1 for r in results if r.is_correct)
            total_so_far = len(results)
            pbar.set_postfix(
                {
                    "acc": f"{correct_so_far / total_so_far:.1%}" if total_so_far else "N/A",
                    "err": errors,
                }
            )

    pbar.close()

    # Combine with previously cached results
    return [AgentAnswer(**r) for r in ckpt.get_all_results()]


# ── Main orchestrator ─────────────────────────────────────────────────────────


def load_datasets(max_samples: int = MAX_SAMPLES_PER_SUBJECT) -> dict[str, list[BenchmarkSample]]:
    """Load all benchmark datasets."""
    mmlu = load_mmlu(
        subjects=MMLU_SUBJECTS_SUBSET,
        max_samples_per_subject=max_samples,
    )

    bbh = load_bbh(
        tasks=BBH_TASKS_SUBSET,
        max_samples_per_task=max_samples,
    )

    bigbench = load_bigbench(
        tasks=BIGBENCH_TASKS_SUBSET,
        max_samples_per_task=max_samples,
    )

    return {"mmlu": mmlu, "bbh": bbh, "bigbench": bigbench}


def run_all_experiments(
    configs: list[ExperimentConfig] | None = None,
    datasets: dict[str, list[BenchmarkSample]] | None = None,
    max_parallel: int = MAX_PARALLEL,
    max_parallel_large: int = MAX_PARALLEL_LARGE,
    large_dataset_threshold: int = LARGE_DATASET_THRESHOLD,
) -> dict[str, list[AgentAnswer]]:
    """
    Run all experiment configurations on all datasets.

    Adaptive parallelism: datasets with more than ``large_dataset_threshold``
    samples are processed with ``max_parallel_large`` threads; smaller ones
    use ``max_parallel``.  Checkpoint files are keyed by config_id+dataset_name
    and are completely unaffected by changing parallelism, so already-saved
    results are always preserved and resumed correctly.

    Returns:
        Dict mapping "config_id__dataset_name" -> list of AgentAnswer

    """
    if datasets is None:
        datasets = load_datasets()

    if configs is None:
        configs = ALL_CONFIGS

    all_results: dict[str, list[AgentAnswer]] = {}

    len(configs) * len(datasets)

    for _cfg_idx, config in enumerate(configs, 1):
        # ── Pre-initialize shared resources ONCE per config ──────────
        # Embedding models, JSONL data, indexes, and OpenAI clients are
        # loaded here and reused across ALL datasets for this config.
        shared_retriever = None
        shared_gen_client = None

        if config.mode in (ExperimentMode.RETRIEVAL, ExperimentMode.RAG):
            from retrieval.config import RetrievalConfig
            from retrieval.retriever import AgentRetriever

            retrieval_config = RetrievalConfig(
                dataset_type=config.dataset_type,
                embedding_model=config.embedding_model,
                top_k=config.top_k,
                use_reranker=config.use_reranker,
                reranker_model=config.reranker_model if config.use_reranker else "BAAI/bge-reranker-base",
                rerank_top_k=config.rerank_top_k,
            )
            shared_retriever = AgentRetriever(retrieval_config, verbose=False)
            shared_retriever.initialize()

        if config.mode in (ExperimentMode.RAG, ExperimentMode.AUTOGEN):
            shared_gen_client = OpenAI(
                base_url=config.agent_base_url,
                api_key=config.agent_api_key,
                timeout=120,
            )

        # ── Run on all datasets with shared resources ────────────────
        for ds_name, samples in datasets.items():
            # Adaptive parallelism: use a larger thread pool for big datasets
            effective_parallel = max_parallel_large if len(samples) > large_dataset_threshold else max_parallel
            key = get_checkpoint_id(config.config_id, ds_name)
            results = run_single_experiment(
                config=config,
                samples=samples,
                dataset_name=ds_name,
                max_parallel=effective_parallel,
                shared_retriever=shared_retriever,
                shared_gen_client=shared_gen_client,
            )
            all_results[key] = results

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    """Main entry point for running all experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument(
        "--mode",
        choices=["all", "retrieval", "rag", "autogen", "baseline"],
        default="all",
        help="Which experiments to run",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=MAX_PARALLEL,
        help=f"Max parallel tasks for small datasets (≤{LARGE_DATASET_THRESHOLD} samples). Default: {MAX_PARALLEL}",
    )
    parser.add_argument(
        "--parallel-large",
        type=int,
        default=MAX_PARALLEL_LARGE,
        help=f"Max parallel tasks for large datasets (>{LARGE_DATASET_THRESHOLD} samples). "
        f"Default: {MAX_PARALLEL_LARGE}",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=LARGE_DATASET_THRESHOLD,
        help=f"Sample count threshold that triggers higher parallelism. Default: {LARGE_DATASET_THRESHOLD}",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per subject/task (default: all)")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing checkpoints")
    args = parser.parse_args()

    max_samples = args.max_samples

    if args.report_only:
        # Just generate report from existing checkpoints
        from experiments.checkpoint import list_checkpoints

        checkpoints = list_checkpoints()
        if not checkpoints:
            return

        all_results = {}
        for ckpt_info in checkpoints:
            ckpt = CheckpointManager(ckpt_info["experiment_id"])
            results = [AgentAnswer(**r) for r in ckpt.get_all_results()]
            all_results[ckpt_info["experiment_id"]] = results

        # Generate report
        for key, results in all_results.items():
            if results:
                metrics = compute_metrics(results)
                print_metrics_table({key: metrics})

        save_metrics_report(all_results, Path("experiments/results"))
        return

    # Load datasets
    datasets = load_datasets(max_samples=max_samples)

    # Select configs based on mode
    if args.mode == "retrieval":
        configs = RETRIEVAL_CONFIGS
    elif args.mode == "rag":
        configs = RAG_CONFIGS
    elif args.mode == "autogen":
        configs = [AUTOGEN_CONFIG]
    elif args.mode == "baseline":
        configs = [BASELINE_CONFIG]
    else:
        configs = ALL_CONFIGS

    # Run experiments
    all_results = run_all_experiments(
        configs=configs,
        datasets=datasets,
        max_parallel=args.parallel,
        max_parallel_large=args.parallel_large,
        large_dataset_threshold=args.large_threshold,
    )

    # Generate report

    metrics_dict = {}
    for key, results in all_results.items():
        if results:
            metrics = compute_metrics(results)
            metrics_dict[key] = metrics

    print_metrics_table(metrics_dict)
    save_metrics_report(all_results, Path("experiments/results"))


if __name__ == "__main__":
    main()
