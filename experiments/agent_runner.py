"""
Agent runner that takes an AgentSpec from LLM-Agent-Factory and runs it
as a single agent via the DiMAS framework (GraphBuilder + MACPRunner).

The agent answers benchmark questions using its persona/description.
Tracks token usage and latency.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI

from experiments.benchmark_data import BenchmarkSample

# ── Retry logic for network / server errors ──────────────────────────────────

T = TypeVar("T")

# Exceptions that indicate a transient network / server problem worth retrying
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,  # covers socket-level errors
)

try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    _RETRYABLE_EXCEPTIONS = (
        *_RETRYABLE_EXCEPTIONS,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except ImportError:
    pass

# Also catch generic httpx transport errors if available
try:
    import httpx

    _RETRYABLE_EXCEPTIONS = (*_RETRYABLE_EXCEPTIONS, httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
except ImportError:
    pass


def retry_on_network_error[T](
    fn: Callable[..., T],
    *args,
    max_retries: int = 60,
    initial_wait: float = 5.0,
    max_wait: float = 60.0,
    backoff_factor: float = 1.5,
    **kwargs,
) -> T:
    """
    Call *fn* and retry on transient network / server errors.

    Waits with exponential backoff between retries.  Prints a message
    so the user knows the runner is waiting for connectivity.

    Args:
        fn: Callable to execute.
        max_retries: How many times to retry (default 60 ≈ up to ~30 min).
        initial_wait: Seconds to wait after the first failure.
        max_wait: Cap on the wait between retries.
        backoff_factor: Multiplier for the wait after each failure.

    """
    wait = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE_EXCEPTIONS:
            if attempt == max_retries:
                raise
            time.sleep(wait)
            wait = min(wait * backoff_factor, max_wait)
    # Should never reach here, but just in case
    return fn(*args, **kwargs)


@dataclass
class AgentAnswer:
    """Result of an agent answering a benchmark question."""

    sample_id: str
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    # Tokens spent on agent EXECUTION (answering the question via DiMAS)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Tokens spent on agent GENERATION (RAG / AutoGen creating the agent spec)
    gen_prompt_tokens: int = 0
    gen_completion_tokens: int = 0
    gen_total_tokens: int = 0
    # Timing
    retrieval_time: float = 0.0  # Time to retrieve/generate the agent spec
    execution_time: float = 0.0  # Time to run the agent on the question
    latency_seconds: float = 0.0  # Total (retrieval + execution)
    # Agent info
    agent_id: str = ""
    agent_display_name: str = ""
    error: str | None = None


def _extract_answer(response_text: str, sample: BenchmarkSample) -> str:
    """Extract the answer from model response."""
    text = response_text.strip()

    # For MMLU and BIG-Bench multiple-choice: try to extract just the letter
    if sample.dataset_name in ("mmlu", "bigbench") and sample.choices:
        chr(64 + len(sample.choices)) if sample.choices else "D"
        valid_letters = [chr(65 + i) for i in range(len(sample.choices))] if sample.choices else ["A", "B", "C", "D"]

        # Look for a standalone letter
        for line in text.split("\n"):
            line = line.strip().upper()
            if line in valid_letters:
                return line
            # Handle "A." or "(A)" patterns
            for letter in valid_letters:
                if line.startswith((f"{letter}.", f"({letter})")) or line == f"**{letter}**":
                    return letter

        # Check first character
        if text and text[0].upper() in valid_letters:
            return text[0].upper()

        # Last resort: look for any single letter in the response
        for ch in text.upper():
            if ch in valid_letters:
                return ch

    # For BBH / free-form: return the full cleaned text
    # Try to extract from common patterns like "The answer is X"
    lower = text.lower()
    for prefix in ["the answer is ", "answer: ", "answer is "]:
        if prefix in lower:
            idx = lower.index(prefix) + len(prefix)
            return text[idx:].strip().rstrip(".")

    return text


def _check_correct(predicted: str, correct: str, dataset_name: str) -> bool:
    """Check if the predicted answer matches the correct answer."""
    pred = predicted.strip().lower()
    corr = correct.strip().lower()

    if dataset_name == "mmlu":
        # For MMLU, compare just the letter
        return pred[:1] == corr[:1]

    if dataset_name == "bigbench":
        # BIG-Bench multiple-choice: compare letter or full text
        if len(corr) == 1 and corr.isalpha():
            return pred[:1] == corr[:1]
        # Full text comparison
        if pred == corr:
            return True
        return bool(corr in pred or pred in corr)

    # For BBH, more flexible matching
    if pred == corr:
        return True
    if corr in pred or pred in corr:
        return True
    # Handle True/False
    if corr in ("true", "false") and pred in ("true", "false"):
        return pred == corr
    # Handle (A), (B) etc.
    if corr.startswith("(") and corr.endswith(")"):
        inner = corr[1:-1]
        return pred == inner or pred.startswith(inner)

    return False


def _build_dimas_answer_prompt(agent_spec: dict, question: str) -> str:
    """
    Build the instruction that DiMAS will use as the task query.

    DiMAS sends the task query + agent persona/description to the LLM.
    We embed the benchmark question into the task query.
    """
    return (
        "Answer the following question.\n"
        "For multiple-choice questions respond with ONLY the letter (A, B, C, or D).\n"
        "For free-form questions give a concise answer.\n"
        "Do NOT explain your reasoning.\n\n"
        f"{question}"
    )


def run_agent_on_sample(
    agent_spec: dict,
    sample: BenchmarkSample,
    client: OpenAI,
    model: str = "gpt-oss",
    temperature: float = 0.1,
    max_tokens: int = 256,
    timeout: int = 120,
) -> AgentAnswer:
    """
    Run a single agent on a single benchmark sample via DiMAS framework.

    Creates a single-agent RoleGraph using GraphBuilder, then executes it
    with MACPRunner. The agent's persona/description come from the agent_spec
    generated by retrieval/RAG/AutoGen.

    Args:
        agent_spec: Agent specification dict (from retrieval, RAG, or AutoGen).
        sample: The benchmark question.
        client: OpenAI client configured for the API.
        model: Model name.
        temperature: Sampling temperature.
        max_tokens: Max tokens for response.
        timeout: Request timeout.

    Returns:
        AgentAnswer with results and metrics.

    """
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig

    agent_id = agent_spec.get("agent_id", "agent")
    display_name = agent_spec.get("display_name", "AI Assistant")
    persona = agent_spec.get("persona", "")
    description = agent_spec.get("description", "")
    tools = agent_spec.get("tools", [])

    # Build the task query for DiMAS
    task_query = _build_dimas_answer_prompt(agent_spec, sample.question)

    # --- Track execution tokens via a wrapper around the OpenAI client ---
    exec_tokens = {"prompt": 0, "completion": 0, "total": 0}

    def llm_caller(prompt: str) -> str:
        """LLM caller for DiMAS MACPRunner that tracks token usage."""

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = retry_on_network_error(_call)
        # Track tokens
        usage = response.usage
        if usage:
            exec_tokens["prompt"] += usage.prompt_tokens
            exec_tokens["completion"] += usage.completion_tokens
            exec_tokens["total"] += usage.total_tokens

        content = response.choices[0].message.content or ""
        if not content:
            msg = response.choices[0].message
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                content = msg.reasoning_content
        return content

    t0 = time.perf_counter()
    try:
        # Step 1: Build single-agent graph via DiMAS GraphBuilder
        builder_config = BuilderConfig(
            include_task_node=True,
            validate=True,
        )
        builder = GraphBuilder(builder_config)

        # Add task node with the benchmark question
        builder.add_task(
            task_id="__task__",
            query=task_query,
            description="Benchmark question to answer",
        )

        # Add the single agent with its spec from LLM-Agent-Factory
        builder.add_agent(
            agent_id=agent_id,
            display_name=display_name,
            persona=persona,
            description=description,
            llm_backbone=model,
            base_url=str(client.base_url),
            api_key=client.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools if isinstance(tools, list) else [],
        )

        # Connect task to agent
        builder.connect_task_to_agents(agent_ids=[agent_id], bidirectional=False)

        # Build the RoleGraph
        graph = builder.build()

        # Step 2: Run via MACPRunner
        runner = MACPRunner(
            llm_caller=llm_caller,
            config=RunnerConfig(
                timeout=float(timeout),
                adaptive=False,
                update_states=True,
                broadcast_task_to_all=True,
            ),
        )

        result = runner.run_round(graph, final_agent_id=agent_id)

        execution_time = time.perf_counter() - t0

        # ── Use DiMAS MACPResult metrics ──────────────────────────────
        # result.total_tokens  — DiMAS-estimated tokens (word-based)
        # result.total_time    — DiMAS-measured execution time
        # result.metrics       — ExecutionMetrics with detailed breakdown
        #
        # We prefer exact API tokens (exec_tokens) but also store DiMAS
        # metrics for cross-validation.
        dimas_total_tokens = result.total_tokens
        dimas_total_time = result.total_time

        # Use exact API tokens if available, fall back to DiMAS estimate
        final_prompt = exec_tokens["prompt"]
        final_completion = exec_tokens["completion"]
        final_total = exec_tokens["total"]
        if final_total == 0 and dimas_total_tokens > 0:
            # API didn't return usage — use DiMAS estimate
            final_total = dimas_total_tokens

        # Use DiMAS execution time if our wall-clock is off
        if dimas_total_time > 0:
            execution_time = min(execution_time, dimas_total_time + 0.01)

        # Extract the answer from DiMAS result
        response_text = result.final_answer or ""
        if not response_text and result.messages:
            # Try to get from agent messages
            response_text = result.messages.get(agent_id, "")

        predicted = _extract_answer(response_text, sample)
        is_correct = _check_correct(predicted, sample.correct_answer, sample.dataset_name)

        return AgentAnswer(
            sample_id=sample.sample_id,
            predicted_answer=predicted,
            correct_answer=sample.correct_answer,
            is_correct=is_correct,
            prompt_tokens=final_prompt,
            completion_tokens=final_completion,
            total_tokens=final_total,
            execution_time=execution_time,
            latency_seconds=execution_time,
            agent_id=agent_id,
            agent_display_name=display_name,
        )

    except Exception as e:
        execution_time = time.perf_counter() - t0
        return AgentAnswer(
            sample_id=sample.sample_id,
            predicted_answer="",
            correct_answer=sample.correct_answer,
            is_correct=False,
            prompt_tokens=exec_tokens["prompt"],
            completion_tokens=exec_tokens["completion"],
            total_tokens=exec_tokens["total"],
            execution_time=execution_time,
            latency_seconds=execution_time,
            agent_id=agent_id,
            agent_display_name=display_name,
            error=str(e),
        )
