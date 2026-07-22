"""Integration checks for the official gMAS submodule."""

from types import SimpleNamespace

from experiments.agent_runner import run_agent_on_sample
from experiments.benchmark_data import BenchmarkSample


class _StubCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=1, total_tokens=12),
            choices=[SimpleNamespace(message=SimpleNamespace(content="B"))],
        )


class _StubClient:
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.base_url = "https://api.example.com/v1"
        self.chat = SimpleNamespace(completions=_StubCompletions())


def test_agent_runner_uses_current_gmas_api() -> None:
    sample = BenchmarkSample(
        dataset_name="mmlu",
        subject="integration",
        question="Which option is correct?",
        choices=["first", "second", "third", "fourth"],
        correct_answer="B",
        sample_id="gmas-integration-1",
    )
    agent_spec = {
        "agent_id": "reviewer",
        "display_name": "Reviewer",
        "persona": "A careful reviewer",
        "description": "Selects the correct option.",
        "tools": [],
    }

    answer = run_agent_on_sample(agent_spec, sample, _StubClient(), timeout=5)

    assert answer.error is None
    assert answer.predicted_answer == "B"
    assert answer.is_correct is True
    assert answer.total_tokens == 12
