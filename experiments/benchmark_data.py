"""
Benchmark data loaders for MMLU, BIG-Bench, and BIG-Bench Hard datasets.

Loads questions from HuggingFace datasets and formats them for agent evaluation.
Each sample contains: question text, choices, correct answer, subject/task.
"""

import random
from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class BenchmarkSample:
    """A single benchmark question."""

    dataset_name: str  # "mmlu" or "bbh"
    subject: str  # e.g. "abstract_algebra" or "boolean_expressions"
    question: str  # The full question text
    choices: list[str]  # Answer options (for MMLU: A/B/C/D)
    correct_answer: str  # The correct answer letter or text
    sample_id: str  # Unique identifier for checkpoint/resume
    raw: dict = field(default_factory=dict, repr=False)


# ── MMLU ──────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

ANSWER_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(
    subjects: list[str] | None = None,
    max_samples_per_subject: int | None = None,
    split: str = "test",
    seed: int = 42,
) -> list[BenchmarkSample]:
    """
    Load MMLU dataset from HuggingFace.

    Args:
        subjects: List of subjects to load (None = all 57).
        max_samples_per_subject: Limit samples per subject for faster testing.
        split: Dataset split to use.
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkSample objects.

    """
    subjects = subjects or MMLU_SUBJECTS
    samples: list[BenchmarkSample] = []
    rng = random.Random(seed)

    for _i, subj in enumerate(subjects, 1):
        try:
            ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception:
            try:
                ds = load_dataset("lukaemon/mmlu", subj, split=split)
            except Exception:
                continue

        items = list(ds)
        if max_samples_per_subject and len(items) > max_samples_per_subject:
            items = rng.sample(items, max_samples_per_subject)

        for idx, row in enumerate(items):
            question_text = row["question"]
            choices = (
                row["choices"]
                if "choices" in row
                else [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
            )
            # answer is an int index (0-3) in cais/mmlu
            answer_idx = row["answer"]
            correct = ANSWER_LETTERS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx).strip().upper()

            # Build formatted question
            formatted_q = f"{question_text}\n"
            for letter_i, ch in enumerate(choices):
                formatted_q += f"{ANSWER_LETTERS[letter_i]}. {ch}\n"

            samples.append(
                BenchmarkSample(
                    dataset_name="mmlu",
                    subject=subj,
                    question=formatted_q.strip(),
                    choices=choices,
                    correct_answer=correct,
                    sample_id=f"mmlu_{subj}_{idx}",
                    raw=dict(row),
                )
            )

    return samples


# ── BIG-Bench Hard ────────────────────────────────────────────────────────────

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def load_bbh(
    tasks: list[str] | None = None,
    max_samples_per_task: int | None = None,
    seed: int = 42,
) -> list[BenchmarkSample]:
    """
    Load BIG-Bench Hard dataset from HuggingFace.

    Args:
        tasks: List of tasks to load (None = all).
        max_samples_per_task: Limit samples per task.
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkSample objects.

    """
    tasks = tasks or BBH_TASKS
    samples: list[BenchmarkSample] = []
    rng = random.Random(seed)

    for _i, task in enumerate(tasks, 1):
        try:
            # Primary source: lukaemon/bbh (maveriq/bigbenchhard is deprecated)
            ds = load_dataset("lukaemon/bbh", task, split="test")
            data = list(ds)
        except Exception:
            try:
                ds = load_dataset("lukaemon/bbh", task, split="train")
                data = list(ds)
            except Exception:
                continue

        if max_samples_per_task and len(data) > max_samples_per_task:
            data = rng.sample(data, max_samples_per_task)

        for idx, row in enumerate(data):
            question_text = row.get("input", row.get("question", ""))
            target = row.get("target", row.get("answer", ""))

            samples.append(
                BenchmarkSample(
                    dataset_name="bbh",
                    subject=task,
                    question=question_text.strip(),
                    choices=[],  # BBH is mostly free-form
                    correct_answer=str(target).strip(),
                    sample_id=f"bbh_{task}_{idx}",
                    raw=dict(row),
                )
            )

    return samples


# ── BIG-Bench (regular) ───────────────────────────────────────────────────────

BIGBENCH_TASKS = [
    "abstract_narrative_understanding",
    "anachronisms",
    "analogical_similarity",
    "causal_judgment",
    "cause_and_effect",
    "elementary_math_qa",
    "epistemic_reasoning",
    "general_knowledge",
    "logical_args",
    "logical_fallacy_detection",
    "logical_sequence",
    "movie_dialog_same_or_different",
    "novel_concepts",
    "odd_one_out",
    "play_dialog_same_or_different",
    "presuppositions_as_nli",
    "riddle_sense",
    "strange_stories",
    "strategyqa",
    "vitaminc_fact_verification",
]


def load_bigbench(
    tasks: list[str] | None = None,
    max_samples_per_task: int | None = None,
    seed: int = 42,
) -> list[BenchmarkSample]:
    """
    Load regular BIG-Bench dataset from HuggingFace (tasksource/bigbench).

    Unlike BBH (free-form), regular BIG-Bench has multiple-choice format
    with 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores'.

    Args:
        tasks: List of tasks to load (None = default subset).
        max_samples_per_task: Limit samples per task.
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkSample objects.

    """
    tasks = tasks or BIGBENCH_TASKS
    samples: list[BenchmarkSample] = []
    rng = random.Random(seed)

    for i, task in enumerate(tasks, 1):
        try:
            ds = load_dataset("tasksource/bigbench", task, split="train")
            data = list(ds)
        except Exception:
            try:
                ds = load_dataset("tasksource/bigbench", task, split="validation")
                data = list(ds)
            except Exception:
                continue

        if max_samples_per_task and len(data) > max_samples_per_task:
            data = rng.sample(data, max_samples_per_task)

        for idx, row in enumerate(data):
            question_text = row.get("inputs", "")
            targets = row.get("targets", [])
            mc_targets = row.get("multiple_choice_targets", [])
            mc_scores = row.get("multiple_choice_scores", [])

            # Determine correct answer
            if mc_targets and mc_scores:
                # Multiple-choice format
                correct_idx = None
                for i, score in enumerate(mc_scores):
                    if score == 1:
                        correct_idx = i
                        break

                if correct_idx is not None and correct_idx < len(mc_targets):
                    correct_answer = mc_targets[correct_idx]
                elif targets:
                    correct_answer = targets[0] if isinstance(targets, list) else str(targets)
                else:
                    continue

                # Format as multiple-choice question
                choices = mc_targets
                formatted_q = f"{question_text}\n"
                for i, ch in enumerate(choices):
                    letter = chr(65 + i)  # A, B, C, ...
                    formatted_q += f"{letter}. {ch}\n"

                # Store correct as letter
                correct_letter = chr(65 + correct_idx) if correct_idx is not None else str(correct_answer)
            else:
                # Free-form
                correct_answer = targets[0] if isinstance(targets, list) and targets else str(targets)
                formatted_q = question_text
                choices = []
                correct_letter = correct_answer

            samples.append(
                BenchmarkSample(
                    dataset_name="bigbench",
                    subject=task,
                    question=formatted_q.strip(),
                    choices=choices,
                    correct_answer=correct_letter.strip(),
                    sample_id=f"bigbench_{task}_{idx}",
                    raw=dict(row),
                )
            )

    return samples


def load_all_benchmarks(
    mmlu_subjects: list[str] | None = None,
    bbh_tasks: list[str] | None = None,
    bigbench_tasks: list[str] | None = None,
    max_samples_per_subject: int | None = None,
    seed: int = 42,
) -> dict[str, list[BenchmarkSample]]:
    """Load MMLU, BBH, and BIG-Bench datasets."""
    return {
        "mmlu": load_mmlu(mmlu_subjects, max_samples_per_subject, seed=seed),
        "bbh": load_bbh(bbh_tasks, max_samples_per_subject, seed=seed),
        "bigbench": load_bigbench(bigbench_tasks, max_samples_per_subject, seed=seed),
    }
