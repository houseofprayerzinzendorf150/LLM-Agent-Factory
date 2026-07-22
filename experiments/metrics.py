"""
Metrics computation and reporting for benchmark experiments.

Computes:
- Accuracy (overall and per-subject/task)
- Token usage (prompt, completion, total)
- Latency (mean, median, p95)
- Error rate
"""

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from experiments.agent_runner import AgentAnswer


def compute_metrics(results: list[AgentAnswer]) -> dict:
    """
    Compute comprehensive metrics from experiment results.

    Returns:
        Dict with accuracy, token usage, latency, and error metrics.

    """
    if not results:
        return {"error": "no results"}

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error)
    valid = total - errors

    # Execution token usage (running the agent on the question)
    prompt_tokens = [r.prompt_tokens for r in results if not r.error]
    completion_tokens = [r.completion_tokens for r in results if not r.error]
    total_tokens = [r.total_tokens for r in results if not r.error]

    # Generation token usage (RAG/AutoGen creating the agent spec)
    gen_prompt_tokens = [r.gen_prompt_tokens for r in results if not r.error]
    gen_completion_tokens = [r.gen_completion_tokens for r in results if not r.error]
    gen_total_tokens = [r.gen_total_tokens for r in results if not r.error]

    # Timing
    retrieval_times = [r.retrieval_time for r in results if not r.error and r.retrieval_time > 0]
    execution_times = [r.execution_time for r in results if not r.error and r.execution_time > 0]
    latencies = [r.latency_seconds for r in results if not r.error and r.latency_seconds > 0]

    # Per-subject accuracy (group by subject from sample_id)
    subject_results: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        # Extract subject from sample_id like "mmlu_abstract_algebra_0"
        parts = r.sample_id.split("_")
        if len(parts) >= 3:
            # Reconstruct subject (may contain underscores)
            dataset = parts[0]
            parts[-1]
            subject = "_".join(parts[1:-1])
            subject_results[f"{dataset}/{subject}"].append(r.is_correct)

    per_subject = {}
    for subj, corrects in sorted(subject_results.items()):
        per_subject[subj] = {
            "accuracy": sum(corrects) / len(corrects) if corrects else 0,
            "total": len(corrects),
            "correct": sum(corrects),
        }

    # Combined tokens = execution + generation
    combined_total = sum(total_tokens) + sum(gen_total_tokens)

    return {
        "total_samples": total,
        "valid_samples": valid,
        "errors": errors,
        "error_rate": errors / total if total else 0,
        "accuracy": correct / valid if valid else 0,
        "correct": correct,
        # Execution tokens (agent answering the question)
        "exec_tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "total": sum(total_tokens),
            "prompt_mean": statistics.mean(prompt_tokens) if prompt_tokens else 0,
            "completion_mean": statistics.mean(completion_tokens) if completion_tokens else 0,
            "total_mean": statistics.mean(total_tokens) if total_tokens else 0,
        },
        # Generation tokens (RAG/AutoGen creating the agent spec)
        "gen_tokens": {
            "prompt_total": sum(gen_prompt_tokens),
            "completion_total": sum(gen_completion_tokens),
            "total": sum(gen_total_tokens),
            "prompt_mean": statistics.mean(gen_prompt_tokens) if gen_prompt_tokens else 0,
            "completion_mean": statistics.mean(gen_completion_tokens) if gen_completion_tokens else 0,
            "total_mean": statistics.mean(gen_total_tokens) if gen_total_tokens else 0,
        },
        # Combined tokens (execution + generation)
        "tokens": {
            "total": combined_total,
            "total_mean": combined_total / valid if valid else 0,
        },
        # Timing breakdown
        "timing": {
            "retrieval_mean": statistics.mean(retrieval_times) if retrieval_times else 0,
            "execution_mean": statistics.mean(execution_times) if execution_times else 0,
        },
        "latency": {
            "mean": statistics.mean(latencies) if latencies else 0,
            "median": statistics.median(latencies) if latencies else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "total": sum(latencies),
        },
        "per_subject": per_subject,
    }


def print_metrics_table(metrics_dict: dict[str, dict]) -> None:
    """
    Print a formatted table of metrics for all experiments.

    Args:
        metrics_dict: Dict mapping experiment_id -> metrics dict

    """
    if not metrics_dict:
        return

    # Header

    for _exp_id, m in sorted(metrics_dict.items()):
        if "error" in m:
            continue

        m.get("accuracy", 0)
        m.get("total_samples", 0)
        m.get("errors", 0)
        m.get("exec_tokens", {}).get("total", 0)
        m.get("gen_tokens", {}).get("total", 0)
        m.get("tokens", {}).get("total", 0)
        m.get("timing", {}).get("retrieval_mean", 0)
        m.get("timing", {}).get("execution_mean", 0)
        m.get("latency", {}).get("mean", 0)

    # Summary: best accuracy
    if metrics_dict:
        best_exp = max(
            ((k, v) for k, v in metrics_dict.items() if "accuracy" in v),
            key=lambda x: x[1]["accuracy"],
            default=None,
        )
        if best_exp:
            pass

        # Most efficient (lowest total tokens per correct answer)
        efficient = []
        for k, v in metrics_dict.items():
            if v.get("correct", 0) > 0:
                tokens_per_correct = v["tokens"]["total"] / v["correct"]
                efficient.append((k, tokens_per_correct))
        if efficient:
            min(efficient, key=lambda x: x[1])


def print_per_subject_breakdown(metrics: dict, experiment_id: str) -> None:
    """Print per-subject accuracy breakdown."""
    per_subject = metrics.get("per_subject", {})
    if not per_subject:
        return

    for _subj, data in sorted(per_subject.items(), key=lambda x: -x[1]["accuracy"]):
        data["accuracy"]
        data["total"]


def save_metrics_report(
    all_results: dict[str, list[AgentAnswer]],
    output_dir: Path,
) -> None:
    """
    Save comprehensive metrics report to JSON and text files.

    Args:
        all_results: Dict mapping experiment_id -> list of AgentAnswer
        output_dir: Directory to save reports

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics for all experiments
    all_metrics = {}
    for exp_id, results in all_results.items():
        if results:
            all_metrics[exp_id] = compute_metrics(results)

    # Save JSON report
    json_path = output_dir / "metrics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)

    # Save detailed text report
    txt_path = output_dir / "metrics_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write(
            f"{'Experiment':<40} {'Acc':>7} {'N':>6} {'Err':>5} "
            f"{'ExecTok':>10} {'GenTok':>10} {'TotalTok':>10} "
            f"{'GenT(s)':>8} {'ExecT(s)':>8} {'TotT(s)':>8}\n"
        )
        f.write("-" * 130 + "\n")

        for exp_id, m in sorted(all_metrics.items()):
            if "error" in m:
                f.write(f"{exp_id:<40} {'N/A':>7}\n")
                continue

            acc = m.get("accuracy", 0)
            total = m.get("total_samples", 0)
            errors = m.get("errors", 0)
            exec_tok = m.get("exec_tokens", {}).get("total", 0)
            gen_tok = m.get("gen_tokens", {}).get("total", 0)
            total_tok = m.get("tokens", {}).get("total", 0)
            gen_time = m.get("timing", {}).get("retrieval_mean", 0)
            exec_time = m.get("timing", {}).get("execution_mean", 0)
            lat_mean = m.get("latency", {}).get("mean", 0)

            f.write(
                f"{exp_id:<40} {acc:>6.1%} {total:>6} {errors:>5} "
                f"{exec_tok:>10,} {gen_tok:>10,} {total_tok:>10,} "
                f"{gen_time:>7.2f}s {exec_time:>7.2f}s {lat_mean:>7.2f}s\n"
            )

        f.write("-" * 130 + "\n\n")

        # Detailed per-experiment breakdown
        for exp_id, m in sorted(all_metrics.items()):
            if "error" in m:
                continue

            f.write(f"\n{'=' * 60}\n")
            f.write(f"Experiment: {exp_id}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"  Accuracy:        {m['accuracy']:.1%} ({m['correct']}/{m['valid_samples']})\n")
            f.write(f"  Error rate:      {m['error_rate']:.1%} ({m['errors']}/{m['total_samples']})\n")
            f.write(f"  Exec tokens:     {m['exec_tokens']['total']:,}\n")
            f.write(f"  Gen tokens:      {m['gen_tokens']['total']:,}\n")
            f.write(f"  Total tokens:    {m['tokens']['total']:,}\n")
            f.write(f"  Gen time (avg):  {m['timing']['retrieval_mean']:.2f}s\n")
            f.write(f"  Exec time (avg): {m['timing']['execution_mean']:.2f}s\n")
            f.write(f"  Mean latency:    {m['latency']['mean']:.2f}s\n")
            f.write(f"  P95 latency:     {m['latency']['p95']:.2f}s\n\n")

            per_subject = m.get("per_subject", {})
            if per_subject:
                f.write(f"  {'Subject':<45} {'Acc':>7} {'N':>5}\n")
                f.write(f"  {'-' * 60}\n")
                f.writelines(
                    f"  {subj:<45} {data['accuracy']:>6.1%} {data['total']:>5}\n"
                    for subj, data in sorted(per_subject.items(), key=lambda x: -x[1]["accuracy"])
                )
                f.write("\n")

    # Save raw results as JSONL
    for exp_id, results in all_results.items():
        if results:
            jsonl_path = output_dir / f"{exp_id}_results.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results)


def compare_modes(all_metrics: dict[str, dict]) -> None:
    """Compare retrieval vs RAG vs AutoGen modes."""
    mode_metrics: dict[str, list[dict]] = defaultdict(list)

    for exp_id, m in all_metrics.items():
        if "error" in m:
            continue
        if "ret_" in exp_id:
            mode_metrics["retrieval"].append(m)
        elif "rag_" in exp_id:
            mode_metrics["rag"].append(m)
        elif "autogen" in exp_id:
            mode_metrics["autogen"].append(m)

    for metrics_list in mode_metrics.values():
        if not metrics_list:
            continue

        statistics.mean(m["accuracy"] for m in metrics_list)
        statistics.mean(m["tokens"]["total"] for m in metrics_list)
        statistics.mean(m["latency"]["mean"] for m in metrics_list)
        sum(m["errors"] for m in metrics_list)
