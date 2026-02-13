"""ROUGE evaluation for note generation quality.

Compares machine-generated structured notes against human-written reference
notes using ROUGE-1, ROUGE-2, and ROUGE-L metrics.

Usage:
    python -m tests.evaluation.rouge_eval --reference ref.txt --hypothesis hyp.txt
    python -m tests.evaluation.rouge_eval --reference-dir refs/ --hypothesis-dir hyps/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer


def compute_rouge(
    reference: str,
    hypothesis: str,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute ROUGE scores between reference and hypothesis text.

    Args:
        reference: Ground truth text.
        hypothesis: Generated text.
        metrics: ROUGE metrics to compute. Defaults to ROUGE-1, ROUGE-2, ROUGE-L.

    Returns:
        Dictionary mapping metric name to precision, recall, fmeasure.
    """
    if metrics is None:
        metrics = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    result: dict[str, dict[str, float]] = {}
    for metric_name, score in scores.items():
        result[metric_name] = {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fmeasure": round(score.fmeasure, 4),
        }

    return result


def evaluate_notes(
    reference_path: Path,
    hypothesis_path: Path,
) -> dict[str, Any]:
    """Evaluate ROUGE for a pair of note files.

    Supports plain text (.txt) and JSON (.json) formats.

    Args:
        reference_path: Path to reference notes.
        hypothesis_path: Path to generated notes.

    Returns:
        ROUGE metrics dictionary.
    """
    ref_text = _load_text(reference_path)
    hyp_text = _load_text(hypothesis_path)

    if not ref_text.strip():
        raise ValueError(f"Empty reference file: {reference_path}")
    if not hyp_text.strip():
        raise ValueError(f"Empty hypothesis file: {hypothesis_path}")

    return compute_rouge(ref_text, hyp_text)


def _load_text(path: Path) -> str:
    """Load text from a file, handling JSON format."""
    content = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(content)
        return _extract_notes_text(data)

    return content.strip()


def _extract_notes_text(data: dict) -> str:
    """Extract all text content from a structured notes JSON."""
    parts: list[str] = []

    # Handle GeneratedOutput format
    if "structured_notes" in data:
        notes = data["structured_notes"]
        if "title" in notes:
            parts.append(notes["title"])
        if "summary" in notes:
            parts.append(notes["summary"])
        for section in notes.get("sections", []):
            if "heading" in section:
                parts.append(section["heading"])
            for point in section.get("key_points", []):
                parts.append(point)

    # Handle timestamps
    for ts in data.get("timestamps", []):
        if "label" in ts:
            parts.append(ts["label"])

    # Handle action items
    for item in data.get("action_items", []):
        if "action" in item:
            parts.append(item["action"])

    return " ".join(parts) if parts else json.dumps(data)


def evaluate_directory(
    reference_dir: Path,
    hypothesis_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Evaluate ROUGE for all matching file pairs in directories.

    Args:
        reference_dir: Directory of reference notes.
        hypothesis_dir: Directory of generated notes.

    Returns:
        Per-file ROUGE results.
    """
    results: dict[str, dict[str, Any]] = {}

    ref_files = {f.stem: f for f in reference_dir.glob("*") if f.suffix in (".txt", ".json")}
    hyp_files = {f.stem: f for f in hypothesis_dir.glob("*") if f.suffix in (".txt", ".json")}

    matched = set(ref_files.keys()) & set(hyp_files.keys())

    if not matched:
        print(f"Warning: No matching files found.")
        return results

    for name in sorted(matched):
        try:
            metrics = evaluate_notes(ref_files[name], hyp_files[name])
            results[name] = metrics
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def aggregate_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate ROUGE statistics across files.

    Args:
        results: Per-file ROUGE results.

    Returns:
        Aggregated metrics.
    """
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"error": "No valid results to aggregate"}

    metrics_keys = ["rouge1", "rouge2", "rougeL"]
    aggregate: dict[str, Any] = {"num_files": len(valid)}

    for metric in metrics_keys:
        fscores = [v[metric]["fmeasure"] for v in valid.values() if metric in v]
        if fscores:
            aggregate[f"{metric}_mean_f"] = round(sum(fscores) / len(fscores), 4)
            aggregate[f"{metric}_min_f"] = round(min(fscores), 4)
            aggregate[f"{metric}_max_f"] = round(max(fscores), 4)

    return aggregate


def print_report(results: dict, aggregate: dict) -> None:
    """Print formatted ROUGE evaluation report."""
    print("=" * 70)
    print("NOTE GENERATION QUALITY REPORT (ROUGE)")
    print("=" * 70)

    for name, metrics in sorted(results.items()):
        if "error" in metrics:
            print(f"\n  {name}: ERROR - {metrics['error']}")
        else:
            print(f"\n  {name}:")
            for metric_name in ["rouge1", "rouge2", "rougeL"]:
                if metric_name in metrics:
                    m = metrics[metric_name]
                    print(
                        f"    {metric_name:8s}: "
                        f"P={m['precision']:.2%}  "
                        f"R={m['recall']:.2%}  "
                        f"F={m['fmeasure']:.2%}"
                    )

    print("\n" + "-" * 70)
    print("AGGREGATE:")
    if "error" in aggregate:
        print(f"  {aggregate['error']}")
    else:
        for key in ["rouge1_mean_f", "rouge2_mean_f", "rougeL_mean_f"]:
            if key in aggregate:
                print(f"  {key}: {aggregate[key]:.2%}")
        print(f"  Files evaluated: {aggregate.get('num_files', 0)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate note quality with ROUGE.")
    parser.add_argument("--reference", type=Path, help="Reference notes file.")
    parser.add_argument("--hypothesis", type=Path, help="Generated notes file.")
    parser.add_argument("--reference-dir", type=Path, help="Directory of reference notes.")
    parser.add_argument("--hypothesis-dir", type=Path, help="Directory of generated notes.")
    parser.add_argument("--output", type=Path, help="Save results as JSON.")

    args = parser.parse_args()

    if args.reference and args.hypothesis:
        metrics = evaluate_notes(args.reference, args.hypothesis)
        print(json.dumps(metrics, indent=2))
        results = {"single": metrics}
    elif args.reference_dir and args.hypothesis_dir:
        results = evaluate_directory(args.reference_dir, args.hypothesis_dir)
        aggregate = aggregate_results(results)
        print_report(results, aggregate)
        results["_aggregate"] = aggregate
    else:
        parser.error("Provide either --reference/--hypothesis or --reference-dir/--hypothesis-dir")
        return

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
