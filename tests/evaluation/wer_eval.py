"""Word Error Rate (WER) evaluation for transcription quality.

Compares machine-generated transcripts against human reference transcripts
using the jiwer library. Reports WER, MER, WIL, and WIP metrics.

Usage:
    python -m tests.evaluation.wer_eval --reference ref.txt --hypothesis hyp.txt
    python -m tests.evaluation.wer_eval --reference-dir refs/ --hypothesis-dir hyps/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer


def compute_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute Word Error Rate and related metrics.

    Args:
        reference: Ground truth transcript text.
        hypothesis: Machine-generated transcript text.

    Returns:
        Dictionary with wer, mer, wil, wip metrics.
    """
    # Normalize text
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    measures = jiwer.compute_measures(
        reference,
        hypothesis,
        truth_transform=transform,
        hypothesis_transform=transform,
    )

    return {
        "wer": round(measures["wer"], 4),
        "mer": round(measures["mer"], 4),
        "wil": round(measures["wil"], 4),
        "wip": round(measures["wip"], 4),
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
    }


def evaluate_file_pair(
    reference_path: Path,
    hypothesis_path: Path,
) -> dict[str, float]:
    """Evaluate WER for a single file pair.

    Args:
        reference_path: Path to reference transcript file.
        hypothesis_path: Path to hypothesis transcript file.

    Returns:
        WER metrics dictionary.
    """
    ref_text = reference_path.read_text(encoding="utf-8").strip()
    hyp_text = hypothesis_path.read_text(encoding="utf-8").strip()

    if not ref_text:
        raise ValueError(f"Empty reference file: {reference_path}")
    if not hyp_text:
        raise ValueError(f"Empty hypothesis file: {hypothesis_path}")

    return compute_wer(ref_text, hyp_text)


def evaluate_directory(
    reference_dir: Path,
    hypothesis_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate WER for all file pairs in matching directories.

    Files are matched by name (same stem). Supports .txt and .json formats.

    Args:
        reference_dir: Directory containing reference transcripts.
        hypothesis_dir: Directory containing hypothesis transcripts.

    Returns:
        Dictionary mapping filename to WER metrics.
    """
    results: dict[str, dict[str, float]] = {}

    ref_files = {f.stem: f for f in reference_dir.glob("*") if f.suffix in (".txt", ".json")}
    hyp_files = {f.stem: f for f in hypothesis_dir.glob("*") if f.suffix in (".txt", ".json")}

    matched = set(ref_files.keys()) & set(hyp_files.keys())

    if not matched:
        print(f"Warning: No matching files found between {reference_dir} and {hypothesis_dir}")
        return results

    for name in sorted(matched):
        ref_path = ref_files[name]
        hyp_path = hyp_files[name]

        # Handle JSON format (extract text from transcript structure)
        if ref_path.suffix == ".json":
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
            ref_text = _extract_text_from_transcript(ref_data)
        else:
            ref_text = ref_path.read_text(encoding="utf-8").strip()

        if hyp_path.suffix == ".json":
            hyp_data = json.loads(hyp_path.read_text(encoding="utf-8"))
            hyp_text = _extract_text_from_transcript(hyp_data)
        else:
            hyp_text = hyp_path.read_text(encoding="utf-8").strip()

        try:
            metrics = compute_wer(ref_text, hyp_text)
            results[name] = metrics
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def _extract_text_from_transcript(data: dict) -> str:
    """Extract full text from a transcript JSON structure."""
    if "segments" in data:
        segments = data["segments"]
        return " ".join(seg.get("text", "").strip() for seg in segments)
    elif isinstance(data, str):
        return data
    return str(data)


def aggregate_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute aggregate statistics across multiple evaluations.

    Args:
        results: Per-file WER results.

    Returns:
        Aggregated metrics (mean, min, max).
    """
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"error": "No valid results to aggregate"}

    wers = [v["wer"] for v in valid.values()]
    return {
        "mean_wer": round(sum(wers) / len(wers), 4),
        "min_wer": round(min(wers), 4),
        "max_wer": round(max(wers), 4),
        "num_files": len(valid),
        "num_errors": len(results) - len(valid),
    }


def print_report(results: dict, aggregate: dict) -> None:
    """Print a formatted evaluation report."""
    print("=" * 70)
    print("TRANSCRIPTION QUALITY REPORT (Word Error Rate)")
    print("=" * 70)

    for name, metrics in sorted(results.items()):
        if "error" in metrics:
            print(f"\n  {name}: ERROR - {metrics['error']}")
        else:
            print(f"\n  {name}:")
            print(f"    WER:  {metrics['wer']:.2%}")
            print(f"    MER:  {metrics['mer']:.2%}")
            print(f"    WIL:  {metrics['wil']:.2%}")
            print(f"    WIP:  {metrics['wip']:.2%}")
            subs = metrics['substitutions']
            dels = metrics['deletions']
            ins = metrics['insertions']
            hits = metrics['hits']
            print(f"    Edits: {subs}S {dels}D {ins}I {hits}H")

    print("\n" + "-" * 70)
    print("AGGREGATE:")
    if "error" in aggregate:
        print(f"  {aggregate['error']}")
    else:
        print(f"  Mean WER: {aggregate['mean_wer']:.2%}")
        print(f"  Min WER:  {aggregate['min_wer']:.2%}")
        print(f"  Max WER:  {aggregate['max_wer']:.2%}")
        print(f"  Files:    {aggregate['num_files']} evaluated, {aggregate['num_errors']} errors")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transcription WER.")
    parser.add_argument("--reference", type=Path, help="Reference transcript file.")
    parser.add_argument("--hypothesis", type=Path, help="Hypothesis transcript file.")
    parser.add_argument("--reference-dir", type=Path, help="Directory of reference transcripts.")
    parser.add_argument("--hypothesis-dir", type=Path, help="Directory of hypothesis transcripts.")
    parser.add_argument("--output", type=Path, help="Save results as JSON.")

    args = parser.parse_args()

    if args.reference and args.hypothesis:
        metrics = evaluate_file_pair(args.reference, args.hypothesis)
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
