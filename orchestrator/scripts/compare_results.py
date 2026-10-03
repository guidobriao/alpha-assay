#!/usr/bin/env python3
"""
Compare reproduced results against paper's reported values.

Usage:
    python compare_results.py --paper results.json --reproduced results.json
    python compare_results.py --interactive

Results JSON format:
{
    "paper": {
        "accuracy": 95.2,
        "f1": 93.1,
        "loss": 0.123
    },
    "reproduced": {
        "accuracy": 94.8,
        "f1": 91.5,
        "loss": 0.145
    }
}
"""

import argparse
import json
import sys


# Tolerance thresholds
THRESHOLDS = {
    # Metric name patterns -> (PASS threshold, WARN threshold)
    "accuracy": (1.0, 3.0),
    "acc": (1.0, 3.0),
    "f1": (1.0, 3.0),
    "precision": (1.0, 3.0),
    "recall": (1.0, 3.0),
    "bleu": (1.0, 3.0),
    "rouge": (1.0, 3.0),
    "map": (1.0, 3.0),
    "miou": (1.0, 3.0),
    "iou": (1.0, 3.0),
    "top1": (1.0, 3.0),
    "top5": (0.5, 1.5),
    "loss": (5.0, 15.0),  # percentage
    "perplexity": (5.0, 20.0),
    "ppl": (5.0, 20.0),
}


def get_threshold(metric_name):
    """Get tolerance thresholds for a metric."""
    name_lower = metric_name.lower()
    for pattern, thresholds in THRESHOLDS.items():
        if pattern in name_lower:
            return thresholds
    # Default: 1% PASS, 3% WARN
    return (1.0, 3.0)


def compare_metric(name, paper_val, repro_val):
    """Compare a single metric and return status."""
    pass_thresh, warn_thresh = get_threshold(name)

    # For loss-like metrics (lower is better), check if paper uses this convention
    is_lower_better = any(k in name.lower() for k in ["loss", "perplexity", "ppl", "error"])

    if is_lower_better:
        delta = repro_val - paper_val  # positive = worse
        delta_pct = (delta / paper_val) * 100 if paper_val != 0 else 0
    else:
        delta = repro_val - paper_val  # positive = better
        delta_pct = (abs(delta) / paper_val) * 100 if paper_val != 0 else 0

    if delta_pct <= pass_thresh:
        status = "PASS"
    elif delta_pct <= warn_thresh:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "name": name,
        "paper": paper_val,
        "reproduced": repro_val,
        "delta": delta,
        "delta_pct": delta_pct,
        "status": status,
        "is_lower_better": is_lower_better,
    }


def print_report(results):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("REPRODUCTION RESULT COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Metric':<20} {'Paper':>10} {'Reproduced':>12} {'Delta':>10} {'Delta%':>8} {'Status':>8}")
    print("-" * 70)

    all_pass = True
    any_fail = False

    for r in results:
        direction = "↓" if r["is_lower_better"] else "↑"
        delta_str = f"{r['delta']:+.2f}"

        status_marker = {
            "PASS": "✓ PASS",
            "WARN": "⚠ WARN",
            "FAIL": "✗ FAIL",
        }[r["status"]]

        print(f"{r['name']:<20} {r['paper']:>10.2f} {r['reproduced']:>12.2f} {delta_str:>10} {r['delta_pct']:>7.1f}% {status_marker:>8}")

        if r["status"] != "PASS":
            all_pass = False
        if r["status"] == "FAIL":
            any_fail = True

    print("-" * 70)

    # Summary
    if all_pass:
        print("\n✓ ALL METRICS PASS — Successful reproduction!")
    elif any_fail:
        print("\n✗ SOME METRICS FAIL — Review implementation for issues")
    else:
        print("\n⚠ SOME METRICS WARN — Minor discrepancies, likely acceptable")

    # Legend
    print("\nTolerance: PASS (within 1%), WARN (1-3%), FAIL (>3%)")
    print("  ↓ = lower is better (loss, perplexity)")
    print("  ↑ = higher is better (accuracy, F1, BLEU)")
    print("=" * 70)

    return not any_fail


def interactive_mode():
    """Interactive mode for entering results."""
    print("=== Interactive Result Comparison ===")
    print("Enter metrics one by one (empty line to finish):\n")

    paper_vals = {}
    repro_vals = {}

    while True:
        name = input("Metric name (e.g., accuracy): ").strip()
        if not name:
            break
        paper_val = float(input(f"  Paper value for {name}: "))
        repro_val = float(input(f"  Reproduced value for {name}: "))
        paper_vals[name] = paper_val
        repro_vals[name] = repro_val
        print()

    return paper_vals, repro_vals


def main():
    parser = argparse.ArgumentParser(description="Compare paper reproduction results")
    parser.add_argument("--paper", help="JSON file with paper's reported results")
    parser.add_argument("--reproduced", help="JSON file with reproduced results")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        paper_vals, repro_vals = interactive_mode()
    elif args.paper and args.reproduced:
        with open(args.paper) as f:
            paper_vals = json.load(f)
        with open(args.reproduced) as f:
            repro_vals = json.load(f)
    else:
        # Try to load from reproduction_results.json
        try:
            with open("reproduction_results.json") as f:
                data = json.load(f)
            paper_vals = data.get("paper", {})
            repro_vals = data.get("reproduced", {})
        except FileNotFoundError:
            print("ERROR: Provide --paper and --reproduced JSON files, or use --interactive")
            sys.exit(1)

    # Compare
    results = []
    for metric in paper_vals:
        if metric in repro_vals:
            results.append(compare_metric(metric, paper_vals[metric], repro_vals[metric]))
        else:
            print(f"WARNING: Metric '{metric}' in paper but not in reproduced results")

    if not results:
        print("ERROR: No matching metrics found to compare")
        sys.exit(1)

    # Print report
    success = print_report(results)

    # Save report
    report = {
        "comparisons": results,
        "all_pass": success,
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r["status"] == "PASS"),
            "warn": sum(1 for r in results if r["status"] == "WARN"),
            "fail": sum(1 for r in results if r["status"] == "FAIL"),
        },
    }

    with open("comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: comparison_report.json")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
