#!/usr/bin/env python3
"""
Test the optimized prompt against existing batch results.

This script:
1. Loads existing successful assessments (with audio files)
2. Re-runs assessment with the new optimized prompt
3. Compares old vs new scores
4. Generates a comparison report
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import statistics

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.assessment.openai_assessment import assess_pronunciation_openai, load_optimized_prompt

# Directories
PROJECT_ROOT = Path(__file__).parent
BATCH_RESULTS_DIR = PROJECT_ROOT / "data" / "batch_assessment" / "results"
AUDIO_DIR = PROJECT_ROOT / "data" / "batch_assessment" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / f"optimized_prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
CLOUD_SUMMARY = PROJECT_ROOT / "data" / "reports" / "batch_results_20260416_130305" / "summary.json"


def load_existing_results(limit: int = 20):
    """Load existing successful batch results that have audio files."""
    results = []

    for json_file in sorted(BATCH_RESULTS_DIR.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        # Skip failed assessments
        if "error" in data or "openai_final_score" not in data:
            continue

        # Check if audio file exists
        audio_file = AUDIO_DIR / f"{json_file.stem}.wav"
        if not audio_file.exists():
            # Try alternate location
            audio_file = PROJECT_ROOT / "legacy" / "batch_assessment" / "audio" / f"{json_file.stem}.wav"

        if not audio_file.exists():
            continue

        results.append({
            "file": json_file.name,
            "audio_path": str(audio_file),
            "name": data.get("name", json_file.stem),
            "existing_score": data.get("existing_score"),
            "old_openai_score": data.get("openai_final_score"),
            "old_assessment": data.get("assessment", {})
        })

        if len(results) >= limit:
            break

    return results


def run_comparison(entries: list, dry_run: bool = False):
    """Run assessments with optimized prompt and compare."""
    results = []

    # Check if optimized prompt is loaded
    optimized = load_optimized_prompt()
    if optimized:
        print(f"Using optimized prompt with {len(optimized.get('demos', []))} demos")
    else:
        print("WARNING: Optimized prompt not found, using default prompt")

    for i, entry in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] Processing: {entry['name'][:40]}")
        print(f"  Existing score: {entry['existing_score']}")
        print(f"  Old OpenAI score: {entry['old_openai_score']:.1f}")

        if dry_run:
            # Simulate for testing
            new_score = entry['old_openai_score'] - 15  # Simulate lower scores
            result = {
                **entry,
                "new_openai_score": new_score,
                "new_assessment": {"simulated": True}
            }
        else:
            # Run actual assessment
            assessment = assess_pronunciation_openai(
                entry["audio_path"],
                language="en-US",
                use_optimized_prompt=True
            )

            if "error" in assessment:
                print(f"  ERROR: {assessment['error']}")
                result = {
                    **entry,
                    "error": assessment["error"]
                }
            else:
                new_score = assessment.get("final_score", 0)
                print(f"  New OpenAI score: {new_score:.1f}")
                print(f"  Change: {new_score - entry['old_openai_score']:+.1f}")

                result = {
                    **entry,
                    "new_openai_score": new_score,
                    "new_assessment": assessment
                }

        results.append(result)

    return results


def generate_report(results: list, cloud_summary: dict):
    """Generate comparison report."""
    successful = [r for r in results if "new_openai_score" in r]
    failed = [r for r in results if "error" in r]

    if not successful:
        print("No successful assessments to report!")
        return {}

    # Extract scores
    existing_scores = [r["existing_score"] for r in successful]
    old_openai_scores = [r["old_openai_score"] for r in successful]
    new_openai_scores = [r["new_openai_score"] for r in successful]

    # Calculate statistics
    def calc_stats(scores):
        return {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "min": min(scores),
            "max": max(scores),
            "stddev": statistics.stdev(scores) if len(scores) > 1 else 0
        }

    # Calculate correlation
    def calc_correlation(x, y):
        n = len(x)
        mean_x, mean_y = sum(x)/n, sum(y)/n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x)**2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y)**2 for yi in y) ** 0.5
        return num / (den_x * den_y) if den_x > 0 and den_y > 0 else 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(results),
        "successful": len(successful),
        "failed": len(failed),

        "existing_score_stats": calc_stats(existing_scores),
        "old_openai_stats": calc_stats(old_openai_scores),
        "new_openai_stats": calc_stats(new_openai_scores),

        "old_vs_existing_correlation": calc_correlation(existing_scores, old_openai_scores),
        "new_vs_existing_correlation": calc_correlation(existing_scores, new_openai_scores),

        "score_changes": {
            "mean_change": statistics.mean(n - o for n, o in zip(new_openai_scores, old_openai_scores)),
            "scores_decreased": sum(1 for n, o in zip(new_openai_scores, old_openai_scores) if n < o),
            "scores_increased": sum(1 for n, o in zip(new_openai_scores, old_openai_scores) if n > o),
            "scores_same": sum(1 for n, o in zip(new_openai_scores, old_openai_scores) if abs(n - o) < 1)
        },

        "cloud_comparison": {
            "cloud_openai_mean": cloud_summary.get("batch_report_summary", {}).get("openai_score_stats_successful", {}).get("mean"),
            "cloud_openai_stddev": cloud_summary.get("batch_report_summary", {}).get("openai_score_stats_successful", {}).get("stddev"),
            "cloud_correlation": cloud_summary.get("batch_report_summary", {}).get("successful_score_correlation")
        },

        "individual_results": [
            {
                "name": r["name"],
                "existing": r["existing_score"],
                "old_openai": r["old_openai_score"],
                "new_openai": r["new_openai_score"],
                "change": r["new_openai_score"] - r["old_openai_score"],
                "new_vs_existing_diff": r["new_openai_score"] - r["existing_score"]
            }
            for r in successful
        ]
    }

    return report


def print_report(report: dict):
    """Print formatted report."""
    print("\n" + "="*70)
    print("OPTIMIZED PROMPT TEST RESULTS")
    print("="*70)

    print(f"\nSamples tested: {report['successful']}/{report['total_tested']}")

    print("\n--- Score Statistics ---")
    print(f"{'Metric':<25} {'Old OpenAI':>12} {'New OpenAI':>12} {'Change':>10}")
    print("-" * 60)

    old = report["old_openai_stats"]
    new = report["new_openai_stats"]

    print(f"{'Mean':<25} {old['mean']:>12.1f} {new['mean']:>12.1f} {new['mean']-old['mean']:>+10.1f}")
    print(f"{'Median':<25} {old['median']:>12.1f} {new['median']:>12.1f} {new['median']-old['median']:>+10.1f}")
    print(f"{'Std Dev':<25} {old['stddev']:>12.1f} {new['stddev']:>12.1f} {new['stddev']-old['stddev']:>+10.1f}")
    print(f"{'Min':<25} {old['min']:>12.1f} {new['min']:>12.1f} {new['min']-old['min']:>+10.1f}")
    print(f"{'Max':<25} {old['max']:>12.1f} {new['max']:>12.1f} {new['max']-old['max']:>+10.1f}")

    print("\n--- Correlation with Existing Scores ---")
    print(f"Old OpenAI vs Existing: {report['old_vs_existing_correlation']:.3f}")
    print(f"New OpenAI vs Existing: {report['new_vs_existing_correlation']:.3f}")
    print(f"Change: {report['new_vs_existing_correlation'] - report['old_vs_existing_correlation']:+.3f}")

    print("\n--- Score Movement ---")
    changes = report["score_changes"]
    print(f"Mean change: {changes['mean_change']:+.1f}")
    print(f"Scores decreased: {changes['scores_decreased']}")
    print(f"Scores increased: {changes['scores_increased']}")
    print(f"Scores similar (±1): {changes['scores_same']}")

    print("\n--- Comparison with Cloud Results ---")
    cloud = report["cloud_comparison"]
    print(f"{'Metric':<25} {'Cloud':>12} {'New Local':>12}")
    print("-" * 50)
    print(f"{'OpenAI Mean':<25} {cloud['cloud_openai_mean']:>12.1f} {new['mean']:>12.1f}")
    print(f"{'OpenAI Std Dev':<25} {cloud['cloud_openai_stddev']:>12.1f} {new['stddev']:>12.1f}")
    print(f"{'Correlation':<25} {cloud['cloud_correlation']:>12.3f} {report['new_vs_existing_correlation']:>12.3f}")

    print("\n--- Individual Results ---")
    print(f"{'Name':<30} {'Existing':>8} {'Old':>8} {'New':>8} {'Change':>8}")
    print("-" * 70)
    for r in report["individual_results"]:
        print(f"{r['name'][:30]:<30} {r['existing']:>8.0f} {r['old_openai']:>8.1f} {r['new_openai']:>8.1f} {r['change']:>+8.1f}")

    print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test optimized prompt")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without calling API")
    args = parser.parse_args()

    # Load cloud summary for comparison
    if CLOUD_SUMMARY.exists():
        with open(CLOUD_SUMMARY) as f:
            cloud_summary = json.load(f)
        print(f"Loaded cloud summary from {CLOUD_SUMMARY}")
    else:
        cloud_summary = {}
        print("Cloud summary not found")

    # Load existing results
    print(f"\nLoading up to {args.limit} existing results with audio files...")
    entries = load_existing_results(limit=args.limit)
    print(f"Found {len(entries)} valid entries")

    if not entries:
        print("No valid entries found!")
        sys.exit(1)

    # Run comparison
    print("\nRunning assessments with optimized prompt...")
    results = run_comparison(entries, dry_run=args.dry_run)

    # Generate report
    report = generate_report(results, cloud_summary)

    # Print report
    print_report(report)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / "comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
