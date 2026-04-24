#!/usr/bin/env python3
"""
Compare V2 Prompt Service vs Existing "Video 1 Score" Service.

This script compares the new V2 anti-compression prompt against the existing
"Video 1 score" field in Airtable (the previous/old service scores).

Usage:
    python compare_services.py [--samples N]
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from statistics import mean, stdev
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from src.config import EXISTING_SCORE_FIELD, VIDEO_URL_FIELD
from src.assessment.openai_assessment import assess_pronunciation_openai
from src.utils.audio_converter import convert_video_to_audio_rendi_with_details
from src.airtable.client import get_airtable_table


def pearson_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_records_with_existing_scores(table, batch_size: int = 50) -> list:
    """
    Fetch records that have both a video URL and an existing score.

    Args:
        table: Airtable table object
        batch_size: Number of records to fetch

    Returns:
        List of record dicts with name, video_url, existing_score
    """
    results = []

    # Filter for records with existing Video 1 scores
    # Created from Feb 1, 2026 onwards
    date_filter = "IS_AFTER({Created}, '2026-01-31')"
    formula = f"AND({{{VIDEO_URL_FIELD}}} != '', {{{EXISTING_SCORE_FIELD}}} != '', {date_filter})"

    # Fetch all matching records
    all_records = table.all(formula=formula)

    for record in all_records:
        fields = record.get("fields", {})

        name = fields.get("Name", "") or fields.get("Name and Date", "Unknown")
        video_url = fields.get(VIDEO_URL_FIELD)
        existing_score = fields.get(EXISTING_SCORE_FIELD)

        if video_url and existing_score is not None:
            try:
                score_val = float(existing_score)
                results.append({
                    "id": record["id"],
                    "name": name,
                    "video_url": video_url,
                    "existing_score": score_val,
                })
            except (ValueError, TypeError):
                continue

        if len(results) >= batch_size:
            break

    return results


def compare_services(samples: int = 50, prompt_version: str = "v2"):
    """Compare prompt against existing Video 1 score field."""

    print("=" * 70)
    print(f"SERVICE COMPARISON: {prompt_version.upper()} Prompt vs Existing Video 1 Score")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target samples: {samples}")
    print(f"Prompt version: {prompt_version}")
    print("=" * 70)

    # Get records with existing scores
    api, table = get_airtable_table()
    records = get_records_with_existing_scores(table, batch_size=samples)

    if not records:
        print("No records found with existing scores!")
        return

    print(f"\nFound {len(records)} records with existing Video 1 scores")

    results = []
    failures = []

    for i, record in enumerate(records):
        print(f"\n--- Record {i+1}/{len(records)}: {record['name'][:40]} ---")
        print(f"  Existing (Video 1) score: {record['existing_score']}")

        # Convert video to audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            conversion = convert_video_to_audio_rendi_with_details(record['video_url'], audio_path)
            if not conversion["ok"]:
                print(f"  SKIP: Conversion failed - {conversion.get('error', 'unknown')}")
                failures.append({
                    "name": record["name"],
                    "reason": f"Conversion failed: {conversion.get('error', 'unknown')}"
                })
                continue

            # Run prompt assessment
            print(f"  Running {prompt_version} assessment...")
            v2_result = assess_pronunciation_openai(audio_path, prompt_version=prompt_version)

            v2_score = v2_result.get("final_score") or v2_result.get("score")
            v2_ratings = v2_result.get("dimension_ratings", {})
            v2_reasoning = v2_result.get("reasoning", "")

            if v2_score:
                result = {
                    "name": record["name"],
                    "existing_score": record["existing_score"],
                    "v2_score": v2_score,
                    "difference": v2_score - record["existing_score"],
                    "v2_ratings": v2_ratings,
                    "v2_reasoning": v2_reasoning,
                    "v2_confidence": v2_result.get("confidence", ""),
                }
                results.append(result)

                diff_str = f"+{result['difference']:.1f}" if result['difference'] >= 0 else f"{result['difference']:.1f}"
                print(f"  V2 Score: {v2_score:.1f} | Difference: {diff_str}")
                print(f"  V2 Ratings: {v2_ratings}")
            else:
                print(f"  ERROR: {v2_result.get('error', 'unknown')}")
                failures.append({
                    "name": record["name"],
                    "reason": f"V2 assessment failed: {v2_result.get('error', 'unknown')}"
                })

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Generate analysis and report
    if results:
        generate_report(results, failures, samples, prompt_version)
    else:
        print("\nNo successful comparisons to report!")


def generate_report(results: list, failures: list, target_samples: int, prompt_version: str = "v2"):
    """Generate comprehensive comparison report."""

    existing_scores = [r["existing_score"] for r in results]
    v2_scores = [r["v2_score"] for r in results]
    differences = [r["difference"] for r in results]

    # Calculate statistics
    existing_mean = mean(existing_scores)
    existing_std = stdev(existing_scores) if len(existing_scores) > 1 else 0
    existing_min = min(existing_scores)
    existing_max = max(existing_scores)

    v2_mean = mean(v2_scores)
    v2_std = stdev(v2_scores) if len(v2_scores) > 1 else 0
    v2_min = min(v2_scores)
    v2_max = max(v2_scores)

    correlation = pearson_correlation(existing_scores, v2_scores)

    # Agreement rate (within ±10 points)
    agreement_count = sum(1 for d in differences if abs(d) <= 10)
    agreement_rate = agreement_count / len(results) * 100

    # Score distribution buckets
    buckets = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 101)]
    bucket_labels = ["0-29", "30-49", "50-69", "70-84", "85-100"]

    existing_dist = []
    v2_dist = []
    for low, high in buckets:
        existing_dist.append(sum(1 for s in existing_scores if low <= s < high))
        v2_dist.append(sum(1 for s in v2_scores if low <= s < high))

    # Notable discrepancies (>20 point difference)
    discrepancies = [r for r in results if abs(r["difference"]) > 20]

    # Build report
    version_label = prompt_version.upper()
    report_lines = [
        "=" * 80,
        "SERVICE COMPARISON REPORT",
        f"{version_label} Prompt vs Existing Video 1 Score (Old Service)",
        "=" * 80,
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        f"Target Samples: {target_samples}",
        f"Successful Comparisons: {len(results)}",
        f"Failures: {len(failures)}",
        "",
        "=" * 80,
        "EXECUTIVE SUMMARY",
        "=" * 80,
        "",
        f"Correlation coefficient (Pearson r): {correlation:.3f}",
        f"Agreement rate (within ±10 points): {agreement_rate:.1f}%",
        f"Mean difference (V2 - Old): {mean(differences):+.1f}",
        "",
    ]

    # Interpretation
    if correlation > 0.7:
        interp = "Strong positive correlation - V2 scores track well with old service."
    elif correlation > 0.4:
        interp = "Moderate correlation - V2 and old service show some agreement."
    elif correlation > 0:
        interp = "Weak correlation - V2 and old service scores differ significantly."
    else:
        interp = "No correlation or negative - Services measure differently."

    report_lines.extend([
        f"Interpretation: {interp}",
        "",
        "=" * 80,
        "SCORE STATISTICS",
        "=" * 80,
        "",
        f"{'Metric':<20} {'Old Service':>15} {'V2 Prompt':>15} {'Difference':>15}",
        "-" * 65,
        f"{'Mean':<20} {existing_mean:>15.1f} {v2_mean:>15.1f} {v2_mean - existing_mean:>+15.1f}",
        f"{'Std Dev':<20} {existing_std:>15.1f} {v2_std:>15.1f} {v2_std - existing_std:>+15.1f}",
        f"{'Min':<20} {existing_min:>15.1f} {v2_min:>15.1f} {v2_min - existing_min:>+15.1f}",
        f"{'Max':<20} {existing_max:>15.1f} {v2_max:>15.1f} {v2_max - existing_max:>+15.1f}",
        f"{'Range':<20} {existing_max - existing_min:>15.1f} {v2_max - v2_min:>15.1f} {(v2_max - v2_min) - (existing_max - existing_min):>+15.1f}",
        "",
        "=" * 80,
        "SCORE DISTRIBUTION",
        "=" * 80,
        "",
        f"{'Range':<12} {'Old Service':>15} {'V2 Prompt':>15}",
        "-" * 45,
    ])

    for label, old_count, v2_count in zip(bucket_labels, existing_dist, v2_dist):
        report_lines.append(f"{label:<12} {old_count:>15} {v2_count:>15}")

    report_lines.extend([
        "",
        "=" * 80,
        "INDIVIDUAL RESULTS",
        "=" * 80,
        "",
        f"{'Name':<35} {'Old':>8} {'V2':>8} {'Diff':>8} {'V2 Ratings'}",
        "-" * 90,
    ])

    for r in sorted(results, key=lambda x: x["difference"]):
        ratings_str = "/".join([
            r["v2_ratings"].get("phoneme_accuracy", "?")[0],
            r["v2_ratings"].get("rhythm_and_stress", "?")[0],
            r["v2_ratings"].get("fluency", "?")[0],
            r["v2_ratings"].get("intelligibility", "?")[0],
        ]) if r["v2_ratings"] else "N/A"

        diff_str = f"+{r['difference']:.0f}" if r['difference'] >= 0 else f"{r['difference']:.0f}"
        report_lines.append(f"{r['name'][:35]:<35} {r['existing_score']:>8.0f} {r['v2_score']:>8.0f} {diff_str:>8} {ratings_str}")

    report_lines.extend([
        "",
        "=" * 80,
        f"NOTABLE DISCREPANCIES (>{20} point difference): {len(discrepancies)} cases",
        "=" * 80,
        "",
    ])

    if discrepancies:
        for r in sorted(discrepancies, key=lambda x: abs(x["difference"]), reverse=True):
            direction = "HIGHER" if r["difference"] > 0 else "LOWER"
            report_lines.extend([
                f"Name: {r['name']}",
                f"  Old Service Score: {r['existing_score']:.0f}",
                f"  V2 Score: {r['v2_score']:.0f} ({direction} by {abs(r['difference']):.0f} points)",
                f"  V2 Ratings: {r['v2_ratings']}",
                f"  V2 Reasoning: {r['v2_reasoning'][:200]}..." if r['v2_reasoning'] else "",
                "",
            ])
    else:
        report_lines.append("No discrepancies >20 points found.")

    if failures:
        report_lines.extend([
            "",
            "=" * 80,
            f"FAILURES ({len(failures)} records)",
            "=" * 80,
            "",
        ])
        for i, f in enumerate(failures, 1):
            report_lines.append(f"{i}. {f['name']}: {f['reason']}")

    report_lines.extend([
        "",
        "=" * 80,
        "CONCLUSION",
        "=" * 80,
        "",
        f"The {version_label} prompt shows {correlation:.2f} correlation with the old service.",
        "",
        f"Score range comparison:",
        f"  - Old service range: {existing_min:.0f} to {existing_max:.0f} ({existing_max - existing_min:.0f} points)",
        f"  - {version_label} prompt range: {v2_min:.0f} to {v2_max:.0f} ({v2_max - v2_min:.0f} points)",
        "",
        f"Standard deviation comparison:",
        f"  - Old service std dev: {existing_std:.1f}",
        f"  - {version_label} prompt std dev: {v2_std:.1f}",
        "",
    ])

    if v2_std > existing_std:
        report_lines.append(f"{version_label} produces more differentiated scores (higher variance).")
    else:
        report_lines.append(f"{version_label} produces similar or less differentiated scores compared to old service.")

    report_lines.extend([
        "",
        "=" * 80,
        f"Report generated: {datetime.now().isoformat()}",
        "=" * 80,
    ])

    # Write report to file
    report_text = "\n".join(report_lines)

    output_file = f"data/reports/service_comparison_{prompt_version}_report.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report_text)

    # Also save JSON for further analysis
    json_file = f"data/reports/service_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target_samples": target_samples,
            "successful": len(results),
            "failures": len(failures),
            "correlation": correlation,
            "agreement_rate": agreement_rate,
            "old_service_stats": {
                "mean": existing_mean,
                "std": existing_std,
                "min": existing_min,
                "max": existing_max,
            },
            "v2_stats": {
                "mean": v2_mean,
                "std": v2_std,
                "min": v2_min,
                "max": v2_max,
            },
            "results": results,
            "failures": failures,
        }, f, indent=2)

    # Print report
    print("\n" + report_text)
    print(f"\nReport saved to: {output_file}")
    print(f"JSON data saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare prompt vs existing Video 1 score')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to compare')
    parser.add_argument('--prompt', type=str, default='v2', choices=['v2', 'v3', 'v4', 'dspy', 'basic'],
                        help='Prompt version to use (default: v2)')
    args = parser.parse_args()

    compare_services(args.samples, args.prompt)


if __name__ == "__main__":
    main()
