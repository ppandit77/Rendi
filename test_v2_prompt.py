#!/usr/bin/env python3
"""
Test the v2 anti-compression prompt against a sample of records.

This script compares the new v2 prompt against the old DSPy prompt to verify
that score compression is reduced.

Usage:
    python test_v2_prompt.py [--samples N]
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.config import AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID, RENDI_API_KEY
from src.assessment.openai_assessment import assess_pronunciation_openai
from src.utils.audio_converter import convert_video_to_audio_rendi_with_details
from src.airtable.client import get_airtable_table
from src.airtable.records import get_records_needing_assessment


def test_prompt_versions(samples: int = 10):
    """Test both prompt versions on the same samples."""

    print("=" * 70)
    print("V2 PROMPT TEST - Anti-Compression vs Legacy DSPy")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Samples: {samples}")
    print("=" * 70)

    # Get records
    api, table = get_airtable_table()
    records = get_records_needing_assessment(table, batch_size=samples, field_exists=True, reprocess=True)

    if not records:
        print("No records found!")
        return

    print(f"\nFound {len(records)} records to test")

    results = []

    for i, record in enumerate(records[:samples]):
        print(f"\n--- Record {i+1}/{samples}: {record['name'][:40]} ---")
        print(f"  Existing score: {record['existing_score']}")

        # Convert video to audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            conversion = convert_video_to_audio_rendi_with_details(record['video_url'], audio_path)
            if not conversion["ok"]:
                print(f"  SKIP: Conversion failed - {conversion.get('error', 'unknown')}")
                continue

            # Test v2 prompt
            print("  Testing v2 prompt...")
            v2_result = assess_pronunciation_openai(audio_path, prompt_version="v2")
            v2_score = v2_result.get("final_score") or v2_result.get("score")
            v2_ratings = v2_result.get("dimension_ratings", {})

            # Test legacy DSPy prompt
            print("  Testing DSPy prompt...")
            dspy_result = assess_pronunciation_openai(audio_path, prompt_version="dspy")
            dspy_score = dspy_result.get("final_score")

            if v2_score and dspy_score:
                result = {
                    "name": record["name"],
                    "existing_score": record["existing_score"],
                    "v2_score": v2_score,
                    "dspy_score": dspy_score,
                    "v2_ratings": v2_ratings,
                    "v2_reasoning": v2_result.get("reasoning", ""),
                    "v2_confidence": v2_result.get("confidence", ""),
                }
                results.append(result)

                print(f"  V2 Score: {v2_score:.1f} | DSPy Score: {dspy_score:.1f} | Existing: {record['existing_score']}")
                print(f"  V2 Ratings: {v2_ratings}")
                if v2_result.get("reasoning"):
                    print(f"  V2 Reasoning: {v2_result['reasoning'][:100]}...")
            else:
                print(f"  ERROR: v2={v2_result.get('error', 'ok')} dspy={dspy_result.get('error', 'ok')}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Analysis
    if results:
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        v2_scores = [r["v2_score"] for r in results]
        dspy_scores = [r["dspy_score"] for r in results]
        existing_scores = [r["existing_score"] for r in results if r["existing_score"]]

        print(f"\nV2 Scores:")
        print(f"  Range: {min(v2_scores):.1f} - {max(v2_scores):.1f}")
        print(f"  Mean: {sum(v2_scores)/len(v2_scores):.1f}")
        print(f"  Unique values: {len(set(int(s) for s in v2_scores))}")

        print(f"\nDSPy Scores:")
        print(f"  Range: {min(dspy_scores):.1f} - {max(dspy_scores):.1f}")
        print(f"  Mean: {sum(dspy_scores)/len(dspy_scores):.1f}")
        print(f"  Unique values: {len(set(int(s) for s in dspy_scores))}")

        # Score distribution comparison
        print("\n--- Score Distribution ---")
        for bucket_start in range(0, 100, 10):
            bucket_end = bucket_start + 10
            v2_in_bucket = sum(1 for s in v2_scores if bucket_start <= s < bucket_end)
            dspy_in_bucket = sum(1 for s in dspy_scores if bucket_start <= s < bucket_end)
            print(f"  {bucket_start:2d}-{bucket_end:2d}: V2={'#'*v2_in_bucket:<10} DSPy={'#'*dspy_in_bucket}")

        # Individual results
        print("\n--- Individual Results ---")
        print(f"{'Name':<30} {'Exist':>6} {'V2':>6} {'DSPy':>6} {'V2 Ratings'}")
        print("-" * 80)
        for r in results:
            ratings_str = "/".join([
                r["v2_ratings"].get("phoneme_accuracy", "?")[0],
                r["v2_ratings"].get("rhythm_and_stress", "?")[0],
                r["v2_ratings"].get("fluency", "?")[0],
                r["v2_ratings"].get("intelligibility", "?")[0],
            ]) if r["v2_ratings"] else "N/A"

            print(f"{r['name'][:30]:<30} {r['existing_score'] or 0:>6.0f} {r['v2_score']:>6.1f} {r['dspy_score']:>6.1f} {ratings_str}")

        # Save results
        output_file = f"data/reports/v2_prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "samples": len(results),
                "v2_stats": {
                    "min": min(v2_scores),
                    "max": max(v2_scores),
                    "mean": sum(v2_scores)/len(v2_scores),
                    "unique_int_scores": len(set(int(s) for s in v2_scores)),
                },
                "dspy_stats": {
                    "min": min(dspy_scores),
                    "max": max(dspy_scores),
                    "mean": sum(dspy_scores)/len(dspy_scores),
                    "unique_int_scores": len(set(int(s) for s in dspy_scores)),
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    print(f"\nCompleted: {datetime.now().isoformat()}")


def main():
    parser = argparse.ArgumentParser(description='Test v2 anti-compression prompt')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to test')
    args = parser.parse_args()

    test_prompt_versions(args.samples)


if __name__ == "__main__":
    main()
