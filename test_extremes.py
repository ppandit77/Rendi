#!/usr/bin/env python3
"""
Test v2 prompt on EXTREME cases only - those with existing scores near 0 or 90+.
This helps verify the prompt can produce scores outside the 50-70 range.
"""

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.airtable.client import get_airtable_table
from src.airtable.records import get_records_needing_assessment
from src.assessment.openai_assessment import assess_pronunciation_openai
from src.utils.audio_converter import convert_video_to_audio_rendi_with_details


def test_extremes():
    print("=" * 70)
    print("EXTREME CASE TEST - V2 Prompt on Low and High Scorers")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    api, table = get_airtable_table()
    records = get_records_needing_assessment(table, batch_size=500, field_exists=True, reprocess=True)

    # Filter to extremes
    low_scorers = [r for r in records if (r['existing_score'] or 0) <= 10][:3]
    high_scorers = [r for r in records if (r['existing_score'] or 0) >= 80][:3]

    test_records = low_scorers + high_scorers
    print(f"\nTesting {len(low_scorers)} low scorers and {len(high_scorers)} high scorers\n")

    for record in test_records:
        print(f"--- {record['name'][:40]} (Existing: {record['existing_score']}) ---")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            conversion = convert_video_to_audio_rendi_with_details(record['video_url'], audio_path)
            if not conversion["ok"]:
                print(f"  SKIP: Conversion failed")
                continue

            result = assess_pronunciation_openai(audio_path, prompt_version="v2")

            score = result.get("final_score") or result.get("score")
            ratings = result.get("dimension_ratings", {})
            reasoning = result.get("reasoning", "")

            if score:
                print(f"  V2 Score: {score:.1f}")
                print(f"  Ratings: {ratings}")
                print(f"  Reasoning: {reasoning[:150]}...")
                print(f"  Expected: {'LOW (<50)' if record['existing_score'] <= 10 else 'HIGH (>75)'}")
                print()
            else:
                print(f"  ERROR: {result.get('error', 'unknown')}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    test_extremes()
