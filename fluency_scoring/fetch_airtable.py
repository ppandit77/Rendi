#!/usr/bin/env python3
"""
Fetch Airtable records and generate labels.csv for fluency scoring pipeline.

This script pulls all records from Airtable that have:
- A video URL in "Question 1 DO URL"
- A human-assigned score in "Video 1 score"

Outputs:
- labels.csv: Normalized dataset with record_id, video_url, score, name
- Console report: Dataset statistics and score distribution
"""

import csv
import hashlib
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import VIDEO_URL_FIELD, EXISTING_SCORE_FIELD
from src.airtable.client import get_airtable_table


def fetch_all_records_with_scores():
    """
    Fetch all records that have both a video URL and an existing score.

    Returns:
        List of dicts with: record_id, video_url, score, name, created_time
    """
    api, table = get_airtable_table()

    # Formula: has video URL AND has existing score
    formula = f"AND({{{VIDEO_URL_FIELD}}} != '', {{{EXISTING_SCORE_FIELD}}} != '')"

    print(f"Fetching records with formula: {formula}")
    print(f"Video URL field: {VIDEO_URL_FIELD}")
    print(f"Score field: {EXISTING_SCORE_FIELD}")

    all_records = table.all(formula=formula)

    results = []
    skipped_invalid_score = 0
    skipped_invalid_url = 0

    for record in all_records:
        fields = record.get("fields", {})
        record_id = record["id"]

        # Get video URL
        video_url = fields.get(VIDEO_URL_FIELD)
        if not video_url or not isinstance(video_url, str):
            skipped_invalid_url += 1
            continue

        # Get score - must be numeric
        score_raw = fields.get(EXISTING_SCORE_FIELD)
        try:
            score = float(score_raw)
        except (ValueError, TypeError):
            skipped_invalid_score += 1
            continue

        # Get name for reference
        name = fields.get("Name", "") or fields.get("Name and Date", "Unknown")

        # Get created time if available
        created_time = record.get("createdTime", "")

        # Generate URL hash for cache validation
        url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]

        # Get email for GroupKFold (speaker grouping)
        email = fields.get("Email", "")

        results.append({
            "record_id": record_id,
            "video_url": video_url,
            "score": score,
            "name": name,
            "email": email,
            "created_time": created_time,
            "url_hash": url_hash,
        })

    print(f"\nFetch summary:")
    print(f"  Total records fetched: {len(all_records)}")
    print(f"  Valid records: {len(results)}")
    print(f"  Skipped (invalid score): {skipped_invalid_score}")
    print(f"  Skipped (invalid URL): {skipped_invalid_url}")

    return results


def analyze_scores(records):
    """Analyze and report score distribution."""
    if not records:
        print("No records to analyze!")
        return

    scores = [r["score"] for r in records]

    # Basic statistics
    n = len(scores)
    mean_score = sum(scores) / n
    sorted_scores = sorted(scores)
    median_score = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
    min_score = min(scores)
    max_score = max(scores)

    # Standard deviation
    variance = sum((s - mean_score) ** 2 for s in scores) / n
    std_score = variance ** 0.5

    # Score distribution (histogram buckets)
    buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
               (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
    bucket_counts = {f"{lo}-{hi-1}": 0 for lo, hi in buckets}

    for s in scores:
        for lo, hi in buckets:
            if lo <= s < hi:
                bucket_counts[f"{lo}-{hi-1}"] = bucket_counts.get(f"{lo}-{hi-1}", 0) + 1
                break

    # Check for duplicates (same URL)
    url_counts = Counter(r["video_url"] for r in records)
    duplicate_urls = {url: count for url, count in url_counts.items() if count > 1}

    # Check for potential speaker IDs
    has_speaker_field = any("speaker" in r.get("name", "").lower() for r in records[:100])

    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nSample size: {n}")
    print(f"\nScore Statistics:")
    print(f"  Min:    {min_score:.1f}")
    print(f"  Max:    {max_score:.1f}")
    print(f"  Mean:   {mean_score:.2f}")
    print(f"  Median: {median_score:.1f}")
    print(f"  Std:    {std_score:.2f}")
    print(f"  Range:  {max_score - min_score:.1f}")

    print(f"\nScore Distribution (Histogram):")
    print("-" * 40)
    max_bar = max(bucket_counts.values()) if bucket_counts else 1
    for bucket, count in bucket_counts.items():
        bar = "#" * int(40 * count / max_bar) if max_bar > 0 else ""
        pct = 100 * count / n
        print(f"  {bucket:>6}: {count:4d} ({pct:5.1f}%) {bar}")

    # Skewness check
    skewness = sum((s - mean_score) ** 3 for s in scores) / (n * std_score ** 3) if std_score > 0 else 0
    print(f"\nSkewness: {skewness:.3f}", end="")
    if abs(skewness) < 0.5:
        print(" (approximately symmetric)")
    elif skewness > 0:
        print(" (RIGHT-SKEWED - more low scores)")
    else:
        print(" (LEFT-SKEWED - more high scores)")

    print(f"\nDuplicate URLs: {len(duplicate_urls)}")
    if duplicate_urls:
        print("  WARNING: Some URLs appear multiple times:")
        for url, count in list(duplicate_urls.items())[:5]:
            print(f"    {url[:60]}... ({count} times)")
        if len(duplicate_urls) > 5:
            print(f"    ... and {len(duplicate_urls) - 5} more")

    print(f"\nSpeaker/Group ID: ", end="")
    # Check field names for potential grouping fields
    if records:
        sample_fields = records[0].keys()
        potential_group_fields = [f for f in sample_fields if any(x in f.lower() for x in ['speaker', 'user', 'session', 'batch', 'uploader'])]
        if potential_group_fields:
            print(f"Potential fields found: {potential_group_fields}")
        else:
            print("NOT FOUND - Risk of speaker leakage in CV splits!")

    # Unique score values
    unique_scores = sorted(set(scores))
    print(f"\nUnique score values: {len(unique_scores)}")
    if len(unique_scores) <= 20:
        print(f"  Values: {unique_scores}")
    else:
        print(f"  First 10: {unique_scores[:10]}")
        print(f"  Last 10: {unique_scores[-10:]}")

    # Check if ordinal (discrete clusters)
    if len(unique_scores) <= 10:
        print("\n  NOTE: Score appears ORDINAL (few discrete values).")
        print("  Consider ordinal regression or treating as classification.")

    return {
        "n": n,
        "min": min_score,
        "max": max_score,
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "skewness": skewness,
        "unique_values": len(unique_scores),
        "duplicates": len(duplicate_urls),
    }


def save_labels_csv(records, output_path):
    """Save normalized labels.csv."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['record_id', 'video_url', 'url_hash', 'score', 'name', 'email', 'created_time'])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved labels.csv to: {output_path}")
    print(f"  Total records: {len(records)}")


def main():
    print("=" * 70)
    print("AIRTABLE DATA FETCH FOR FLUENCY SCORING PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Fetch records
    records = fetch_all_records_with_scores()

    if not records:
        print("\nERROR: No valid records found!")
        return

    # Analyze scores
    stats = analyze_scores(records)

    # Save labels.csv
    output_dir = Path(__file__).parent / "data"
    save_labels_csv(records, output_dir / "labels.csv")

    # Training strategy recommendation based on sample count
    n = stats["n"]
    print("\n" + "=" * 70)
    print("TRAINING STRATEGY RECOMMENDATION")
    print("=" * 70)

    if n < 500:
        print(f"\nSample count: {n} (< 500)")
        print("STRATEGY: Frozen encoder + classical regressor (XGBoost, Ridge)")
        print("  - No fine-tuning of speech encoder")
        print("  - Extract embeddings once, train lightweight models")
    elif n < 2000:
        print(f"\nSample count: {n} (500-2000)")
        print("STRATEGY: Frozen encoder + small MLP head")
        print("  - Speech encoder stays frozen")
        print("  - Train 2-layer MLP on cached embeddings")
    else:
        print(f"\nSample count: {n} (> 2000)")
        print("STRATEGY: Partial encoder fine-tuning is possible")
        print("  - Can unfreeze top layers of speech encoder")
        print("  - Start with frozen baseline first")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the score distribution above")
    print("2. Run download_audio.py to fetch and convert videos")
    print("3. Check download_manifest.csv for success rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
