#!/usr/bin/env python3
"""
Analyze ALL Airtable fields to find potential group keys and analyze non-decade scores.

This script:
1. Pulls ALL fields from Airtable records (not just video URL and score)
2. Identifies potential group key candidates (speaker ID, email, session, etc.)
3. Analyzes the non-decade scores (65.8, 71.2, 96) for clustering patterns
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import VIDEO_URL_FIELD, EXISTING_SCORE_FIELD
from src.airtable.client import get_airtable_table


def fetch_all_fields():
    """Fetch all records with ALL available fields."""
    api, table = get_airtable_table()

    # Get records with scores - no field filter to get ALL fields
    formula = f"AND({{{VIDEO_URL_FIELD}}} != '', {{{EXISTING_SCORE_FIELD}}} != '')"

    print("Fetching ALL fields from Airtable...")
    all_records = table.all(formula=formula)

    print(f"Fetched {len(all_records)} records")

    return all_records


def analyze_field_structure(records):
    """Analyze all available fields and identify potential group keys."""

    # Collect all unique field names across all records
    all_fields = set()
    field_examples = defaultdict(list)
    field_non_null_counts = Counter()

    for record in records:
        fields = record.get("fields", {})
        for field_name, value in fields.items():
            all_fields.add(field_name)
            if value is not None and value != "":
                field_non_null_counts[field_name] += 1
                if len(field_examples[field_name]) < 5:
                    field_examples[field_name].append(str(value)[:100])

    print("\n" + "=" * 80)
    print("ALL AVAILABLE FIELDS")
    print("=" * 80)

    # Sort fields by non-null count
    sorted_fields = sorted(all_fields, key=lambda x: -field_non_null_counts.get(x, 0))

    print(f"\nTotal unique fields: {len(all_fields)}")
    print(f"Total records: {len(records)}")
    print("\n{:<40} {:>10} {:>8}  Examples".format("Field Name", "Non-Null", "Fill %"))
    print("-" * 100)

    potential_group_keys = []

    for field in sorted_fields:
        count = field_non_null_counts.get(field, 0)
        pct = 100 * count / len(records)
        examples = field_examples.get(field, [])
        examples_str = " | ".join(examples[:3])[:50]

        print(f"{field:<40} {count:>10} {pct:>7.1f}%  {examples_str}")

        # Check if this could be a group key
        field_lower = field.lower()
        is_potential_key = any(x in field_lower for x in [
            'speaker', 'user', 'email', 'name', 'session', 'batch',
            'uploader', 'applicant', 'candidate', 'person', 'id',
            'submitter', 'creator', 'author', 'account'
        ])
        if is_potential_key and count > len(records) * 0.5:  # At least 50% filled
            potential_group_keys.append((field, count, pct))

    print("\n" + "=" * 80)
    print("POTENTIAL GROUP KEY CANDIDATES")
    print("=" * 80)

    if potential_group_keys:
        for field, count, pct in potential_group_keys:
            # Count unique values
            unique_values = set()
            for record in records:
                val = record.get("fields", {}).get(field)
                if val:
                    unique_values.add(str(val))

            print(f"\n  {field}:")
            print(f"    Non-null: {count} ({pct:.1f}%)")
            print(f"    Unique values: {len(unique_values)}")
            print(f"    Ratio (records/unique): {len(records)/len(unique_values):.1f}")

            if len(unique_values) < 20:
                print(f"    Values: {list(unique_values)[:10]}")
    else:
        print("\n  NO OBVIOUS GROUP KEY CANDIDATES FOUND")
        print("  Will need to fall back to speaker clustering")

    return all_fields, potential_group_keys


def analyze_non_decade_scores(records):
    """Analyze the non-decade scores (65.8, 71.2, 96, etc.)."""

    # Standard decade scores
    decade_scores = {0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

    non_decade_records = []
    decade_records = []

    for record in records:
        fields = record.get("fields", {})
        score_raw = fields.get(EXISTING_SCORE_FIELD)
        try:
            score = float(score_raw)
        except (ValueError, TypeError):
            continue

        record_data = {
            "record_id": record["id"],
            "score": score,
            "created_time": record.get("createdTime", ""),
            "fields": fields
        }

        if score not in decade_scores:
            non_decade_records.append(record_data)
        else:
            decade_records.append(record_data)

    print("\n" + "=" * 80)
    print("NON-DECADE SCORE ANALYSIS")
    print("=" * 80)

    total = len(records)
    non_decade_count = len(non_decade_records)
    decade_count = len(decade_records)

    print(f"\nTotal records with valid scores: {decade_count + non_decade_count}")
    print(f"Decade scores (standard): {decade_count} ({100*decade_count/total:.1f}%)")
    print(f"Non-decade scores (unusual): {non_decade_count} ({100*non_decade_count/total:.2f}%)")

    if non_decade_records:
        # Group by score value
        by_score = defaultdict(list)
        for r in non_decade_records:
            by_score[r["score"]].append(r)

        print(f"\nBreakdown of non-decade scores:")
        print("-" * 60)
        for score in sorted(by_score.keys()):
            recs = by_score[score]
            print(f"\n  Score {score}: {len(recs)} records")

            # Check proximity to decades
            nearest_decade = round(score / 10) * 10
            distance = abs(score - nearest_decade)
            print(f"    Nearest decade: {nearest_decade} (distance: {distance:.1f})")

            # Check time clustering
            times = [r["created_time"] for r in recs if r["created_time"]]
            if times:
                times_sorted = sorted(times)
                print(f"    Time range: {times_sorted[0][:10]} to {times_sorted[-1][:10]}")

            # Show sample names
            names = [r["fields"].get("Name", r["fields"].get("Name and Date", "?"))[:30] for r in recs[:3]]
            print(f"    Sample names: {names}")

        # Check if non-decade scores are clustered in time
        print("\n" + "-" * 60)
        print("TIME CLUSTERING ANALYSIS")

        if non_decade_records:
            # Extract dates
            non_decade_dates = []
            for r in non_decade_records:
                ct = r["created_time"]
                if ct:
                    date = ct[:10]  # YYYY-MM-DD
                    non_decade_dates.append(date)

            date_counts = Counter(non_decade_dates)
            print(f"\nNon-decade scores by date:")
            for date, count in sorted(date_counts.items()):
                print(f"  {date}: {count}")

            # Check if they're from a specific batch
            if len(date_counts) <= 3:
                print("\n  FINDING: Non-decade scores appear clustered in specific dates")
                print("  This suggests they're from a different scoring system/batch")
            else:
                print("\n  Non-decade scores are spread across multiple dates")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    pct_non_decade = 100 * non_decade_count / total
    if pct_non_decade < 2:
        print(f"\nNon-decade scores are {pct_non_decade:.2f}% of data (< 2%)")
        print("RECOMMENDATION: Exclude from Stage 2 baseline, keep in held-out eval")
    else:
        print(f"\nNon-decade scores are {pct_non_decade:.2f}% of data (>= 2%)")
        print("RECOMMENDATION: Include in training, but tag for analysis")

    return non_decade_records, decade_records


def check_name_as_group_key(records):
    """Analyze if 'Name' field can serve as a speaker grouping key."""

    print("\n" + "=" * 80)
    print("NAME FIELD AS GROUP KEY ANALYSIS")
    print("=" * 80)

    names = []
    for record in records:
        fields = record.get("fields", {})
        name = fields.get("Name") or fields.get("Name and Date") or ""
        names.append(name.strip())

    # Count name occurrences
    name_counts = Counter(names)
    unique_names = len(name_counts)
    total_records = len(records)

    print(f"\nTotal records: {total_records}")
    print(f"Unique names: {unique_names}")
    print(f"Average records per name: {total_records / unique_names:.2f}")

    # Distribution of records per name
    count_dist = Counter(name_counts.values())
    print(f"\nRecords-per-name distribution:")
    for n_records, n_names in sorted(count_dist.items()):
        print(f"  {n_records} record(s): {n_names} unique names")

    # Names with most records (potential repeat speakers)
    print(f"\nTop 20 most common names:")
    for name, count in name_counts.most_common(20):
        name_display = name[:40] if name else "(empty)"
        print(f"  {count:4d}x  {name_display}")

    # Check for empty/generic names
    empty_names = sum(1 for n in names if not n or n.lower() in ['unknown', 'test', 'n/a', ''])
    print(f"\nEmpty/generic names: {empty_names} ({100*empty_names/total_records:.1f}%)")

    # Recommendation
    if unique_names < total_records * 0.3:
        print("\n  FINDING: Many repeat names - Name field may be usable as group key")
        print(f"  But verify these are actually same speakers, not just same common name")
    elif empty_names > total_records * 0.1:
        print("\n  FINDING: Too many empty names - cannot use as reliable group key")
    else:
        print("\n  FINDING: Most names are unique - limited grouping possible")
        print("  May need speaker clustering for robust GroupKFold")

    return name_counts


def save_full_dataset(records, output_path):
    """Save complete dataset with all fields for further analysis."""

    # Flatten all fields
    all_field_names = set()
    for record in records:
        all_field_names.update(record.get("fields", {}).keys())

    # Standard fields first, then alphabetical
    priority_fields = ["record_id", "created_time", "Name", "Name and Date",
                       VIDEO_URL_FIELD, EXISTING_SCORE_FIELD]
    other_fields = sorted(all_field_names - set(priority_fields))
    fieldnames = ["record_id", "created_time"] + [f for f in priority_fields[2:] if f in all_field_names] + other_fields

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for record in records:
            row = {"record_id": record["id"], "created_time": record.get("createdTime", "")}
            row.update(record.get("fields", {}))
            writer.writerow(row)

    print(f"\nSaved full dataset to: {output_path}")


def main():
    print("=" * 80)
    print("AIRTABLE FIELD ANALYSIS FOR GROUP KEY DISCOVERY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Fetch all records with all fields
    records = fetch_all_fields()

    if not records:
        print("ERROR: No records found!")
        return

    # Analyze field structure
    all_fields, potential_keys = analyze_field_structure(records)

    # Analyze name field specifically
    name_counts = check_name_as_group_key(records)

    # Analyze non-decade scores
    non_decade, decade = analyze_non_decade_scores(records)

    # Save full dataset
    output_dir = Path(__file__).parent / "data"
    save_full_dataset(records, output_dir / "full_dataset.csv")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n1. Total records: {len(records)}")
    print(f"2. Total fields available: {len(all_fields)}")
    print(f"3. Potential group key candidates: {len(potential_keys)}")
    print(f"4. Unique names: {len(name_counts)}")
    print(f"5. Non-decade scores: {len(non_decade)} ({100*len(non_decade)/len(records):.2f}%)")

    if potential_keys:
        print(f"\n   Best group key candidates: {[k[0] for k in potential_keys]}")
    else:
        print(f"\n   NO GROUP KEYS FOUND - will need speaker clustering")


if __name__ == "__main__":
    main()
