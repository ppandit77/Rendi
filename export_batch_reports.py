#!/usr/bin/env python3
"""
Download Airtable batch results and save local analysis reports.

Outputs are written to a timestamped folder under `data/reports/` by default.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.airtable.client import get_airtable_table
from src.config import DATA_DIR, EXISTING_SCORE_FIELD, NEW_SCORE_FIELD, VIDEO_URL_FIELD
from src.utils.logging_utils import setup_logging


REPORTS_DIR = os.path.join(DATA_DIR, "reports")
DEFAULT_BATCH_REPORT = os.path.join(DATA_DIR, "batch_assessment", "comparison_report.json")
DEFAULT_START_DATE = "2026-02-01"
DEFAULT_FIELDS = [
    "Record",
    "Name",
    "Name and Date",
    "Created",
    "created date",
    "Source",
    VIDEO_URL_FIELD,
    EXISTING_SCORE_FIELD,
    NEW_SCORE_FIELD,
]

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Airtable batch results and local statistics reports")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Only include Airtable records created on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to data/reports/batch_results_<timestamp>",
    )
    parser.add_argument(
        "--batch-report",
        default=DEFAULT_BATCH_REPORT,
        help="Optional path to a local batch comparison report JSON",
    )
    return parser.parse_args()


def parse_float(value):
    if value in ("", None):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_datetime(value: str | None):
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    for candidate in (
        text.replace("Z", "+00:00"),
        text,
    ):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass

    for fmt in ("%m/%d/%Y %I:%M%p", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    return None


def make_output_dir(requested_dir: str | None) -> str:
    if requested_dir:
        output_dir = requested_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(REPORTS_DIR, f"batch_results_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def score_bucket(value):
    if value is None:
        return "missing"
    if value < 50:
        return "0-49"
    if value < 60:
        return "50-59"
    if value < 70:
        return "60-69"
    if value < 80:
        return "70-79"
    if value < 90:
        return "80-89"
    return "90-100"


def safe_mean(values):
    return round(sum(values) / len(values), 4) if values else None


def safe_median(values):
    return round(statistics.median(values), 4) if values else None


def safe_pstdev(values):
    return round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0 if values else None


def score_stats(values):
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "mean": safe_mean(numeric),
        "median": safe_median(numeric),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
        "stddev": safe_pstdev(numeric),
    }


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))

    if not x_denom or not y_denom:
        return None

    return round(numerator / (x_denom * y_denom), 4)


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_airtable_records(start_date: str) -> list[dict]:
    logger.info("Fetching Airtable records created on or after %s", start_date)
    _, table = get_airtable_table()
    formula = f"AND({{{VIDEO_URL_FIELD}}} != '', IS_AFTER({{Created}}, '{start_date}') )"
    records = table.all(formula=formula)
    logger.info("Fetched %s Airtable records", len(records))
    return records


def normalize_airtable_record(record: dict) -> dict:
    fields = record.get("fields", {})
    created_value = fields.get("Created") or fields.get("created date")
    created_at = parse_datetime(created_value)
    record_label = fields.get("Record", "")
    source = fields.get("Source", "")
    video_url = fields.get(VIDEO_URL_FIELD, "")
    name = fields.get("Name") or fields.get("Name and Date") or record_label

    existing_score = parse_float(fields.get(EXISTING_SCORE_FIELD))
    pronunciation_score = parse_float(fields.get(NEW_SCORE_FIELD))
    is_xobin = "xobin" in " ".join(
        [
            str(video_url).lower(),
            str(source).lower(),
            str(record_label).lower(),
        ]
    )

    return {
        "record_id": record.get("id"),
        "record": record_label,
        "name": name,
        "created": created_at.isoformat() if created_at else created_value,
        "created_date": created_at.date().isoformat() if created_at else "",
        "source": source,
        "video_url": video_url,
        "existing_score": existing_score,
        "pronunciation_assessment_score": pronunciation_score,
        "has_existing_score": existing_score is not None,
        "has_pronunciation_assessment_score": pronunciation_score is not None,
        "status": "processed" if pronunciation_score is not None else "pending",
        "is_xobin": is_xobin,
    }


def summarize_airtable_records(records: list[dict]) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    processed = [record for record in records if record["has_pronunciation_assessment_score"]]
    pending = [record for record in records if not record["has_pronunciation_assessment_score"]]
    both_scores = [
        record for record in records
        if record["existing_score"] is not None and record["pronunciation_assessment_score"] is not None
    ]

    source_groups = defaultdict(list)
    date_groups = defaultdict(list)
    processed_distribution = Counter()
    existing_distribution = Counter()
    comparison_rows = []

    for record in records:
        source_groups["xobin" if record["is_xobin"] else (record["source"] or "unknown")].append(record)
        if record["created_date"]:
            date_groups[record["created_date"]].append(record)
        processed_distribution[score_bucket(record["pronunciation_assessment_score"])] += 1
        existing_distribution[score_bucket(record["existing_score"])] += 1

    for record in both_scores:
        difference = round(record["pronunciation_assessment_score"] - record["existing_score"], 4)
        comparison_rows.append(
            {
                "record_id": record["record_id"],
                "name": record["name"],
                "created_date": record["created_date"],
                "existing_score": record["existing_score"],
                "pronunciation_assessment_score": record["pronunciation_assessment_score"],
                "difference": difference,
            }
        )

    comparison_rows.sort(key=lambda row: row["difference"])

    source_rows = []
    for source_name in sorted(source_groups):
        rows = source_groups[source_name]
        source_rows.append(
            {
                "source": source_name,
                "records": len(rows),
                "processed": sum(1 for row in rows if row["has_pronunciation_assessment_score"]),
                "pending": sum(1 for row in rows if not row["has_pronunciation_assessment_score"]),
            }
        )

    daily_rows = []
    for created_date in sorted(date_groups):
        rows = date_groups[created_date]
        daily_rows.append(
            {
                "created_date": created_date,
                "records": len(rows),
                "processed": sum(1 for row in rows if row["has_pronunciation_assessment_score"]),
                "pending": sum(1 for row in rows if not row["has_pronunciation_assessment_score"]),
            }
        )

    score_distribution_rows = []
    bucket_order = ["0-49", "50-59", "60-69", "70-79", "80-89", "90-100", "missing"]
    for bucket in bucket_order:
        score_distribution_rows.append(
            {
                "bucket": bucket,
                "existing_score_count": existing_distribution.get(bucket, 0),
                "pronunciation_assessment_score_count": processed_distribution.get(bucket, 0),
            }
        )

    pronunciation_values = [record["pronunciation_assessment_score"] for record in processed]
    existing_values = [record["existing_score"] for record in records if record["existing_score"] is not None]
    comparison_differences = [row["difference"] for row in comparison_rows]

    summary = {
        "total_records": len(records),
        "processed_records": len(processed),
        "pending_records": len(pending),
        "xobin_records": sum(1 for record in records if record["is_xobin"]),
        "non_xobin_records": sum(1 for record in records if not record["is_xobin"]),
        "existing_score_stats": score_stats(existing_values),
        "pronunciation_assessment_score_stats": score_stats(pronunciation_values),
        "comparison_stats": {
            "count_with_both_scores": len(both_scores),
            "difference_stats": score_stats(comparison_differences),
            "correlation": pearson_correlation(
                [record["existing_score"] for record in both_scores],
                [record["pronunciation_assessment_score"] for record in both_scores],
            ),
        },
    }

    return summary, source_rows, daily_rows, score_distribution_rows, comparison_rows


def load_batch_report(path: str):
    if not path or not os.path.exists(path):
        logger.info("Local batch comparison report not found at %s", path)
        return None

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_batch_report(report: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    results = report.get("results", [])
    successes = [item for item in results if "openai_final_score" in item]
    failures = [item for item in results if "openai_final_score" not in item]

    failure_reasons = Counter()
    failure_rows = []
    attempts_rows = []

    for item in results:
        assessment = item.get("assessment") if isinstance(item.get("assessment"), dict) else {}
        error = item.get("error") or assessment.get("error")
        error_stage = item.get("error_stage") or assessment.get("error_stage") or ""
        error_type = item.get("error_type") or assessment.get("error_type") or ""
        reason = error or "success"

        if reason != "success":
            failure_reasons[reason] += 1
            failure_rows.append(
                {
                    "index": item.get("index"),
                    "name": item.get("name"),
                    "error": error,
                    "error_stage": error_stage,
                    "error_type": error_type,
                    "url": item.get("url"),
                }
            )

        attempts_rows.append(
            {
                "index": item.get("index"),
                "name": item.get("name"),
                "existing_score": item.get("existing_score"),
                "openai_final_score": item.get("openai_final_score"),
                "score_difference": item.get("score_difference"),
                "status": "success" if "openai_final_score" in item else "failed",
                "error": error,
                "error_stage": error_stage,
                "error_type": error_type,
                "is_xobin": item.get("is_xobin"),
                "url": item.get("url"),
            }
        )

    successful_existing = [item["existing_score"] for item in successes if item.get("existing_score") is not None]
    successful_openai = [item["openai_final_score"] for item in successes if item.get("openai_final_score") is not None]
    successful_differences = [item["score_difference"] for item in successes if item.get("score_difference") is not None]

    failure_reason_rows = [
        {"reason": reason, "count": count}
        for reason, count in failure_reasons.most_common()
    ]

    summary = {
        "total_attempted": len(results),
        "successful": len(successes),
        "failed": len(failures),
        "xobin_attempts": sum(1 for item in results if item.get("is_xobin")),
        "existing_score_stats_successful": score_stats(successful_existing),
        "openai_score_stats_successful": score_stats(successful_openai),
        "difference_stats_successful": score_stats(successful_differences),
        "successful_score_correlation": pearson_correlation(successful_existing, successful_openai),
        "failure_reasons": dict(failure_reasons),
    }

    return summary, attempts_rows, failure_rows, failure_reason_rows


def build_markdown_summary(
    airtable_summary: dict,
    batch_summary: dict | None,
    airtable_files: dict,
    batch_files: dict,
    start_date: str,
) -> str:
    lines = [
        "# Batch Results Export",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Airtable start date filter: {start_date}",
        "",
        "## Airtable Snapshot",
        "",
        f"- Total records: {airtable_summary['total_records']}",
        f"- Processed records: {airtable_summary['processed_records']}",
        f"- Pending records: {airtable_summary['pending_records']}",
        f"- Xobin records: {airtable_summary['xobin_records']}",
        "",
        "### Score Stats",
        "",
        f"- Existing score stats: {json.dumps(airtable_summary['existing_score_stats'])}",
        f"- Pronunciation assessment score stats: {json.dumps(airtable_summary['pronunciation_assessment_score_stats'])}",
        f"- Comparison stats: {json.dumps(airtable_summary['comparison_stats'])}",
        "",
        "### Files",
        "",
        f"- Normalized records JSON: {airtable_files['records_json']}",
        f"- Normalized records CSV: {airtable_files['records_csv']}",
        f"- Source breakdown CSV: {airtable_files['sources_csv']}",
        f"- Daily counts CSV: {airtable_files['daily_csv']}",
        f"- Score distribution CSV: {airtable_files['distribution_csv']}",
        f"- Score comparison CSV: {airtable_files['comparison_csv']}",
    ]

    if batch_summary:
        lines.extend(
            [
                "",
                "## Local Batch Comparison Report",
                "",
                f"- Total attempted: {batch_summary['total_attempted']}",
                f"- Successful: {batch_summary['successful']}",
                f"- Failed: {batch_summary['failed']}",
                f"- Failure reasons: {json.dumps(batch_summary['failure_reasons'])}",
                f"- Successful score correlation: {batch_summary['successful_score_correlation']}",
                "",
                "### Files",
                "",
                f"- Batch summary JSON: {batch_files['summary_json']}",
                f"- Batch attempts CSV: {batch_files['attempts_csv']}",
                f"- Batch failures CSV: {batch_files['failures_csv']}",
                f"- Batch failure reasons CSV: {batch_files['failure_reasons_csv']}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)
    logger.info("Writing reports to %s", output_dir)

    airtable_records_raw = fetch_airtable_records(args.start_date)
    airtable_records = [normalize_airtable_record(record) for record in airtable_records_raw]
    airtable_summary, source_rows, daily_rows, distribution_rows, comparison_rows = summarize_airtable_records(airtable_records)

    airtable_files = {
        "records_json": os.path.join(output_dir, "airtable_records.json"),
        "raw_json": os.path.join(output_dir, "airtable_records_raw.json"),
        "records_csv": os.path.join(output_dir, "airtable_records.csv"),
        "sources_csv": os.path.join(output_dir, "airtable_source_breakdown.csv"),
        "daily_csv": os.path.join(output_dir, "airtable_daily_counts.csv"),
        "distribution_csv": os.path.join(output_dir, "airtable_score_distribution.csv"),
        "comparison_csv": os.path.join(output_dir, "airtable_score_comparison.csv"),
        "summary_json": os.path.join(output_dir, "airtable_summary.json"),
    }

    write_json(airtable_files["records_json"], airtable_records)
    write_json(airtable_files["raw_json"], airtable_records_raw)
    write_json(airtable_files["summary_json"], airtable_summary)
    write_csv(airtable_files["records_csv"], airtable_records, list(airtable_records[0].keys()) if airtable_records else [])
    write_csv(airtable_files["sources_csv"], source_rows, ["source", "records", "processed", "pending"])
    write_csv(airtable_files["daily_csv"], daily_rows, ["created_date", "records", "processed", "pending"])
    write_csv(
        airtable_files["distribution_csv"],
        distribution_rows,
        ["bucket", "existing_score_count", "pronunciation_assessment_score_count"],
    )
    write_csv(
        airtable_files["comparison_csv"],
        comparison_rows,
        ["record_id", "name", "created_date", "existing_score", "pronunciation_assessment_score", "difference"],
    )

    batch_report = load_batch_report(args.batch_report)
    batch_summary = None
    batch_files = {}
    if batch_report:
        batch_summary, attempts_rows, failure_rows, failure_reason_rows = summarize_batch_report(batch_report)
        batch_files = {
            "summary_json": os.path.join(output_dir, "batch_report_summary.json"),
            "attempts_csv": os.path.join(output_dir, "batch_attempts.csv"),
            "failures_csv": os.path.join(output_dir, "batch_failures.csv"),
            "failure_reasons_csv": os.path.join(output_dir, "batch_failure_reasons.csv"),
        }
        write_json(batch_files["summary_json"], batch_summary)
        write_csv(
            batch_files["attempts_csv"],
            attempts_rows,
            [
                "index",
                "name",
                "existing_score",
                "openai_final_score",
                "score_difference",
                "status",
                "error",
                "error_stage",
                "error_type",
                "is_xobin",
                "url",
            ],
        )
        write_csv(
            batch_files["failures_csv"],
            failure_rows,
            ["index", "name", "error", "error_stage", "error_type", "url"],
        )
        write_csv(
            batch_files["failure_reasons_csv"],
            failure_reason_rows,
            ["reason", "count"],
        )

    combined_summary = {
        "generated_at": datetime.now().isoformat(),
        "start_date": args.start_date,
        "output_dir": output_dir,
        "airtable_summary": airtable_summary,
        "batch_report_summary": batch_summary,
    }
    write_json(os.path.join(output_dir, "summary.json"), combined_summary)

    markdown_summary = build_markdown_summary(
        airtable_summary=airtable_summary,
        batch_summary=batch_summary,
        airtable_files=airtable_files,
        batch_files=batch_files,
        start_date=args.start_date,
    )
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown_summary)

    logger.info("Export complete")
    logger.info("Summary written to %s", os.path.join(output_dir, "summary.md"))


if __name__ == "__main__":
    main()
