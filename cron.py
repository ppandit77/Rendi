#!/usr/bin/env python3
"""
Airtable Pronunciation Assessment Cron Job.

Automatically processes records from Airtable Test Results table:
1. Finds records with video URLs that need assessment (from Feb 1, 2026)
2. Converts videos to audio using Rendi API
3. Runs OpenAI GPT-4o pronunciation assessment
4. Updates Airtable with the score

Business Logic:
- If "Video 1 score" is empty -> update "Video 1 score"
- If "Video 1 score" has value -> update "Pronunciation Assessment Score"

Usage:
    python cron.py [--dry-run] [--batch-size N]

Examples:
    python cron.py                    # Process all new records
    python cron.py --dry-run          # Preview without updating
    python cron.py --batch-size 50    # Process 50 records
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import OPENAI_API_KEY, AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID, RENDI_API_KEY
from src.assessment.openai_assessment import assess_pronunciation_openai
from src.utils.audio_converter import convert_video_to_audio_rendi_with_details
from src.utils.logging_utils import log_error_result, setup_logging
from src.airtable.client import get_airtable_table, ensure_field_exists
from src.airtable.records import get_records_needing_assessment, update_airtable_score


logger = setup_logging(__name__)


def extract_error_result(result: dict, *, default_stage: str, **context) -> dict:
    """
    Normalize nested error payloads from shared modules into a flat structure for logging.
    """
    error_result = {
        "error": result.get("error", "Unknown error"),
        "error_stage": result.get("error_stage", default_stage),
        "error_type": result.get("error_type", "UnknownError"),
    }

    error_context = dict(result.get("error_context", {}))
    if "raw_response" in result:
        error_context["raw_response"] = result["raw_response"]
    error_context.update(context)

    if error_context:
        error_result["error_context"] = error_context

    return error_result


def process_pending_records(dry_run=False, batch_size=None):
    """
    Main processing function.

    Args:
        dry_run: If True, don't update Airtable
        batch_size: Maximum records to process (None for all)
    """
    # Verify configuration
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured")
        return

    if not AIRTABLE_API_KEY:
        logger.error("AIRTABLE_API_KEY not configured")
        return

    if not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
        logger.error("AIRTABLE_BASE_ID or AIRTABLE_TABLE_ID not configured")
        return

    if not RENDI_API_KEY:
        logger.error("RENDI_API_KEY not configured")
        return

    logger.info("=" * 60)
    logger.info("Airtable Pronunciation Assessment Cron")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Connect to Airtable
    api, table = get_airtable_table()

    # Ensure the Pronunciation Assessment Score field exists
    field_exists = ensure_field_exists(api)

    # Get records needing assessment
    records = get_records_needing_assessment(table, batch_size, field_exists)

    if not records:
        logger.info("No records need assessment. Exiting.")
        return

    # Process each record
    processed = 0
    successful = 0
    failed = 0
    failures = []

    for i, record in enumerate(records):
        logger.info("")
        logger.info(f"Processing {i+1}/{len(records)}: {record['name'][:40] if record['name'] else 'Unknown'}")
        logger.info(f"  Record ID: {record['record_id']}")
        logger.info(f"  Existing score: {record['existing_score']}")
        processed += 1

        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            # Step 1: Convert video to audio
            logger.info("  Converting video to audio...")
            conversion_result = convert_video_to_audio_rendi_with_details(record['video_url'], audio_path)
            if not conversion_result["ok"]:
                failure = {
                    **extract_error_result(
                        conversion_result,
                        default_stage="convert_video_to_audio",
                        record_id=record["record_id"],
                        name=record["name"],
                    )
                }
                failures.append(failure)
                log_error_result(logger, "Record failed", failure)
                failed += 1
                continue

            # Step 2: Run pronunciation assessment
            logger.info("  Running OpenAI assessment...")
            assessment = assess_pronunciation_openai(audio_path)

            if 'error' in assessment:
                failure = extract_error_result(
                    assessment,
                    default_stage="assess_pronunciation",
                    record_id=record["record_id"],
                    name=record["name"],
                )
                failures.append(failure)
                log_error_result(logger, "Record failed", failure)
                failed += 1
                continue

            # Get final score
            final_score = assessment.get('final_score')
            if final_score is None:
                failure = {
                    "error": "No final score in assessment",
                    "error_stage": "validate_assessment",
                    "error_type": "MissingFinalScore",
                    "error_context": {
                        "record_id": record["record_id"],
                        "name": record["name"],
                        "assessment_keys": list(assessment.keys()),
                    },
                }
                failures.append(failure)
                log_error_result(logger, "Record failed", failure)
                failed += 1
                continue

            # Log scores
            scores = assessment.get('scores', {})
            logger.info(f"  Scores: Acc={scores.get('accuracy')}, Flu={scores.get('fluency')}, "
                       f"Pro={scores.get('pronunciation')}, Prs={scores.get('prosody')}")
            logger.info(f"  Final Score: {final_score:.1f}")

            # Step 3: Update Airtable
            has_existing = record['existing_score'] is not None
            if update_airtable_score(table, record['record_id'], final_score, has_existing, dry_run):
                successful += 1
            else:
                failure = {
                    "error": "Failed to update Airtable record",
                    "error_stage": "update_airtable",
                    "error_type": "AirtableUpdateFailed",
                    "error_context": {
                        "record_id": record["record_id"],
                        "name": record["name"],
                        "final_score": final_score,
                        "has_existing_score": has_existing,
                    },
                }
                failures.append(failure)
                log_error_result(logger, "Record failed", failure)
                failed += 1

        except Exception as e:
            failure = {
                "error": str(e),
                "error_stage": "process_record",
                "error_type": type(e).__name__,
                "error_context": {
                    "record_id": record["record_id"],
                    "name": record["name"],
                    "audio_path": audio_path,
                },
            }
            failures.append(failure)
            logger.exception("Unhandled error while processing record %s", record["record_id"])
            failed += 1

        finally:
            # Clean up temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Rate limiting - respect Airtable's 5 req/sec limit
        time.sleep(0.5)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(records)}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    if failures:
        logger.info("Failure summary:")
        for failure in failures:
            log_error_result(logger, "Failure summary", failure, level=logging.INFO)
    logger.info(f"Completed at: {datetime.now().isoformat()}")


def main():
    parser = argparse.ArgumentParser(
        description='Airtable Pronunciation Assessment Cron Job',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without updating Airtable'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Maximum number of records to process (default: all)'
    )

    args = parser.parse_args()

    process_pending_records(
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Cron script crashed")
        raise
