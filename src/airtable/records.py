"""
Airtable Records Module.

Handles fetching and updating records in Airtable.
"""

import logging

from ..config import (
    VIDEO_URL_FIELD,
    EXISTING_SCORE_FIELD,
    NEW_SCORE_FIELD
)


logger = logging.getLogger(__name__)


def get_records_needing_assessment(table, batch_size=None, field_exists=True, reprocess=False):
    """
    Get records that need pronunciation assessment.

    Criteria:
    - Has video URL in 'Question 1 DO URL'
    - Does NOT have 'Pronunciation Assessment Score' (not yet processed by us)
      OR reprocess=True to include already processed records
    - Created from Feb 1, 2026 onwards

    Args:
        table: Airtable Table object
        batch_size: Maximum number of records to return (None for all)
        field_exists: Whether the Pronunciation Assessment Score field exists
        reprocess: If True, include records that already have scores (for re-assessment)

    Returns:
        List of dicts with record_id, video_url, existing_score, name
    """
    logger.info("Fetching records from Airtable...")
    if reprocess:
        logger.info("REPROCESS MODE: Including records with existing Pronunciation Assessment Score")

    # Build formula to filter records
    # Records with video URL, created from Feb 1, 2026
    date_filter = "IS_AFTER({Created}, '2026-01-31')"

    if reprocess:
        # Reprocess mode: get records WITH existing Pronunciation Assessment Score
        # These are records we've already processed that need re-scoring with new prompt
        formula = f"AND({{Question 1 DO URL}} != '', {{Pronunciation Assessment Score}} != '', {date_filter})"
    elif field_exists:
        formula = f"AND({{Question 1 DO URL}} != '', {{Pronunciation Assessment Score}} = '', {date_filter})"
    else:
        # If field doesn't exist yet, just get records with video URL and date filter
        formula = f"AND({{Question 1 DO URL}} != '', {date_filter})"

    try:
        # Fetch all matching records
        records = table.all(formula=formula)
        logger.info(f"Found {len(records)} records needing assessment")

        # Process records
        results = []
        for record in records:
            fields = record.get('fields', {})
            video_url = fields.get(VIDEO_URL_FIELD, '')

            # Skip if no valid URL
            if not video_url or not video_url.startswith('http'):
                continue

            # Get existing score
            existing_score = fields.get(EXISTING_SCORE_FIELD)
            if isinstance(existing_score, str):
                try:
                    existing_score = float(existing_score) if existing_score else None
                except ValueError:
                    existing_score = None

            results.append({
                'record_id': record['id'],
                'video_url': video_url,
                'existing_score': existing_score,
                'name': fields.get('Name', '') or fields.get('Name and Date', ''),
            })

        logger.info(f"Found {len(results)} records with valid video URLs")

        # Apply batch size limit if specified
        if batch_size and len(results) > batch_size:
            results = results[:batch_size]
            logger.info(f"Limited to {batch_size} records")

        return results

    except Exception as e:
        logger.exception("Error fetching records")
        return []


def update_airtable_score(table, record_id: str, score: float, has_existing_score: bool, dry_run: bool = False) -> bool:
    """
    Update Airtable record with pronunciation score.

    Business Logic:
    - If has_existing_score=False: update 'Video 1 score'
    - If has_existing_score=True: update 'Pronunciation Assessment Score'

    Args:
        table: Airtable Table object
        record_id: Airtable record ID
        score: Pronunciation score to save
        has_existing_score: Whether the record already has a Video 1 score
        dry_run: If True, don't actually update

    Returns:
        True if update succeeded, False otherwise
    """
    if has_existing_score:
        field_name = NEW_SCORE_FIELD
    else:
        field_name = EXISTING_SCORE_FIELD

    logger.info(f"Updating field '{field_name}' with score {score:.1f}")

    if dry_run:
        logger.info(f"[DRY RUN] Would update record {record_id}")
        return True

    try:
        table.update(record_id, {field_name: round(score, 1)})
        logger.info("Successfully updated Airtable record")
        return True
    except Exception as e:
        logger.exception("Error updating Airtable")
        return False
