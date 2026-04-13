"""
Airtable Pronunciation Assessment Cron Script

This script:
1. Reads records from Airtable Test Results table
2. Finds records with video URLs that need assessment
3. Converts videos to audio using Rendi API
4. Runs OpenAI GPT-4o pronunciation assessment
5. Updates the appropriate score field in Airtable

Business Logic:
- If "Video 1 score" is empty → update "Video 1 score"
- If "Video 1 score" has value → update "Pronunciation Assessment Score"

Usage:
    python airtable_pronunciation_cron.py [--dry-run] [--batch-size N]
"""

import os
import sys
import json
import time
import base64
import tempfile
import logging
import argparse
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pyairtable import Api, Table

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
# Use Test Results table (not Applicants)
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TEST_RESULTS_TABLE_ID", "tblR3TK9Dieqncu7l")

# Rendi API (FFmpeg-as-a-service)
# This is the base64 encoded API key for Rendi
RENDI_API_KEY = os.getenv("RENDI_API_KEY")
RENDI_API_URL = "https://api.rendi.dev/v1/run-ffmpeg-command"
RENDI_STATUS_URL = "https://api.rendi.dev/v1/commands"

# Field names
VIDEO_URL_FIELD = "Question 1 DO URL"
EXISTING_SCORE_FIELD = "Video 1 score"
NEW_SCORE_FIELD = "Pronunciation Assessment Score"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


ASSESSMENT_PROMPT = """You are an expert pronunciation assessor and speech language pathologist. Analyze this audio recording and provide a detailed pronunciation assessment.

The speaker's target language is: {language}

Evaluate the speaker on these dimensions using a 0-100 scale:

1. ACCURACY (0-100): How correctly individual phonemes and words are pronounced
2. FLUENCY (0-100): Speech flow, pace, hesitations, and naturalness
3. PRONUNCIATION (0-100): Overall pronunciation quality combining clarity and correctness
4. PROSODY (0-100): Rhythm, stress patterns, and intonation

Also identify specific words that were mispronounced or unclear.

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{{
  "transcription": "full transcription of what was said",
  "scores": {{
    "accuracy": <number 0-100>,
    "fluency": <number 0-100>,
    "pronunciation": <number 0-100>,
    "prosody": <number 0-100>
  }},
  "words": [
    {{"word": "problematic_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"}}
  ],
  "assessment_notes": "brief notes on pronunciation patterns observed"
}}

IMPORTANT:
- Only include words in the "words" array that have pronunciation issues (accuracy_score < 80)
- If all words are pronounced well, the "words" array can be empty
- Scores should reflect realistic assessment - perfect 100s are rare"""


def get_airtable_table():
    """Initialize Airtable connection."""
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)
    return api, table


def ensure_field_exists(api):
    """
    Ensure the Pronunciation Assessment Score field exists in the table.
    Creates it if it doesn't exist.
    """
    try:
        base = api.base(AIRTABLE_BASE_ID)
        schema = base.schema()

        # Find the Test Results table
        test_results_table = None
        for table in schema.tables:
            if table.id == AIRTABLE_TABLE_ID:
                test_results_table = table
                break

        if not test_results_table:
            logger.error("Test Results table not found")
            return False

        # Check if field exists
        field_exists = any(
            field.name == NEW_SCORE_FIELD
            for field in test_results_table.fields
        )

        if field_exists:
            logger.info(f"Field '{NEW_SCORE_FIELD}' already exists")
            return True

        # Create the field
        logger.info(f"Creating field '{NEW_SCORE_FIELD}'...")

        # Use Airtable API to create field
        import requests
        url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables/{AIRTABLE_TABLE_ID}/fields"
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": NEW_SCORE_FIELD,
            "type": "number",
            "options": {
                "precision": 1
            }
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info(f"Field '{NEW_SCORE_FIELD}' created successfully")
            return True
        else:
            logger.error(f"Failed to create field: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error checking/creating field: {e}")
        return False


def get_records_needing_assessment(table, batch_size=None, field_exists=True):
    """
    Get records that need pronunciation assessment.

    Criteria:
    - Has video URL in 'Question 1 DO URL'
    - Does NOT have 'Pronunciation Assessment Score' (not yet processed by us)
    """
    logger.info("Fetching records from Airtable...")

    # Build formula to filter records
    # Records with video URL, no Pronunciation Assessment Score, created from Feb 1, 2026
    date_filter = "IS_AFTER({Created}, '2026-01-31')"

    if field_exists:
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
        logger.error(f"Error fetching records: {e}")
        return []


def convert_video_to_audio_rendi(video_url, output_audio_path):
    """
    Convert video to audio using Rendi API (FFmpeg-as-a-service).
    """
    try:
        headers = {
            "X-API-KEY": RENDI_API_KEY,
            "Content-Type": "application/json"
        }

        # FFmpeg command for 16kHz mono WAV
        payload = {
            "input_files": {
                "in_1": video_url
            },
            "output_files": {
                "out_1": "output.wav"
            },
            "ffmpeg_command": "-i {{in_1}} -ar 16000 -ac 1 -acodec pcm_s16le {{out_1}}"
        }

        # Submit job
        logger.info(f"  Submitting to Rendi API...")
        response = requests.post(RENDI_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        job_data = response.json()
        command_id = job_data.get("command_id")

        if not command_id:
            logger.error("  No command_id returned from Rendi")
            return False

        logger.info(f"  Rendi job started: {command_id}")

        # Poll for completion
        status_url = f"{RENDI_STATUS_URL}/{command_id}"
        for _ in range(60):  # Max 5 minutes
            time.sleep(3)
            status_response = requests.get(status_url, headers=headers, timeout=30)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")

            if status == "SUCCESS":
                # Download the output file
                output_files = status_data.get("output_files", {})
                out_file_info = output_files.get("out_1")

                if not out_file_info:
                    logger.error("  No output file in Rendi response")
                    return False

                download_url = out_file_info.get("storage_url") if isinstance(out_file_info, dict) else out_file_info

                if not download_url:
                    logger.error("  No download URL found")
                    return False

                audio_response = requests.get(download_url, timeout=60)
                audio_response.raise_for_status()

                with open(output_audio_path, 'wb') as f:
                    f.write(audio_response.content)

                logger.info(f"  Audio saved: {output_audio_path}")
                return True

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                logger.error(f"  Rendi job failed: {error}")
                return False

            logger.debug(f"    Status: {status}...")

        logger.error("  Rendi job timed out")
        return False

    except Exception as e:
        logger.error(f"  Error with Rendi API: {e}")
        return False


def assess_pronunciation_openai(audio_file, language="en-US"):
    """Run OpenAI pronunciation assessment on audio file."""
    try:
        # Read and encode audio
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        # Get file extension
        ext = audio_file.lower().rsplit(".", 1)[-1]

        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Call GPT-4o with audio
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ASSESSMENT_PROMPT.format(language=language)
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": ext if ext in ["wav", "mp3"] else "wav"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3
        )

        response_text = response.choices[0].message.content

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())

        # Calculate final score
        scores = result.get("scores", {})
        valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
        if valid_scores:
            result["final_score"] = sum(valid_scores) / len(valid_scores)

        return result

    except Exception as e:
        return {"error": str(e)}


def update_airtable_score(table, record_id, score, has_existing_score, dry_run=False):
    """
    Update Airtable record with pronunciation score.

    Business Logic:
    - If has_existing_score=False: update 'Video 1 score'
    - If has_existing_score=True: update 'Pronunciation Assessment Score'
    """
    if has_existing_score:
        field_name = NEW_SCORE_FIELD
    else:
        field_name = EXISTING_SCORE_FIELD

    logger.info(f"  Updating field '{field_name}' with score {score:.1f}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would update record {record_id}")
        return True

    try:
        table.update(record_id, {field_name: round(score, 1)})
        logger.info(f"  Successfully updated Airtable record")
        return True
    except Exception as e:
        logger.error(f"  Error updating Airtable: {e}")
        return False


def process_pending_records(dry_run=False, batch_size=None):
    """
    Main processing function.

    Processes all records that need assessment.
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

    for i, record in enumerate(records):
        logger.info("")
        logger.info(f"Processing {i+1}/{len(records)}: {record['name'][:40] if record['name'] else 'Unknown'}")
        logger.info(f"  Record ID: {record['record_id']}")
        logger.info(f"  Existing score: {record['existing_score']}")

        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            # Step 1: Convert video to audio
            logger.info("  Converting video to audio...")
            if not convert_video_to_audio_rendi(record['video_url'], audio_path):
                logger.error("  Skipping due to conversion failure")
                failed += 1
                continue

            # Step 2: Run pronunciation assessment
            logger.info("  Running OpenAI assessment...")
            assessment = assess_pronunciation_openai(audio_path)

            if 'error' in assessment:
                logger.error(f"  Assessment error: {assessment['error']}")
                failed += 1
                continue

            # Get final score
            final_score = assessment.get('final_score')
            if final_score is None:
                logger.error("  No final score in assessment")
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
                failed += 1

            processed += 1

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
    logger.info(f"Completed at: {datetime.now().isoformat()}")


def main():
    parser = argparse.ArgumentParser(
        description='Airtable Pronunciation Assessment Cron Job'
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
    main()
