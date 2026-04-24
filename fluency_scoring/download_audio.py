#!/usr/bin/env python3
"""
Stage 0: Download videos and convert to 16kHz mono WAV audio.

Uses the existing Rendi API approach for video-to-audio conversion.
This script:
1. Reads labels.csv
2. Converts videos to audio using Rendi API (ffmpeg-as-a-service)
3. Implements aggressive caching by record_id + url_hash
4. Generates download_manifest.csv with detailed logging

Usage:
    python download_audio.py [--limit N] [--workers N] [--skip-existing]
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audio_converter import convert_video_to_audio_rendi_with_details

# Configure logging
log_dir = Path(__file__).parent / 'data'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'download.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
MANIFEST_PATH = Path(__file__).parent / "data" / "download_manifest.csv"
LABELS_PATH = Path(__file__).parent / "data" / "labels.csv"

# Audio settings (Rendi API already produces 16kHz mono WAV)
MIN_SPEECH_SECONDS = 5  # Minimum speech required

# Download settings
MAX_CONCURRENT_DOWNLOADS = 1  # Sequential to avoid 429 rate limits
RETRY_DELAYS = [5, 15, 30]  # Longer exponential backoff for rate limits
DELAY_BETWEEN_REQUESTS = 3  # Seconds between API calls


def get_cache_key(record_id: str, url_hash: str) -> str:
    """Generate cache key from record_id and url_hash."""
    return f"{record_id}_{url_hash}"


def get_audio_path(record_id: str, url_hash: str) -> Path:
    """Get the expected audio file path for a record."""
    cache_key = get_cache_key(record_id, url_hash)
    return AUDIO_DIR / f"{cache_key}.wav"


def check_cache(record_id: str, url_hash: str) -> Optional[Path]:
    """Check if audio is already cached and valid."""
    audio_path = get_audio_path(record_id, url_hash)
    if audio_path.exists() and audio_path.stat().st_size > 1000:  # At least 1KB
        return audio_path
    return None


def get_audio_duration_from_file(audio_path: Path) -> Optional[float]:
    """
    Get audio duration in seconds from WAV file header.
    Works without ffprobe by reading WAV header.
    """
    try:
        import wave
        with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        logger.debug(f"Could not read WAV duration: {e}")
        return None


def process_record(record: dict, skip_existing: bool = True) -> dict:
    """
    Process a single record: convert video to audio using Rendi API.

    Returns manifest entry dict.
    """
    record_id = record["record_id"]
    video_url = record["video_url"]
    url_hash = record.get("url_hash") or hashlib.md5(video_url.encode()).hexdigest()[:8]

    result = {
        "record_id": record_id,
        "url_hash": url_hash,
        "video_url": video_url,
        "status": "unknown",
        "audio_path": "",
        "audio_duration_seconds": 0,
        "error": "",
        "timestamp": datetime.now().isoformat()
    }

    audio_path = get_audio_path(record_id, url_hash)

    # Check cache
    if skip_existing and audio_path.exists() and audio_path.stat().st_size > 1000:
        duration = get_audio_duration_from_file(audio_path)
        result.update({
            "status": "cached",
            "audio_path": str(audio_path),
            "audio_duration_seconds": duration or 0
        })
        logger.debug(f"[{record_id}] Using cached audio")
        return result

    # Ensure audio directory exists
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Use Rendi API for conversion with retries
    conversion_result = None
    last_error = None

    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            conversion_result = convert_video_to_audio_rendi_with_details(
                video_url, str(audio_path)
            )

            if conversion_result.get("ok"):
                break
            else:
                last_error = conversion_result.get("error", "Unknown error")
                if attempt < len(RETRY_DELAYS) - 1:
                    logger.warning(f"[{record_id}] Conversion attempt {attempt+1} failed: {last_error}, retrying in {delay}s")
                    time.sleep(delay)

        except Exception as e:
            last_error = str(e)
            if attempt < len(RETRY_DELAYS) - 1:
                logger.warning(f"[{record_id}] Exception on attempt {attempt+1}: {e}, retrying in {delay}s")
                time.sleep(delay)

    if not conversion_result or not conversion_result.get("ok"):
        result.update({
            "status": "conversion_failed",
            "error": last_error or "Unknown error"
        })
        logger.error(f"[{record_id}] Conversion failed: {last_error}")
        return result

    # Check if file was created
    if not audio_path.exists():
        result.update({
            "status": "file_missing",
            "error": "Audio file not created"
        })
        logger.error(f"[{record_id}] Audio file not created")
        return result

    # Get audio duration
    audio_duration = get_audio_duration_from_file(audio_path)

    # Check minimum duration
    if audio_duration and audio_duration < MIN_SPEECH_SECONDS:
        result.update({
            "status": "too_short",
            "audio_duration_seconds": audio_duration,
            "error": f"Audio only {audio_duration:.1f}s (min {MIN_SPEECH_SECONDS}s)"
        })
        # Don't delete - might still be useful
        logger.warning(f"[{record_id}] Audio too short: {audio_duration:.1f}s")
        return result

    result.update({
        "status": "success",
        "audio_path": str(audio_path),
        "audio_duration_seconds": audio_duration or 0
    })

    logger.info(f"[{record_id}] Success: {audio_duration:.1f}s" if audio_duration else f"[{record_id}] Success")

    # Rate limit delay between successful requests
    time.sleep(DELAY_BETWEEN_REQUESTS)

    return result


def load_labels(labels_path: Path, limit: Optional[int] = None, exclude_non_decade: bool = True) -> list:
    """Load records from labels.csv, optionally excluding non-decade scores."""

    # Standard decade scores
    decade_scores = {0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

    records = []
    excluded = []

    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row["score"])
            except (ValueError, KeyError):
                continue

            record = {
                "record_id": row["record_id"],
                "video_url": row["video_url"],
                "url_hash": row.get("url_hash", ""),
                "score": score,
                "name": row.get("name", ""),
                "email": row.get("email", "")
            }

            if exclude_non_decade and score not in decade_scores:
                excluded.append(record)
                continue

            records.append(record)

            if limit and len(records) >= limit:
                break

    logger.info(f"Loaded {len(records)} records from {labels_path}")
    if excluded:
        logger.info(f"Excluded {len(excluded)} non-decade score records")
        # Save excluded records for held-out eval
        excluded_path = labels_path.parent / "held_out_non_decade.csv"
        with open(excluded_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["record_id", "video_url", "url_hash", "score", "name", "email"])
            writer.writeheader()
            writer.writerows(excluded)
        logger.info(f"Saved excluded records to {excluded_path}")

    return records


def save_manifest(results: list, manifest_path: Path):
    """Save download manifest to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "record_id", "url_hash", "video_url", "status", "audio_path",
        "audio_duration_seconds", "error", "timestamp"
    ]

    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved manifest to {manifest_path}")


def print_summary(results: list):
    """Print download summary statistics."""
    total = len(results)
    by_status = {}
    for r in results:
        status = r["status"]
        by_status[status] = by_status.get(status, 0) + 1

    success_count = by_status.get("success", 0) + by_status.get("cached", 0)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\nTotal records processed: {total}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(by_status.items()):
        pct = 100 * count / total
        print(f"  {status:<20}: {count:5d} ({pct:5.1f}%)")

    print(f"\nSuccess rate: {100 * success_count / total:.1f}%")

    # Duration statistics for successful downloads
    durations = [r["audio_duration_seconds"] for r in results if r["status"] in ("success", "cached") and r["audio_duration_seconds"] > 0]
    if durations:
        print(f"\nAudio duration statistics (n={len(durations)}):")
        print(f"  Min:  {min(durations):6.1f}s")
        print(f"  Max:  {max(durations):6.1f}s")
        print(f"  Mean: {sum(durations)/len(durations):6.1f}s")
        print(f"  Total: {sum(durations)/3600:.1f} hours")

    # Error analysis
    errors = [r["error"] for r in results if r["error"]]
    if errors:
        error_counts = {}
        for e in errors:
            # Truncate long errors for grouping
            key = e[:50] if len(e) > 50 else e
            error_counts[key] = error_counts.get(key, 0) + 1
        print(f"\nError breakdown:")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {count:4d}x  {err}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download and convert videos to audio')
    parser.add_argument('--limit', type=int, default=500, help='Max records to process (default: 500)')
    parser.add_argument('--workers', type=int, default=MAX_CONCURRENT_DOWNLOADS, help='Concurrent downloads')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip cached files')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 0: VIDEO DOWNLOAD AND AUDIO CONVERSION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Limit: {args.limit} records")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    # Ensure directories exist
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load labels
    if not LABELS_PATH.exists():
        logger.error(f"Labels file not found: {LABELS_PATH}")
        logger.error("Run fetch_airtable.py first to generate labels.csv")
        sys.exit(1)

    records = load_labels(LABELS_PATH, limit=args.limit, exclude_non_decade=True)

    if not records:
        logger.error("No records to process!")
        sys.exit(1)

    print(f"\nProcessing {len(records)} records...")
    print(f"Using Rendi API for video-to-audio conversion")

    # Process records with thread pool
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_record, record, args.skip_existing): record
            for record in records
        }

        for i, future in enumerate(as_completed(futures)):
            record = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.exception(f"Error processing {record['record_id']}: {e}")
                results.append({
                    "record_id": record["record_id"],
                    "url_hash": record.get("url_hash", ""),
                    "video_url": record["video_url"],
                    "status": "error",
                    "error": str(e),
                    "audio_path": "",
                    "audio_duration_seconds": 0,
                    "timestamp": datetime.now().isoformat()
                })

            # Progress update
            if (i + 1) % 25 == 0 or i + 1 == len(records):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                success = sum(1 for r in results if r["status"] in ("success", "cached"))
                print(f"  Progress: {i+1}/{len(records)} ({100*(i+1)/len(records):.1f}%) - {rate:.2f}/sec - {success} success")

    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Save manifest
    save_manifest(results, MANIFEST_PATH)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
