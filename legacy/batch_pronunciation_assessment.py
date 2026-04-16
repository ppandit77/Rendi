"""
Batch Pronunciation Assessment Script

This script:
1. Extracts video URLs from the CSV file (prioritizing xobin, latest dates)
2. Converts videos to audio using Rendi API
3. Runs OpenAI pronunciation assessment
4. Compares results with existing Video 1 scores

Usage:
    python batch_pronunciation_assessment.py [--limit N] [--skip-conversion]
"""

import csv
import os
import sys
import json
import time
import logging
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.assessment.openai_assessment import assess_pronunciation_openai as assess_pronunciation_openai_shared
from src.utils.audio_converter import convert_video_to_audio_rendi_with_details
from src.utils.logging_utils import log_error_result, setup_logging

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RENDI_API_KEY = os.getenv("RENDI_API_KEY")
RENDI_API_URL = "https://api.rendi.dev/v1/run-ffmpeg-command"
RENDI_STATUS_URL = "https://api.rendi.dev/v1/commands"

# Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("BATCH_ASSESSMENT_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "batch_assessment"))
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
DEFAULT_CSV_PATH = os.getenv("BATCH_ASSESSMENT_CSV", os.path.join(PROJECT_ROOT, "Test Results-Grid view.csv"))

# Create directories
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = setup_logging(__name__)


def parse_date(date_str):
    """Parse date from various formats in the CSV."""
    if not date_str:
        return None
    try:
        # Try format like "11/14/2025 3:36pm"
        return datetime.strptime(date_str.split()[0], "%m/%d/%Y")
    except:
        pass
    try:
        # Try format from Record field like "11/14/25"
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', date_str)
        if match:
            date_part = match.group(1)
            if len(date_part.split('/')[-1]) == 2:
                return datetime.strptime(date_part, "%m/%d/%y")
            else:
                return datetime.strptime(date_part, "%m/%d/%Y")
    except:
        pass
    return None


def extract_video_entries(csv_path, limit=100):
    """
    Extract video URLs from CSV, prioritizing xobin and latest dates.
    Returns list of dicts with url, name, date, existing_score.
    """
    entries = []

    print(f"Reading CSV file: {csv_path}")
    print("This may take a moment for large files...")

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get the Question 1 DO URL (column 25 in 1-indexed)
            video_url = row.get('Question 1 DO URL', '').strip()

            # Skip if no video URL
            if not video_url or not video_url.startswith('http'):
                continue

            # Get existing score
            video_score = row.get('Video 1 score', '').strip()
            try:
                video_score = float(video_score) if video_score else None
            except ValueError:
                video_score = None

            # Skip if no existing score (nothing to compare)
            if video_score is None:
                continue

            # Get metadata
            record = row.get('Record', '')
            name = row.get('Name', '') or row.get('Name and Date', '')
            created_date = row.get('created date', '')
            source = row.get('Source', '')

            # Parse date
            date = parse_date(created_date) or parse_date(record)

            # Check if xobin
            is_xobin = 'xobin' in video_url.lower() or 'xobin' in source.lower() or 'xobin' in record.lower()

            entries.append({
                'url': video_url,
                'name': name,
                'record': record,
                'date': date,
                'existing_score': video_score,
                'is_xobin': is_xobin,
                'source': source
            })

    print(f"Found {len(entries)} entries with video URLs and scores")

    # Sort: xobin first, then by date (newest first)
    entries.sort(key=lambda x: (
        not x['is_xobin'],  # xobin first (False < True)
        -(x['date'].timestamp() if x['date'] else 0)  # newest first
    ))

    # Take top N
    selected = entries[:limit]

    xobin_count = sum(1 for e in selected if e['is_xobin'])
    print(f"Selected {len(selected)} entries ({xobin_count} from xobin)")

    return selected


def download_video(url, output_path):
    """Download video from URL."""
    try:
        print(f"  Downloading: {url[:80]}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def convert_video_to_audio_rendi(video_url, output_audio_path):
    """
    Convert video to audio using Rendi API (FFmpeg-as-a-service).
    Delegates to the shared converter so failures retain structured context.
    """
    conversion_result = convert_video_to_audio_rendi_with_details(video_url, output_audio_path)
    if conversion_result["ok"]:
        print(f"  Audio saved: {output_audio_path}")
        return conversion_result

    print(f"  Error: {conversion_result.get('error', 'Conversion failed')}")
    print("  Falling back to local FFmpeg conversion...")
    fallback_result = convert_video_to_audio_local(video_url, output_audio_path)
    if fallback_result["ok"]:
        return fallback_result

    fallback_result.setdefault("error_context", {})
    fallback_result["error_context"]["rendi_failure"] = conversion_result
    return fallback_result


def convert_video_to_audio_local(video_url, output_audio_path):
    """
    Fallback: Download video and convert locally with ffmpeg.
    """
    try:
        # Download video to temp file
        temp_video = output_audio_path.replace('.wav', '_temp.webm')

        if not download_video(video_url, temp_video):
            return {
                "ok": False,
                "error": "Download failed before local conversion",
                "error_stage": "download_video",
                "error_type": "DownloadFailed",
                "error_context": {"video_url": video_url, "temp_video": temp_video},
            }

        # Convert with ffmpeg
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-y', '-i', temp_video,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ], capture_output=True, text=True, timeout=120)

        # Clean up temp file
        if os.path.exists(temp_video):
            os.remove(temp_video)

        if result.returncode == 0:
            print(f"  Audio converted: {output_audio_path}")
            return {
                "ok": True,
                "output_audio_path": output_audio_path,
                "conversion_method": "local_ffmpeg",
            }
        else:
            print(f"  FFmpeg error: {result.stderr[:200]}")
            return {
                "ok": False,
                "error": "FFmpeg conversion failed",
                "error_stage": "local_ffmpeg",
                "error_type": "FFmpegFailed",
                "error_context": {
                    "video_url": video_url,
                    "output_audio_path": output_audio_path,
                    "stderr": result.stderr[:1000],
                    "returncode": result.returncode,
                },
            }

    except Exception as e:
        print(f"  Error in local conversion: {e}")
        return {
            "ok": False,
            "error": f"Error in local conversion: {e}",
            "error_stage": "local_ffmpeg",
            "error_type": type(e).__name__,
            "error_context": {"video_url": video_url, "output_audio_path": output_audio_path},
        }


def assess_pronunciation_openai(audio_file, language="en-US"):
    """Run OpenAI pronunciation assessment on audio file."""
    return assess_pronunciation_openai_shared(audio_file, language)


def safe_filename(name):
    """Create a safe filename from a name."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe[:50]  # Limit length


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch pronunciation assessment')
    parser.add_argument('--limit', type=int, default=100, help='Number of videos to process')
    parser.add_argument('--skip-conversion', action='store_true', help='Skip video conversion, use existing audio')
    parser.add_argument('--start-from', type=int, default=0, help='Start from entry N (for resuming)')
    args = parser.parse_args()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("ERROR: Please set OPENAI_API_KEY in .env file")
        sys.exit(1)

    if not RENDI_API_KEY:
        print("ERROR: Please set RENDI_API_KEY in .env file")
        sys.exit(1)

    csv_path = DEFAULT_CSV_PATH
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract entries
    entries = extract_video_entries(csv_path, limit=args.limit)

    if not entries:
        print("No valid entries found!")
        sys.exit(1)

    # Save selected entries for reference
    entries_file = os.path.join(OUTPUT_DIR, "selected_entries.json")
    with open(entries_file, 'w') as f:
        json.dump([{
            'url': e['url'],
            'name': e['name'],
            'record': e['record'],
            'date': e['date'].isoformat() if e['date'] else None,
            'existing_score': e['existing_score'],
            'is_xobin': e['is_xobin']
        } for e in entries], f, indent=2)
    print(f"Saved entry list to: {entries_file}")

    # Process each entry
    results = []

    for i, entry in enumerate(entries[args.start_from:], start=args.start_from):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(entries)}: {entry['name'][:40]}")
        print(f"Existing score: {entry['existing_score']}")
        print(f"Source: {'xobin' if entry['is_xobin'] else 'other'}")
        print(f"{'='*60}")

        # Generate safe filename
        safe_name = safe_filename(entry['name']) or f"entry_{i}"
        audio_path = os.path.join(AUDIO_DIR, f"{i:03d}_{safe_name}.wav")
        result_path = os.path.join(RESULTS_DIR, f"{i:03d}_{safe_name}.json")

        # Check if already processed
        if os.path.exists(result_path):
            print(f"  Already processed, loading existing result...")
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            results.append(result_data)
            continue

        try:
            # Step 1: Convert video to audio
            if not args.skip_conversion or not os.path.exists(audio_path):
                print(f"  Converting video to audio...")
                conversion_result = convert_video_to_audio_rendi(entry['url'], audio_path)
                if not conversion_result["ok"]:
                    print(f"  Skipping due to conversion failure")
                    result_data = {
                        'index': i,
                        'name': entry['name'],
                        'url': entry['url'],
                        'existing_score': entry['existing_score'],
                        'is_xobin': entry['is_xobin'],
                        'error': conversion_result.get('error', 'Conversion failed'),
                        'error_stage': conversion_result.get('error_stage', 'convert_video_to_audio'),
                        'error_type': conversion_result.get('error_type', 'ConversionError'),
                        'error_context': conversion_result.get('error_context', {}),
                    }
                    with open(result_path, 'w') as f:
                        json.dump(result_data, f, indent=2)
                    log_error_result(logger, "Batch entry failed", result_data)
                    results.append(result_data)
                    continue

            # Step 2: Run pronunciation assessment
            print(f"  Running OpenAI assessment...")
            assessment = assess_pronunciation_openai(audio_path)

            # Compile result
            result_data = {
                'index': i,
                'name': entry['name'],
                'url': entry['url'],
                'existing_score': entry['existing_score'],
                'is_xobin': entry['is_xobin'],
                'assessment': assessment
            }

            if 'error' in assessment:
                result_data['error'] = assessment.get('error', 'Unknown error')
                result_data['error_stage'] = assessment.get('error_stage', 'assess_pronunciation')
                result_data['error_type'] = assessment.get('error_type', 'AssessmentError')
                result_data['error_context'] = assessment.get('error_context', {})

            # Calculate comparison
            if 'error' not in assessment and 'final_score' in assessment:
                result_data['openai_final_score'] = assessment['final_score']
                result_data['score_difference'] = assessment['final_score'] - entry['existing_score']

            # Save individual result
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            results.append(result_data)

            # Print summary
            if 'error' not in assessment:
                scores = assessment.get('scores', {})
                print(f"  OpenAI Scores:")
                print(f"    Accuracy:      {scores.get('accuracy', 'N/A')}")
                print(f"    Fluency:       {scores.get('fluency', 'N/A')}")
                print(f"    Pronunciation: {scores.get('pronunciation', 'N/A')}")
                print(f"    Prosody:       {scores.get('prosody', 'N/A')}")
                print(f"    Final Score:   {assessment.get('final_score', 'N/A'):.1f}")
                print(f"  Existing Score:  {entry['existing_score']}")
                if 'score_difference' in result_data:
                    diff = result_data['score_difference']
                    print(f"  Difference:      {diff:+.1f} ({'higher' if diff > 0 else 'lower'})")
            else:
                print(f"  Error: {assessment.get('error', 'Unknown')}")
                log_error_result(logger, "Batch entry failed", result_data)

            # Small delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            logger.exception("Unhandled error while processing batch entry %s", entry['name'])
            result_data = {
                'index': i,
                'name': entry['name'],
                'url': entry['url'],
                'existing_score': entry['existing_score'],
                'is_xobin': entry['is_xobin'],
                'error': str(e),
                'error_stage': 'process_entry',
                'error_type': type(e).__name__,
                'error_context': {'audio_path': audio_path},
            }
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            results.append(result_data)

    # Generate final comparison report
    print(f"\n{'='*60}")
    print("FINAL COMPARISON REPORT")
    print(f"{'='*60}")

    successful = [r for r in results if 'openai_final_score' in r]
    failed = [r for r in results if 'openai_final_score' not in r]
    failure_reasons = {}
    for item in failed:
        reason = item.get('error', item.get('assessment', {}).get('error', 'Unknown error'))
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    print(f"\nProcessed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda pair: (-pair[1], pair[0])):
            print(f"  {count}: {reason}")

    if successful:
        # Calculate statistics
        existing_scores = [r['existing_score'] for r in successful]
        openai_scores = [r['openai_final_score'] for r in successful]
        differences = [r['score_difference'] for r in successful]

        print(f"\n--- Score Statistics ---")
        print(f"Existing Video 1 Scores:")
        print(f"  Mean: {sum(existing_scores)/len(existing_scores):.1f}")
        print(f"  Min:  {min(existing_scores):.1f}")
        print(f"  Max:  {max(existing_scores):.1f}")

        print(f"\nOpenAI Assessment Scores:")
        print(f"  Mean: {sum(openai_scores)/len(openai_scores):.1f}")
        print(f"  Min:  {min(openai_scores):.1f}")
        print(f"  Max:  {max(openai_scores):.1f}")

        print(f"\nScore Differences (OpenAI - Existing):")
        print(f"  Mean: {sum(differences)/len(differences):+.1f}")
        print(f"  Min:  {min(differences):+.1f}")
        print(f"  Max:  {max(differences):+.1f}")

        # Correlation analysis
        mean_existing = sum(existing_scores) / len(existing_scores)
        mean_openai = sum(openai_scores) / len(openai_scores)

        numerator = sum((e - mean_existing) * (o - mean_openai)
                       for e, o in zip(existing_scores, openai_scores))
        denom_existing = sum((e - mean_existing)**2 for e in existing_scores) ** 0.5
        denom_openai = sum((o - mean_openai)**2 for o in openai_scores) ** 0.5

        if denom_existing > 0 and denom_openai > 0:
            correlation = numerator / (denom_existing * denom_openai)
            print(f"\nCorrelation: {correlation:.3f}")

    # Save final report
    report = {
        'total_processed': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'failure_reasons': failure_reasons,
        'results': results
    }

    if successful:
        report['statistics'] = {
            'existing_mean': sum(existing_scores)/len(existing_scores),
            'openai_mean': sum(openai_scores)/len(openai_scores),
            'difference_mean': sum(differences)/len(differences),
            'correlation': correlation if 'correlation' in dir() else None
        }

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Batch pronunciation assessment crashed")
        raise
