"""
Audio Conversion Utilities.

Converts video files to audio using Rendi API or local FFmpeg.
"""

import os
import time
import subprocess
import requests
import logging

from ..config import RENDI_API_KEY, RENDI_API_URL, RENDI_STATUS_URL


logger = logging.getLogger(__name__)


def convert_video_to_audio_rendi(video_url: str, output_audio_path: str) -> bool:
    """
    Convert video to audio using Rendi API (FFmpeg-as-a-service).

    Args:
        video_url: URL of the video file
        output_audio_path: Path to save the output audio file

    Returns:
        True if conversion succeeded, False otherwise
    """
    if not RENDI_API_KEY:
        logger.error("RENDI_API_KEY not configured")
        return False

    try:
        headers = {
            "X-API-KEY": RENDI_API_KEY,
            "Content-Type": "application/json"
        }

        # FFmpeg command for 16kHz mono WAV (optimal for speech recognition)
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
        logger.info("Submitting to Rendi API...")
        response = requests.post(RENDI_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        job_data = response.json()
        command_id = job_data.get("command_id")

        if not command_id:
            logger.error("No command_id returned from Rendi")
            return False

        logger.info(f"Rendi job started: {command_id}")

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
                    logger.error("No output file in Rendi response")
                    return False

                download_url = out_file_info.get("storage_url") if isinstance(out_file_info, dict) else out_file_info

                if not download_url:
                    logger.error("No download URL found")
                    return False

                audio_response = requests.get(download_url, timeout=60)
                audio_response.raise_for_status()

                with open(output_audio_path, 'wb') as f:
                    f.write(audio_response.content)

                logger.info(f"Audio saved: {output_audio_path}")
                return True

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                logger.error(f"Rendi job failed: {error}")
                return False

            logger.debug(f"Status: {status}...")

        logger.error("Rendi job timed out")
        return False

    except requests.exceptions.HTTPError as e:
        logger.error(f"Rendi API HTTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error with Rendi API: {e}")
        return False


def convert_video_to_audio_local(video_path: str, output_audio_path: str) -> bool:
    """
    Convert video to audio using local FFmpeg.

    Args:
        video_path: Path to the local video file
        output_audio_path: Path to save the output audio file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.info(f"Audio converted: {output_audio_path}")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg.")
        return False
    except Exception as e:
        logger.error(f"Error in local conversion: {e}")
        return False


def download_video(url: str, output_path: str) -> bool:
    """
    Download video from URL.

    Args:
        url: Video URL
        output_path: Path to save the video

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        logger.info(f"Downloading: {url[:80]}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        logger.error(f"Error downloading: {e}")
        return False


def convert_video_url_to_audio_local(video_url: str, output_audio_path: str) -> bool:
    """
    Download video from URL and convert to audio locally.

    Args:
        video_url: URL of the video
        output_audio_path: Path to save the audio

    Returns:
        True if successful, False otherwise
    """
    # Download video to temp file
    temp_video = output_audio_path.replace('.wav', '_temp.webm')

    if not download_video(video_url, temp_video):
        return False

    # Convert with local FFmpeg
    success = convert_video_to_audio_local(temp_video, output_audio_path)

    # Clean up temp file
    if os.path.exists(temp_video):
        os.remove(temp_video)

    return success
