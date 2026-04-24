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
from .logging_utils import build_error_result


logger = logging.getLogger(__name__)

MAX_HTTP_RETRIES = 3
HTTP_RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds


def _request_with_retry(method: str, url: str, max_retries: int = MAX_HTTP_RETRIES, **kwargs) -> requests.Response:
    """
    Make an HTTP request with retry logic for transient failures.

    Args:
        method: HTTP method ('get' or 'post')
        url: Request URL
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments passed to requests

    Returns:
        requests.Response object

    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, **kwargs)
            else:
                response = requests.post(url, **kwargs)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = HTTP_RETRY_DELAYS[attempt]
                logger.warning("HTTP %s to %s failed (attempt %d/%d), retrying in %ds: %s",
                               method.upper(), url[:60], attempt + 1, max_retries, delay, e)
                time.sleep(delay)
            else:
                logger.error("HTTP %s to %s failed after %d attempts: %s",
                             method.upper(), url[:60], max_retries, e)
                raise
        except requests.exceptions.HTTPError:
            # Don't retry HTTP errors (4xx, 5xx) - re-raise immediately
            raise
    raise last_error


def convert_video_to_audio_rendi_with_details(video_url: str, output_audio_path: str) -> dict:
    """
    Convert video to audio using Rendi API (FFmpeg-as-a-service).

    Args:
        video_url: URL of the video file
        output_audio_path: Path to save the output audio file

    Returns:
        Dict containing success flag and diagnostic details
    """
    if not RENDI_API_KEY:
        error_result = build_error_result(
            "RENDI_API_KEY not configured",
            stage="configuration",
            video_url=video_url,
            output_audio_path=output_audio_path,
        )
        logger.error(error_result["error"])
        return {"ok": False, **error_result}

    command_id = None
    status = None

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

        # Submit job (with retry for transient errors)
        logger.info("Submitting to Rendi API...")
        response = _request_with_retry('post', RENDI_API_URL, json=payload, headers=headers, timeout=30)
        job_data = response.json()
        command_id = job_data.get("command_id")

        if not command_id:
            error_result = build_error_result(
                "No command_id returned from Rendi",
                stage="submit_job",
                video_url=video_url,
                output_audio_path=output_audio_path,
                job_response=job_data,
            )
            logger.error(error_result["error"])
            return {"ok": False, **error_result}

        logger.info(f"Rendi job started: {command_id}")

        # Poll for completion
        status_url = f"{RENDI_STATUS_URL}/{command_id}"
        for _ in range(60):  # Max 5 minutes
            time.sleep(3)
            status_response = _request_with_retry('get', status_url, headers=headers, timeout=30)
            status_data = status_response.json()
            status = status_data.get("status")

            if status == "SUCCESS":
                # Download the output file
                output_files = status_data.get("output_files", {})
                out_file_info = output_files.get("out_1")

                if not out_file_info:
                    error_result = build_error_result(
                        "No output file in Rendi response",
                        stage="poll_status",
                        video_url=video_url,
                        output_audio_path=output_audio_path,
                        command_id=command_id,
                        status=status,
                        status_response=status_data,
                    )
                    logger.error(error_result["error"])
                    return {"ok": False, **error_result}

                download_url = out_file_info.get("storage_url") if isinstance(out_file_info, dict) else out_file_info

                if not download_url:
                    error_result = build_error_result(
                        "No download URL found",
                        stage="poll_status",
                        video_url=video_url,
                        output_audio_path=output_audio_path,
                        command_id=command_id,
                        status=status,
                        status_response=status_data,
                    )
                    logger.error(error_result["error"])
                    return {"ok": False, **error_result}

                audio_response = _request_with_retry('get', download_url, timeout=60)

                with open(output_audio_path, 'wb') as f:
                    f.write(audio_response.content)

                logger.info(f"Audio saved: {output_audio_path}")
                return {
                    "ok": True,
                    "command_id": command_id,
                    "status": status,
                    "download_url": download_url,
                    "output_audio_path": output_audio_path,
                }

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                error_result = build_error_result(
                    f"Rendi job failed: {error}",
                    stage="poll_status",
                    video_url=video_url,
                    output_audio_path=output_audio_path,
                    command_id=command_id,
                    status=status,
                    status_response=status_data,
                )
                logger.error(error_result["error"])
                return {"ok": False, **error_result}

            logger.debug(f"Status: {status}...")

        error_result = build_error_result(
            "Rendi job timed out",
            stage="poll_status",
            video_url=video_url,
            output_audio_path=output_audio_path,
            command_id=command_id,
            status=status,
            poll_attempts=60,
        )
        logger.error(error_result["error"])
        return {"ok": False, **error_result}

    except requests.exceptions.HTTPError as e:
        logger.exception("Rendi API HTTP error")
        return {
            "ok": False,
            **build_error_result(
                f"Rendi API HTTP error: {e}",
                error=e,
                stage="submit_or_poll_http",
                video_url=video_url,
                output_audio_path=output_audio_path,
                command_id=command_id,
                status=status,
            ),
        }
    except Exception as e:
        logger.exception("Error with Rendi API")
        return {
            "ok": False,
            **build_error_result(
                f"Error with Rendi API: {e}",
                error=e,
                stage="submit_or_poll",
                video_url=video_url,
                output_audio_path=output_audio_path,
                command_id=command_id,
                status=status,
            ),
        }


def convert_video_to_audio_rendi(video_url: str, output_audio_path: str) -> bool:
    """
    Backward-compatible bool wrapper around the detailed Rendi conversion API.
    """
    return convert_video_to_audio_rendi_with_details(video_url, output_audio_path)["ok"]


def convert_video_to_audio_local_with_details(video_path: str, output_audio_path: str) -> dict:
    """
    Convert video to audio using local FFmpeg.

    Args:
        video_path: Path to the local video file
        output_audio_path: Path to save the output audio file

    Returns:
        Dict containing success flag and diagnostic details
    """
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.info(f"Audio converted: {output_audio_path}")
            return {"ok": True, "output_audio_path": output_audio_path, "video_path": video_path}
        else:
            logger.error(f"FFmpeg error: {result.stderr[:200]}")
            return {
                "ok": False,
                **build_error_result(
                    "FFmpeg conversion failed",
                    stage="local_ffmpeg",
                    video_path=video_path,
                    output_audio_path=output_audio_path,
                    returncode=result.returncode,
                    stderr=result.stderr[:1000],
                ),
            }

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return {
            "ok": False,
            **build_error_result(
                "FFmpeg conversion timed out",
                stage="local_ffmpeg",
                video_path=video_path,
                output_audio_path=output_audio_path,
            ),
        }
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg.")
        return {
            "ok": False,
            **build_error_result(
                "FFmpeg not found. Please install FFmpeg.",
                stage="local_ffmpeg",
                video_path=video_path,
                output_audio_path=output_audio_path,
            ),
        }
    except Exception as e:
        logger.exception("Error in local conversion")
        return {
            "ok": False,
            **build_error_result(
                f"Error in local conversion: {e}",
                error=e,
                stage="local_ffmpeg",
                video_path=video_path,
                output_audio_path=output_audio_path,
            ),
        }


def convert_video_to_audio_local(video_path: str, output_audio_path: str) -> bool:
    """
    Backward-compatible bool wrapper around the detailed local conversion API.
    """
    return convert_video_to_audio_local_with_details(video_path, output_audio_path)["ok"]


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
