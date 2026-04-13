import os
import requests
import time
import sys
from dotenv import load_dotenv

load_dotenv()

RENDI_API_URL = "https://api.rendi.dev/v1/run-ffmpeg-command"
RENDI_STATUS_URL = "https://api.rendi.dev/v1/commands"
RENDI_API_KEY = os.getenv("RENDI_API_KEY")


def extract_audio(video_url: str, api_key: str | None = None, output_file: str = "output.wav") -> str:
    """
    Extract audio from a video URL using Rendi API.
    Output is optimized for Azure Speech API (16kHz, mono, 16-bit PCM WAV).
    """
    api_key = api_key or RENDI_API_KEY
    if not api_key:
        raise ValueError("RENDI_API_KEY is not configured. Set it in your environment or .env file.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    # FFmpeg command for Azure Speech API compatible audio
    # -ar 16000: 16kHz sample rate
    # -ac 1: mono channel
    # -acodec pcm_s16le: 16-bit PCM little-endian
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
    print(f"Submitting job to Rendi API...")
    response = requests.post(RENDI_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    job_data = response.json()
    command_id = job_data.get("command_id")
    print(f"Job submitted. Command ID: {command_id}")

    # Poll for completion
    status_url = f"{RENDI_STATUS_URL}/{command_id}"
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data.get("status")

        print(f"Status: {status}")

        if status == "SUCCESS":
            break
        elif status == "FAILED":
            error = status_data.get("error", "Unknown error")
            raise Exception(f"Job failed: {error}")

        time.sleep(2)

    # Download the output file
    output_files = status_data.get("output_files", {})
    if not output_files:
        raise Exception("No output files returned")

    out_file_info = output_files.get("out_1")
    if not out_file_info:
        raise Exception("Output file info not found in response")

    # Extract storage_url from the file info dict
    download_url = out_file_info.get("storage_url") if isinstance(out_file_info, dict) else out_file_info
    if not download_url:
        raise Exception("Output file URL not found in response")

    print(f"Downloading audio from: {download_url}")

    audio_response = requests.get(download_url)
    audio_response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(audio_response.content)

    print(f"Audio saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <video_url> [output_file]")
        print("Example: python extract_audio.py 'https://example.com/video.mp4' 'output.wav'")
        sys.exit(1)

    video_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

    extract_audio(video_url, output_file=output_file)
