"""
Configuration settings for the Rendi Pronunciation Assessment System.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-audio-preview")

# Azure Speech Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

# Airtable Configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TEST_RESULTS_TABLE_ID", "tblR3TK9Dieqncu7l")

# Airtable Field Names
VIDEO_URL_FIELD = "Question 1 DO URL"
EXISTING_SCORE_FIELD = "Video 1 score"
NEW_SCORE_FIELD = "Pronunciation Assessment Score"

# Rendi API (FFmpeg-as-a-service)
RENDI_API_KEY = os.getenv("RENDI_API_KEY")
RENDI_API_URL = "https://api.rendi.dev/v1/run-ffmpeg-command"
RENDI_STATUS_URL = "https://api.rendi.dev/v1/commands"

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
