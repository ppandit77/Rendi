# Rendi Pronunciation Assessment Tools

Python scripts for extracting interview audio with Rendi and running pronunciation assessment workflows with OpenAI, Azure Speech, and Airtable.

## Included scripts

- `extract_audio.py`: Extract WAV audio from a remote video URL through the Rendi API.
- `openai_pronunciation_assessment.py`: Run pronunciation analysis on an audio file with OpenAI.
- `pronunciation_assessment.py`: Run pronunciation analysis with Azure Speech.
- `batch_pronunciation_assessment.py`: Process many interview records from a CSV file and save results locally.
- `airtable_pronunciation_cron.py`: Pull pending records from Airtable, assess them, and write scores back.

## Setup

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Fill in the required API keys and service identifiers.

## Environment variables

- `OPENAI_API_KEY`
- `RENDI_API_KEY`
- `AIRTABLE_API_KEY`
- `AIRTABLE_BASE_ID`
- `AIRTABLE_TEST_RESULTS_TABLE_ID`
- `SPEECH_KEY`
- `SPEECH_REGION`
- `BATCH_ASSESSMENT_CSV` (optional override for the source CSV path)
- `BATCH_ASSESSMENT_OUTPUT_DIR` (optional override for the output directory)

Large datasets, generated audio, assessment outputs, and local secrets are intentionally excluded from git.
