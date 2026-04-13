# Rendi - Pronunciation Assessment System

Automated pronunciation assessment for interview recordings using AI.

## Features

- **OpenAI GPT-4o Assessment**: Multimodal pronunciation analysis
- **Azure Speech Assessment**: Alternative assessment using Azure Speech SDK
- **Airtable Integration**: Automatic processing of video recordings from Airtable
- **Rendi API**: Cloud-based video-to-audio conversion (FFmpeg-as-a-service)

## Project Structure

```
rendi/
├── src/                           # Source modules
│   ├── assessment/                # Pronunciation assessment
│   │   ├── openai_assessment.py   # GPT-4o assessment
│   │   └── azure_assessment.py    # Azure Speech assessment
│   ├── airtable/                  # Airtable integration
│   │   ├── client.py              # API connection
│   │   └── records.py             # Record operations
│   ├── utils/                     # Utilities
│   │   └── audio_converter.py     # Video-to-audio conversion
│   └── config.py                  # Configuration
├── data/                          # Data directories
│   ├── audio/                     # Audio files
│   └── results/                   # Assessment results
├── tests/                         # Test files
├── assess.py                      # Single file assessment CLI
├── cron.py                        # Airtable cron job
├── .env                           # Environment variables (not in git)
├── .env.example                   # Example environment file
└── requirements.txt               # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd rendi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your API keys:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Azure Speech (optional)
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=eastus

# Airtable
AIRTABLE_API_KEY=pat...
AIRTABLE_BASE_ID=app...
AIRTABLE_TEST_RESULTS_TABLE_ID=tbl...

# Rendi API
RENDI_API_KEY=...
```

## Usage

### Single File Assessment

```bash
# Using OpenAI (default)
python assess.py audio_1.wav

# Using Azure Speech
python assess.py audio_1.wav --provider azure

# Specify language
python assess.py interview.wav --language en-GB
```

### Airtable Cron Job

```bash
# Process all new records (from Feb 1, 2026)
python cron.py

# Dry run - preview without updating
python cron.py --dry-run

# Limit batch size
python cron.py --batch-size 50
```

## Assessment Scores

The system evaluates four dimensions (0-100 scale):

| Dimension | Description |
|-----------|-------------|
| **Accuracy** | Correct phoneme and word pronunciation |
| **Fluency** | Speech flow, pace, and naturalness |
| **Pronunciation** | Overall pronunciation quality |
| **Prosody** | Rhythm, stress, and intonation |

**Final Score** = Average of all four dimensions

## Cost Estimate

Per video assessment:
- Rendi API: ~$0.02
- OpenAI GPT-4o: ~$0.03
- **Total: ~$0.05/video**

## Legacy Scripts

Original standalone scripts are preserved in `legacy/` for reference:
- `legacy/extract_audio.py`: Video-to-audio extraction
- `legacy/openai_pronunciation_assessment.py`: OpenAI assessment
- `legacy/pronunciation_assessment.py`: Azure assessment
- `legacy/batch_pronunciation_assessment.py`: Batch CSV processing
- `legacy/airtable_pronunciation_cron.py`: Original Airtable cron

## Data Directory

```
data/
├── audio/              # Audio files (WAV)
├── results/            # Assessment JSON results
└── batch_assessment/   # Batch processing outputs
```

## Deployment

The cron job is designed to run on Render as a scheduled task.
