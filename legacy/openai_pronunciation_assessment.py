"""
OpenAI GPT-4o - Pronunciation Assessment for Interviews

This script performs pronunciation assessment on audio files using OpenAI's
GPT-4o multimodal model. It transcribes the speech and assesses pronunciation
quality WITHOUT requiring reference text.

Ideal for candidate interviews where you don't know what will be said.

Requirements:
    pip install openai python-dotenv

Usage:
    python openai_pronunciation_assessment.py <audio_file> [language]

Example:
    python openai_pronunciation_assessment.py audio_1.wav en-US
"""

import base64
import json
import logging
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logging_utils import build_error_result, log_error_result, setup_logging


logger = setup_logging(__name__)


ASSESSMENT_PROMPT = """You are an expert pronunciation assessor and speech language pathologist. Analyze this audio recording and provide a detailed pronunciation assessment.

The speaker's target language is: {language}

Evaluate the speaker on these dimensions using a 0-100 scale:

1. ACCURACY (0-100): How correctly individual phonemes and words are pronounced. Consider:
   - Correct consonant and vowel sounds
   - Proper word stress placement
   - Phoneme substitutions, omissions, or additions

2. FLUENCY (0-100): Speech flow and naturalness. Consider:
   - Pace and rhythm of speech
   - Hesitations, false starts, and filler words
   - Smoothness of connected speech

3. PRONUNCIATION (0-100): Overall pronunciation quality. This is a holistic score combining:
   - Clarity and intelligibility
   - Native-like pronunciation patterns
   - Consistency across utterances

4. PROSODY (0-100): Rhythm, stress, and intonation patterns. Consider:
   - Sentence-level intonation
   - Appropriate stress on content words
   - Natural rhythm and timing

Also identify specific words that were mispronounced or unclear, providing an accuracy score for each.

You MUST respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
  "transcription": "full transcription of what was said",
  "scores": {
    "accuracy": <number 0-100>,
    "fluency": <number 0-100>,
    "pronunciation": <number 0-100>,
    "prosody": <number 0-100>
  },
  "words": [
    {"word": "problematic_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"},
    {"word": "another_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"}
  ],
  "assessment_notes": "brief notes on pronunciation patterns observed, strengths, and areas for improvement"
}

IMPORTANT:
- Only include words in the "words" array that have pronunciation issues (accuracy_score < 80)
- If all words are pronounced well, the "words" array can be empty
- Scores should reflect realistic assessment - perfect 100s are rare
- Be specific in assessment_notes about what the speaker does well and what needs work"""


def assess_pronunciation_openai(
    audio_file: str,
    api_key: str,
    language: str = "en-US"
) -> dict:
    """
    Perform pronunciation assessment using OpenAI GPT-4o.

    Args:
        audio_file: Path to the audio file (WAV format recommended)
        api_key: OpenAI API key
        language: Language code (e.g., "en-US", "en-GB", "es-ES")

    Returns:
        Dictionary containing pronunciation assessment results
    """
    response_text = ""
    try:
        # Read and encode audio file to base64
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        # Determine MIME type based on file extension
        ext = audio_file.lower().rsplit(".", 1)[-1]

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        print(f"Analyzing pronunciation for: {audio_file}")
        print(f"Language: {language}")
        print("Processing with GPT-4o... (this may take a moment)")
        print("-" * 50)

        # Create the message with audio input
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

        # Extract the response content
        response_text = response.choices[0].message.content

        # Parse JSON from response
        # Clean up response if it has markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())
        return result
    except json.JSONDecodeError as e:
        logger.error("Failed to parse GPT-4o response as JSON for %s: %s", audio_file, e)
        error_result = build_error_result(
            f"Failed to parse GPT-4o response as JSON: {e}",
            error=e,
            stage="parse_response",
            audio_file=audio_file,
            language=language,
            raw_response=response_text,
        )
        error_result["raw_response"] = response_text
        return error_result
    except Exception as e:
        logger.exception("Legacy OpenAI assessment failed for %s", audio_file)
        return build_error_result(
            str(e),
            error=e,
            stage="openai_assessment",
            audio_file=audio_file,
            language=language,
        )


def print_assessment(assessment: dict):
    """Pretty print the assessment results."""
    if "error" in assessment:
        print(f"\nError: {assessment['error']}")
        if "raw_response" in assessment:
            print(f"\nRaw response:\n{assessment['raw_response']}")
        return

    print("\n" + "=" * 60)
    print("PRONUNCIATION ASSESSMENT RESULTS (OpenAI GPT-4o)")
    print("=" * 60)

    print(f"\nTranscription:\n{assessment.get('transcription', 'N/A')}")

    print("\n--- Overall Scores (0-100) ---")
    scores = assessment.get("scores", {})
    print(f"  Pronunciation: {scores.get('pronunciation', 'N/A'):.1f}" if isinstance(scores.get('pronunciation'), (int, float)) else f"  Pronunciation: {scores.get('pronunciation', 'N/A')}")
    print(f"  Accuracy:      {scores.get('accuracy', 'N/A'):.1f}" if isinstance(scores.get('accuracy'), (int, float)) else f"  Accuracy:      {scores.get('accuracy', 'N/A')}")
    print(f"  Fluency:       {scores.get('fluency', 'N/A'):.1f}" if isinstance(scores.get('fluency'), (int, float)) else f"  Fluency:       {scores.get('fluency', 'N/A')}")
    print(f"  Prosody:       {scores.get('prosody', 'N/A'):.1f}" if isinstance(scores.get('prosody'), (int, float)) else f"  Prosody:       {scores.get('prosody', 'N/A')}")

    words = assessment.get("words", [])
    if words:
        print(f"\n--- Words Needing Improvement ({len(words)} identified) ---")
        for word in words:
            score = word.get('accuracy_score', 'N/A')
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            print(f"  '{word.get('word', '?')}': {score_str} ({word.get('error_type', 'Unknown')})")
    else:
        print("\n--- Word-Level Analysis ---")
        print("  No significant pronunciation issues detected!")

    if assessment.get("assessment_notes"):
        print(f"\n--- Assessment Notes ---")
        print(f"  {assessment['assessment_notes']}")

    # Calculate final score (average of all scores)
    if scores:
        valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
        if valid_scores:
            final_score = sum(valid_scores) / len(valid_scores)
            print("\n" + "=" * 60)
            print(f"  FINAL SCORE: {final_score:.1f}/100")
            print("=" * 60)


if __name__ == "__main__":
    try:
        # Load environment variables from .env file
        load_dotenv()

        # OpenAI API key from environment
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if len(sys.argv) < 2:
            print(__doc__)
            print("\nUsage: python openai_pronunciation_assessment.py <audio_file> [language]")
            print("\nExamples:")
            print("  python openai_pronunciation_assessment.py audio_1.wav")
            print("  python openai_pronunciation_assessment.py audio_1.wav en-US")
            print("  python openai_pronunciation_assessment.py interview.wav en-GB")
            sys.exit(1)

        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "en-US"

        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            print("\nTo set credentials:")
            print("1. Create a .env file in the project directory")
            print("2. Add: OPENAI_API_KEY=your_key_here")
            sys.exit(1)

        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"ERROR: Audio file not found: {audio_file}")
            sys.exit(1)

        # Perform assessment
        result = assess_pronunciation_openai(
            audio_file=audio_file,
            api_key=OPENAI_API_KEY,
            language=language
        )

        # Display results
        print_assessment(result)
        if "error" in result:
            log_error_result(
                logger,
                "Legacy OpenAI assessment failed",
                {
                    "error": result.get("error", "Unknown error"),
                    "error_stage": result.get("error_stage", "assessment"),
                    "error_type": result.get("error_type", "AssessmentError"),
                    "error_context": result.get("error_context", {"audio_file": audio_file, "language": language}),
                },
                level=logging.INFO,
            )

        # Save JSON output
        output_json = audio_file.rsplit(".", 1)[0] + "_openai_assessment.json"
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    except Exception:
        logger.exception("Legacy OpenAI assessment script crashed")
        raise
