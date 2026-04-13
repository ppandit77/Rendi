"""
OpenAI GPT-4o Pronunciation Assessment Module.

Uses OpenAI's multimodal model to assess pronunciation from audio files.
"""

import base64
import json
from openai import OpenAI

from ..config import OPENAI_API_KEY


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
{{
  "transcription": "full transcription of what was said",
  "scores": {{
    "accuracy": <number 0-100>,
    "fluency": <number 0-100>,
    "pronunciation": <number 0-100>,
    "prosody": <number 0-100>
  }},
  "words": [
    {{"word": "problematic_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"}},
    {{"word": "another_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"}}
  ],
  "assessment_notes": "brief notes on pronunciation patterns observed, strengths, and areas for improvement"
}}

IMPORTANT:
- Only include words in the "words" array that have pronunciation issues (accuracy_score < 80)
- If all words are pronounced well, the "words" array can be empty
- Scores should reflect realistic assessment - perfect 100s are rare
- Be specific in assessment_notes about what the speaker does well and what needs work"""


def assess_pronunciation_openai(audio_file: str, language: str = "en-US", api_key: str = None) -> dict:
    """
    Perform pronunciation assessment using OpenAI GPT-4o.

    Args:
        audio_file: Path to the audio file (WAV format recommended)
        language: Language code (e.g., "en-US", "en-GB", "es-ES")
        api_key: OpenAI API key (optional, uses env var if not provided)

    Returns:
        Dictionary containing pronunciation assessment results with:
        - transcription: Text of what was spoken
        - scores: Dict with accuracy, fluency, pronunciation, prosody
        - words: List of problematic words
        - final_score: Average of all scores
        - assessment_notes: Detailed feedback
    """
    key = api_key or OPENAI_API_KEY

    if not key:
        return {"error": "OpenAI API key not configured"}

    try:
        # Read and encode audio to base64
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        # Determine format from extension
        ext = audio_file.lower().rsplit(".", 1)[-1]

        # Initialize OpenAI client
        client = OpenAI(api_key=key)

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

        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())

        # Calculate final score (average of all dimensions)
        scores = result.get("scores", {})
        valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
        if valid_scores:
            result["final_score"] = sum(valid_scores) / len(valid_scores)

        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse GPT-4o response as JSON: {e}",
            "raw_response": response_text
        }
    except Exception as e:
        return {"error": str(e)}


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
    for key in ["pronunciation", "accuracy", "fluency", "prosody"]:
        value = scores.get(key, 'N/A')
        if isinstance(value, (int, float)):
            print(f"  {key.capitalize():14}: {value:.1f}")
        else:
            print(f"  {key.capitalize():14}: {value}")

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

    if "final_score" in assessment:
        print("\n" + "=" * 60)
        print(f"  FINAL SCORE: {assessment['final_score']:.1f}/100")
        print("=" * 60)
