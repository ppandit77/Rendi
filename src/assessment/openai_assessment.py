"""
OpenAI GPT-4o Pronunciation Assessment Module.

Uses OpenAI's multimodal model to assess pronunciation from audio files.
Supports multiple prompt strategies:
- v3: STRICT prompt calibrated to match human evaluator standards (stricter)
- v2 (default): Anti-compression prompt with chain-of-thought dimension ratings
- dspy: DSPy-optimized prompt (legacy, causes score compression)
- basic: Simple prompt without few-shot examples
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

from ..config import OPENAI_API_KEY
from ..utils.logging_utils import build_error_result
from .prompt_builder import build_assessment_prompt, build_assessment_prompt_v3, build_assessment_prompt_v4, get_system_message


MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds

# Path to DSPy-optimized prompt (if available) - LEGACY, causes score compression
OPTIMIZED_PROMPT_PATH = Path(__file__).parent.parent.parent / "data" / "optimized_prompt.json"

# Default prompt version: "v2" uses anti-compression design, "dspy" uses old DSPy prompt
DEFAULT_PROMPT_VERSION = "v2"


ASSESSMENT_PROMPT = """You are a STRICT pronunciation assessor evaluating non-native English speakers for professional communication roles.

The speaker's target language is: {language}

CRITICAL SCORING RULES - YOU MUST FOLLOW THESE:

1. BE A STRICT ASSESSOR: Most speakers should score between 50-80. Only exceptional speakers get 85+.

2. USE THE FULL 0-100 SCALE - Expected distribution:
   - 90-100: Exceptional (top 5%) - Near-native, broadcast quality, no detectable accent
   - 80-89: Very Good (top 20%) - Professional level, minimal accent, very clear
   - 70-79: Good (middle 40%) - Clear communication, noticeable accent but easily understood
   - 60-69: Fair (lower 25%) - Understandable with effort, frequent errors, strong accent
   - 50-59: Needs Work (bottom 10%) - Difficult to understand, many errors
   - Below 50: Poor - Significant communication barriers

3. EVALUATE EACH DIMENSION INDEPENDENTLY:
   - ACCURACY (0-100): Correct phonemes, word stress, no substitutions/omissions
   - FLUENCY (0-100): Smooth speech flow, minimal hesitations/filler words
   - PRONUNCIATION (0-100): Overall clarity and intelligibility
   - PROSODY (0-100): Natural rhythm, stress, and intonation patterns

4. DEDUCTIONS FOR COMMON ISSUES:
   - Heavy/thick accent: -15 to -25 points on pronunciation
   - Frequent hesitations/filler words (um, uh, like): -10 to -20 points on fluency
   - Monotone/flat delivery: -15 to -25 points on prosody
   - Each mispronounced word: -3 to -5 points on accuracy
   - Unnatural pacing (too fast/slow): -10 to -15 points on fluency
   - Wrong word stress patterns: -10 to -15 points on prosody

5. DO NOT DEFAULT TO 75-85 FOR EVERYONE. Different speakers have different abilities - your scores MUST reflect this. If you find yourself giving similar scores to different audio samples, you are not being discriminating enough.

You MUST respond ONLY with valid JSON in this exact format:
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
  "assessment_notes": "specific observations about accent strength, speech patterns, and areas needing improvement"
}}

REMEMBER: A score of 75+ means the speaker is GOOD. Most non-native speakers should be in the 55-75 range unless they are exceptional."""


logger = logging.getLogger(__name__)


def load_optimized_prompt() -> Optional[dict]:
    """
    Load the DSPy-optimized prompt configuration if available.

    Returns:
        Dict with 'instructions' and 'demos', or None if not available
    """
    if not OPTIMIZED_PROMPT_PATH.exists():
        return None

    try:
        with open(OPTIMIZED_PROMPT_PATH) as f:
            data = json.load(f)

        result = {}

        # Extract instructions from DSPy saved state
        # Path: assess.predict.signature.instructions
        if "assess.predict" in data:
            predict_data = data["assess.predict"]
            if "signature" in predict_data and "instructions" in predict_data["signature"]:
                result["instructions"] = predict_data["signature"]["instructions"]

            # Extract few-shot demos
            if "demos" in predict_data and predict_data["demos"]:
                result["demos"] = predict_data["demos"]

        if result.get("instructions"):
            logger.info("Loaded DSPy-optimized prompt with %d demos",
                       len(result.get("demos", [])))
            return result

        logger.debug("Optimized prompt file found but no instructions extracted")
        return None

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load optimized prompt: %s", e)
        return None


def format_demo_for_prompt(demo: dict) -> str:
    """Format a single DSPy demo as a few-shot example."""
    parts = []

    # Input
    if "transcription" in demo:
        parts.append(f"Transcription: {demo['transcription'][:500]}...")  # Truncate long transcriptions

    # Reasoning (if augmented)
    if demo.get("augmented") and "reasoning" in demo:
        parts.append(f"Reasoning: {demo['reasoning']}")

    # Scores
    if "accuracy_score" in demo:
        parts.append(f"Accuracy Score: {demo['accuracy_score']}")
    if "fluency_score" in demo:
        parts.append(f"Fluency Score: {demo['fluency_score']}")
    if "pronunciation_score" in demo:
        parts.append(f"Pronunciation Score: {demo['pronunciation_score']}")
    if "prosody_score" in demo:
        parts.append(f"Prosody Score: {demo['prosody_score']}")

    # Other fields
    if "problematic_words" in demo:
        parts.append(f"Problematic Words: {demo['problematic_words']}")
    if "assessment_notes" in demo:
        parts.append(f"Assessment Notes: {demo['assessment_notes']}")

    return "\n".join(parts)


def get_assessment_prompt(language: str, use_optimized: bool = True) -> str:
    """
    Get the assessment prompt, preferring optimized version if available.

    Args:
        language: Target language code
        use_optimized: Whether to try loading the optimized prompt

    Returns:
        Formatted assessment prompt
    """
    if use_optimized:
        optimized = load_optimized_prompt()
        if optimized and "instructions" in optimized:
            logger.info("Using DSPy-optimized prompt")

            # Build prompt with instructions
            prompt_parts = [optimized["instructions"]]

            # Add few-shot demos if available (only augmented ones with full scores)
            demos = optimized.get("demos", [])
            augmented_demos = [d for d in demos if d.get("augmented") and "accuracy_score" in d]

            if augmented_demos:
                prompt_parts.append("\n--- CALIBRATION EXAMPLES (study these score ranges carefully) ---\n")
                # Include ALL demos to show full range: high (~84), mid (~64), and low (~51) performers
                for i, demo in enumerate(augmented_demos, 1):
                    prompt_parts.append(f"Example {i}:")
                    prompt_parts.append(format_demo_for_prompt(demo))
                    prompt_parts.append("")
                prompt_parts.append("--- END EXAMPLES ---\n")
                prompt_parts.append("REMEMBER: Use the full range shown above. Most speakers should fall between 50-75.")

            # Add JSON format specification
            prompt_parts.append(f"""
The speaker's target language is: {language}

You MUST respond ONLY with valid JSON in this exact format:
{{
  "transcription": "full transcription of what was said",
  "scores": {{
    "accuracy": <number 0-100>,
    "fluency": <number 0-100>,
    "pronunciation": <number 0-100>,
    "prosody": <number 0-100>
  }},
  "words": [
    {{"word": "problematic_word", "accuracy_score": <number 0-100>, "error_type": "Mispronunciation"}}
  ],
  "assessment_notes": "specific observations about accent strength, speech patterns, and areas needing improvement"
}}""")

            return "\n".join(prompt_parts)

    return ASSESSMENT_PROMPT.format(language=language)


def assess_pronunciation_openai(
    audio_file: str,
    language: str = "en-US",
    api_key: str = None,
    use_optimized_prompt: bool = True,
    prompt_version: str = None
) -> dict:
    """
    Perform pronunciation assessment using OpenAI GPT-4o.

    Args:
        audio_file: Path to the audio file (WAV format recommended)
        language: Language code (e.g., "en-US", "en-GB", "es-ES")
        api_key: OpenAI API key (optional, uses env var if not provided)
        use_optimized_prompt: DEPRECATED - use prompt_version instead
        prompt_version: Which prompt to use:
            - "v2" (default): Anti-compression prompt with chain-of-thought
            - "dspy": Legacy DSPy-optimized prompt (causes score compression)
            - "basic": Simple prompt without few-shot examples

    Returns:
        Dictionary containing pronunciation assessment results with:
        - transcription: Text of what was spoken
        - scores: Dict with accuracy, fluency, pronunciation, prosody
        - dimension_ratings: (v2 only) Categorical ratings for each dimension
        - score: (v2 only) Final overall score
        - final_score: Average of dimension scores (for backward compatibility)
        - assessment_notes/reasoning: Detailed feedback
    """
    # Determine prompt version
    version = prompt_version or DEFAULT_PROMPT_VERSION

    key = api_key or OPENAI_API_KEY

    if not key:
        return build_error_result(
            "OpenAI API key not configured",
            stage="configuration",
            audio_file=audio_file,
            language=language,
        )

    # Get the appropriate prompt based on version
    if version == "v4":
        prompt_text = build_assessment_prompt_v4(language)
        system_msg = "You are a pronunciation assessor using 1-10 dimension scales. Always respond with valid JSON only. Rate each dimension carefully using the provided anchors."
        logger.info("Using v4 10-point scale prompt")
    elif version == "v3":
        prompt_text = build_assessment_prompt_v3(language)
        system_msg = "You are a STRICT pronunciation assessor for call center screening. Always respond with valid JSON only. Most candidates should score 30-55."
        logger.info("Using v3 STRICT prompt")
    elif version == "v2":
        prompt_text = build_assessment_prompt(language)
        system_msg = get_system_message()
        logger.info("Using v2 anti-compression prompt")
    elif version == "dspy":
        prompt_text = get_assessment_prompt(language, use_optimized=True)
        system_msg = "You are a strict pronunciation assessor. Always respond with valid JSON only, no markdown formatting."
        logger.info("Using legacy DSPy prompt")
    else:  # basic
        prompt_text = ASSESSMENT_PROMPT.format(language=language)
        system_msg = "You are a strict pronunciation assessor. Always respond with valid JSON only, no markdown formatting."
        logger.info("Using basic prompt")

    response_text = ""
    try:
        # Read and encode audio to base64
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        # Determine format from extension
        ext = audio_file.lower().rsplit(".", 1)[-1]

        # Initialize OpenAI client
        client = OpenAI(api_key=key)

        # Call GPT-4o with audio (with retry logic for transient errors)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    modalities=["text"],
                    messages=[
                        {
                            "role": "system",
                            "content": system_msg
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
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
                    temperature=0.8
                )
                break  # Success, exit retry loop
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning("OpenAI API error (attempt %d/%d), retrying in %ds: %s",
                                   attempt + 1, MAX_RETRIES, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("OpenAI API error after %d attempts: %s", MAX_RETRIES, e)
                    return build_error_result(
                        f"Connection error after {MAX_RETRIES} retries: {e}",
                        error=e,
                        stage="openai_api_call",
                        audio_file=audio_file,
                        language=language,
                    )

        response_text = response.choices[0].message.content

        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())

        # Calculate final score
        # v2 format: uses "score" field directly, falls back to averaging "scores" dict
        # Legacy format: averages the "scores" dict
        if "score" in result and isinstance(result["score"], (int, float)):
            # v2 format - model provided explicit final score
            result["final_score"] = result["score"]
        else:
            # Legacy format - average the dimension scores
            scores = result.get("scores", {})
            valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
            if valid_scores:
                result["final_score"] = sum(valid_scores) / len(valid_scores)

        # Log dimension ratings if present (v2 format)
        if "dimension_ratings" in result:
            ratings = result["dimension_ratings"]
            logger.debug("Dimension ratings: %s", ratings)

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
        logger.exception("OpenAI pronunciation assessment failed for %s", audio_file)
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
