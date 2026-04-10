"""
Azure AI Speech - Pronunciation Assessment for Interviews

This script performs pronunciation assessment on audio files WITHOUT requiring
reference text. It transcribes the speech and assesses pronunciation quality.

Ideal for candidate interviews where you don't know what will be said.

Requirements:
    pip install azure-cognitiveservices-speech python-dotenv

Usage:
    python pronunciation_assessment.py <audio_file> [language]

Example:
    python pronunciation_assessment.py audio_1.wav en-US
"""

import azure.cognitiveservices.speech as speechsdk
import sys
import json
import os
import time
from dotenv import load_dotenv


def assess_pronunciation_no_reference(
    audio_file: str,
    speech_key: str,
    speech_region: str,
    language: str = "en-US"
) -> dict:
    """
    Perform pronunciation assessment WITHOUT reference text.
    Transcribes the audio and assesses pronunciation of what was spoken.

    Args:
        audio_file: Path to the audio file (WAV format, 16kHz mono recommended)
        speech_key: Azure Speech API subscription key
        speech_region: Azure Speech service region (e.g., "eastus")
        language: Language code (e.g., "en-US", "en-GB", "es-ES")

    Returns:
        Dictionary containing pronunciation assessment results
    """
    # Configure speech service
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.speech_recognition_language = language

    # Configure audio input from file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Configure pronunciation assessment WITHOUT reference text
    # This enables "unreferenced" assessment mode
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
    )

    # Enable prosody assessment (rhythm, stress, intonation)
    pronunciation_config.enable_prosody_assessment()

    # Create speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # Apply pronunciation assessment config
    pronunciation_config.apply_to(speech_recognizer)

    # Storage for results
    all_words = []
    all_scores = []
    all_text = []
    done = False
    error_message = None

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pronunciation_result = speechsdk.PronunciationAssessmentResult(evt.result)

            segment_scores = {
                "accuracy": pronunciation_result.accuracy_score,
                "fluency": pronunciation_result.fluency_score,
                "pronunciation": pronunciation_result.pronunciation_score,
            }

            # Try to get prosody score
            try:
                segment_scores["prosody"] = pronunciation_result.prosody_score
            except AttributeError:
                pass

            all_scores.append(segment_scores)
            all_text.append(evt.result.text)

            # Get word-level details
            for word in pronunciation_result.words:
                word_info = {
                    "word": word.word,
                    "accuracy_score": word.accuracy_score,
                    "error_type": str(word.error_type) if hasattr(word, 'error_type') else "None"
                }
                all_words.append(word_info)

    def on_session_stopped(evt):
        nonlocal done
        done = True

    def on_canceled(evt):
        nonlocal done, error_message
        done = True
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation = evt.result.cancellation_details
            if cancellation.reason == speechsdk.CancellationReason.Error:
                error_message = f"Error: {cancellation.error_details}"

    # Connect event handlers
    speech_recognizer.recognized.connect(on_recognized)
    speech_recognizer.session_stopped.connect(on_session_stopped)
    speech_recognizer.canceled.connect(on_canceled)

    print(f"Analyzing pronunciation for: {audio_file}")
    print(f"Language: {language}")
    print("Processing... (this may take a moment for longer audio)")
    print("-" * 50)

    # Start continuous recognition for full audio
    speech_recognizer.start_continuous_recognition()

    # Wait for completion
    while not done:
        time.sleep(0.5)

    speech_recognizer.stop_continuous_recognition()

    # Check for errors
    if error_message:
        return {"error": error_message}

    if not all_scores:
        return {"error": "No speech could be recognized in the audio"}

    # Calculate average scores
    avg_scores = {}
    score_keys = ["accuracy", "fluency", "pronunciation", "prosody"]
    for key in score_keys:
        values = [s[key] for s in all_scores if key in s]
        if values:
            avg_scores[key] = sum(values) / len(values)

    return {
        "transcription": " ".join(all_text),
        "scores": avg_scores,
        "words": all_words,
        "segment_count": len(all_scores)
    }


def print_assessment(assessment: dict):
    """Pretty print the assessment results."""
    if "error" in assessment:
        print(f"\nError: {assessment['error']}")
        return

    print("\n" + "=" * 60)
    print("PRONUNCIATION ASSESSMENT RESULTS")
    print("=" * 60)

    print(f"\nTranscription:\n{assessment['transcription']}")

    print("\n--- Overall Scores (0-100) ---")
    scores = assessment["scores"]
    print(f"  Pronunciation: {scores.get('pronunciation', 'N/A'):.1f}")
    print(f"  Accuracy:      {scores.get('accuracy', 'N/A'):.1f}")
    print(f"  Fluency:       {scores.get('fluency', 'N/A'):.1f}")
    if "prosody" in scores:
        print(f"  Prosody:       {scores['prosody']:.1f}")

    print(f"\n--- Word-Level Analysis ({len(assessment['words'])} words) ---")

    # Show words with issues first
    problem_words = [w for w in assessment["words"] if w["accuracy_score"] < 80]
    if problem_words:
        print("\nWords needing improvement:")
        for word in problem_words:
            print(f"  '{word['word']}': {word['accuracy_score']:.1f}")

    # Summary statistics
    if assessment["words"]:
        word_scores = [w["accuracy_score"] for w in assessment["words"]]
        print(f"\nWord Statistics:")
        print(f"  Total words: {len(word_scores)}")
        print(f"  Average word accuracy: {sum(word_scores)/len(word_scores):.1f}")
        print(f"  Words below 80%: {len(problem_words)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Azure Speech credentials from environment
    SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python pronunciation_assessment.py <audio_file> [language]")
        print("\nExamples:")
        print("  python pronunciation_assessment.py audio_1.wav")
        print("  python pronunciation_assessment.py audio_1.wav en-US")
        print("  python pronunciation_assessment.py interview.wav en-GB")
        sys.exit(1)

    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "en-US"

    if not SPEECH_KEY:
        print("ERROR: AZURE_SPEECH_KEY not found in environment variables")
        print("\nTo set credentials:")
        print("1. Create a .env file in the project directory")
        print("2. Add: AZURE_SPEECH_KEY=your_key_here")
        print("3. Add: AZURE_SPEECH_REGION=eastus")
        sys.exit(1)

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found: {audio_file}")
        sys.exit(1)

    # Perform assessment
    result = assess_pronunciation_no_reference(
        audio_file=audio_file,
        speech_key=SPEECH_KEY,
        speech_region=SPEECH_REGION,
        language=language
    )

    # Display results
    print_assessment(result)

    # Save JSON output
    output_json = audio_file.rsplit(".", 1)[0] + "_assessment.json"
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_json}")
