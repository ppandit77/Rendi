"""
Azure AI Speech Pronunciation Assessment Module.

Uses Azure's Speech SDK for pronunciation assessment.
"""

import time
import azure.cognitiveservices.speech as speechsdk

from ..config import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION


def assess_pronunciation_azure(
    audio_file: str,
    language: str = "en-US",
    speech_key: str = None,
    speech_region: str = None
) -> dict:
    """
    Perform pronunciation assessment using Azure Speech SDK.

    Args:
        audio_file: Path to the audio file (WAV format, 16kHz mono recommended)
        language: Language code (e.g., "en-US", "en-GB", "es-ES")
        speech_key: Azure Speech API key (optional, uses env var if not provided)
        speech_region: Azure Speech region (optional, uses env var if not provided)

    Returns:
        Dictionary containing pronunciation assessment results
    """
    key = speech_key or AZURE_SPEECH_KEY
    region = speech_region or AZURE_SPEECH_REGION

    if not key:
        return {"error": "Azure Speech API key not configured"}

    # Configure speech service
    speech_config = speechsdk.SpeechConfig(
        subscription=key,
        region=region
    )
    speech_config.speech_recognition_language = language

    # Configure audio input from file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Configure pronunciation assessment WITHOUT reference text
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
    )

    # Enable prosody assessment
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

    # Start continuous recognition
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

    # Calculate final score
    valid_scores = list(avg_scores.values())
    final_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "transcription": " ".join(all_text),
        "scores": avg_scores,
        "words": all_words,
        "final_score": final_score,
        "segment_count": len(all_scores)
    }


def print_assessment(assessment: dict):
    """Pretty print the assessment results."""
    if "error" in assessment:
        print(f"\nError: {assessment['error']}")
        return

    print("\n" + "=" * 60)
    print("PRONUNCIATION ASSESSMENT RESULTS (Azure Speech)")
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

    if "final_score" in assessment and assessment["final_score"]:
        print("\n" + "=" * 60)
        print(f"  FINAL SCORE: {assessment['final_score']:.1f}/100")
        print("=" * 60)
