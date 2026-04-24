"""
DSPy Prompt Optimization for Pronunciation Assessment.

Uses DSPy to optimize the pronunciation assessment prompt for better
score differentiation and correlation with existing baseline scores.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import dspy
from pydantic import BaseModel, Field

from ..config import OPENAI_API_KEY


logger = logging.getLogger(__name__)


# Pydantic models for structured output
class WordAssessment(BaseModel):
    """Assessment of a single word's pronunciation."""
    word: str = Field(description="The mispronounced word")
    accuracy_score: int = Field(ge=0, le=100, description="Accuracy score 0-100")
    error_type: str = Field(default="Mispronunciation")


class PronunciationScores(BaseModel):
    """Scores for each pronunciation dimension."""
    accuracy: int = Field(ge=0, le=100, description="Phoneme and word accuracy score")
    fluency: int = Field(ge=0, le=100, description="Speech flow and naturalness score")
    pronunciation: int = Field(ge=0, le=100, description="Overall pronunciation clarity score")
    prosody: int = Field(ge=0, le=100, description="Rhythm, stress, and intonation score")


class PronunciationAssessment(BaseModel):
    """Complete pronunciation assessment result."""
    transcription: str = Field(description="Full transcription of the spoken audio")
    scores: PronunciationScores
    words: list[WordAssessment] = Field(default_factory=list, description="Words with pronunciation issues")
    assessment_notes: str = Field(description="Detailed notes about pronunciation patterns")


# DSPy Signature for pronunciation assessment
class AssessPronunciation(dspy.Signature):
    """Assess the pronunciation quality of a non-native English speaker from their speech transcription.

    You are a STRICT pronunciation assessor. Your scores should differentiate between speakers
    and use the full 0-100 scale. Most non-native speakers score between 50-75.

    Score Distribution Guidelines:
    - 90-100: Exceptional (top 5%) - Near-native, broadcast quality
    - 80-89: Very Good (top 20%) - Professional level, minimal accent
    - 70-79: Good (middle 40%) - Clear but noticeable accent
    - 60-69: Fair (lower 25%) - Understandable with effort, frequent errors
    - 50-59: Needs Work (bottom 10%) - Difficult to understand
    - Below 50: Poor - Significant communication barriers

    Deductions for common issues:
    - Heavy accent: -15 to -25 on pronunciation
    - Hesitations/fillers: -10 to -20 on fluency
    - Monotone delivery: -15 to -25 on prosody
    - Mispronounced words: -3 to -5 each on accuracy
    """

    transcription: str = dspy.InputField(desc="The transcribed speech from the audio recording")
    target_language: str = dspy.InputField(desc="The target language being assessed (e.g., en-US)")

    accuracy_score: int = dspy.OutputField(desc="Phoneme accuracy score 0-100. Penalize substitutions, omissions, wrong stress.")
    fluency_score: int = dspy.OutputField(desc="Speech flow score 0-100. Penalize hesitations, fillers, unnatural pace.")
    pronunciation_score: int = dspy.OutputField(desc="Overall clarity score 0-100. Penalize heavy accents, unclear speech.")
    prosody_score: int = dspy.OutputField(desc="Rhythm and intonation score 0-100. Penalize monotone, wrong stress patterns.")
    problematic_words: str = dspy.OutputField(desc="Comma-separated list of mispronounced words, or 'none' if all clear")
    assessment_notes: str = dspy.OutputField(desc="Brief notes on accent strength, speech patterns, areas for improvement")


class PronunciationAssessor(dspy.Module):
    """DSPy module for pronunciation assessment with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(AssessPronunciation)

    def forward(self, transcription: str, target_language: str = "en-US"):
        """Assess pronunciation from transcription."""
        result = self.assess(transcription=transcription, target_language=target_language)
        return result


def compute_assessment_metric(example, prediction, trace=None) -> float:
    """
    Metric function for DSPy optimization.

    Evaluates how well the predicted scores correlate with the expected baseline score
    and whether the scores use the full range (not clustering at 75-85).

    Args:
        example: DSPy Example with expected scores
        prediction: Model prediction
        trace: Optimization trace (unused)

    Returns:
        Score between 0 and 1 indicating assessment quality
    """
    try:
        # Parse predicted scores
        pred_accuracy = int(prediction.accuracy_score)
        pred_fluency = int(prediction.fluency_score)
        pred_pronunciation = int(prediction.pronunciation_score)
        pred_prosody = int(prediction.prosody_score)

        pred_final = (pred_accuracy + pred_fluency + pred_pronunciation + pred_prosody) / 4

        # Get expected baseline score
        expected_score = float(example.expected_score)

        # Component 1: Score alignment with baseline (0-0.5 points)
        # We want the predicted score to be closer to the baseline than the old system
        score_diff = abs(pred_final - expected_score)
        alignment_score = max(0, 0.5 - (score_diff / 100))

        # Component 2: Score differentiation (0-0.3 points)
        # Reward scores that are NOT in the 70-85 cluster range
        # This encourages the model to use the full scale
        if pred_final < 65 or pred_final > 85:
            differentiation_score = 0.3  # Full points for using extremes
        elif pred_final < 70 or pred_final > 80:
            differentiation_score = 0.2  # Partial points
        else:
            differentiation_score = 0.1  # Minimal points for clustering in 70-80

        # Component 3: Reasonable scoring (0-0.2 points)
        # Penalize extreme outliers
        all_scores = [pred_accuracy, pred_fluency, pred_pronunciation, pred_prosody]
        if all(30 <= s <= 95 for s in all_scores):
            reasonable_score = 0.2
        elif all(20 <= s <= 100 for s in all_scores):
            reasonable_score = 0.1
        else:
            reasonable_score = 0.0

        total_score = alignment_score + differentiation_score + reasonable_score
        return total_score

    except (ValueError, AttributeError) as e:
        logger.warning("Failed to compute metric: %s", e)
        return 0.0


def load_training_data(results_dir: str = "data/batch_assessment/results") -> list[dspy.Example]:
    """
    Load training examples from batch assessment results.

    Each example contains:
    - transcription: The spoken text
    - expected_score: The baseline/existing score to align with
    - openai_score: The previous OpenAI score (for comparison)

    Args:
        results_dir: Path to the results directory

    Returns:
        List of DSPy Examples for training
    """
    results_path = Path(results_dir)
    examples = []

    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Skip failed assessments
            if "assessment" not in data or "error" in data.get("assessment", {}):
                continue

            transcription = data["assessment"].get("transcription", "")
            existing_score = data.get("existing_score")

            # Skip if missing required data
            if not transcription or existing_score is None:
                continue

            example = dspy.Example(
                transcription=transcription,
                target_language="en-US",
                expected_score=existing_score,
                # Store original scores for analysis
                original_openai_score=data.get("openai_final_score"),
                name=data.get("name", json_file.stem)
            ).with_inputs("transcription", "target_language")

            examples.append(example)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load %s: %s", json_file, e)
            continue

    logger.info("Loaded %d training examples", len(examples))
    return examples


def optimize_prompt(
    trainset: list[dspy.Example],
    auto: Optional[str] = "medium",
    save_path: Optional[str] = None
) -> PronunciationAssessor:
    """
    Optimize the pronunciation assessment prompt using DSPy MIPROv2.

    Args:
        trainset: List of training examples
        auto: Auto configuration level ("light", "medium", "heavy") or None for manual
        save_path: Path to save the optimized program

    Returns:
        Optimized PronunciationAssessor module
    """
    # Configure DSPy with OpenAI
    lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
    dspy.configure(lm=lm)

    # Create base program
    program = PronunciationAssessor()

    # Configure optimizer - when using auto, don't set num_candidates/num_trials
    optimizer = dspy.MIPROv2(
        metric=compute_assessment_metric,
        auto=auto
    )

    # Run optimization
    logger.info("Starting prompt optimization with %d examples (auto=%s)...", len(trainset), auto)
    optimized_program = optimizer.compile(
        program,
        trainset=trainset,
    )

    # Save if path provided
    if save_path:
        optimized_program.save(save_path)
        logger.info("Saved optimized program to %s", save_path)

    return optimized_program


def load_optimized_program(load_path: str) -> PronunciationAssessor:
    """
    Load a previously optimized program.

    Args:
        load_path: Path to the saved program

    Returns:
        Loaded PronunciationAssessor module
    """
    program = PronunciationAssessor()
    program.load(path=load_path)
    return program


def run_optimization_pipeline(
    results_dir: str = "data/batch_assessment/results",
    output_path: str = "data/optimized_prompt.json",
    test_split: float = 0.2
):
    """
    Run the full optimization pipeline.

    Args:
        results_dir: Directory containing assessment results
        output_path: Path to save the optimized program
        test_split: Fraction of data to hold out for testing
    """
    # Load data
    examples = load_training_data(results_dir)

    if len(examples) < 10:
        raise ValueError(f"Need at least 10 examples, got {len(examples)}")

    # Split into train/test
    split_idx = int(len(examples) * (1 - test_split))
    trainset = examples[:split_idx]
    testset = examples[split_idx:]

    logger.info("Training with %d examples, testing with %d", len(trainset), len(testset))

    # Configure DSPy
    lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
    dspy.configure(lm=lm)

    # Optimize
    optimized = optimize_prompt(trainset, save_path=output_path)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=compute_assessment_metric,
        num_threads=4,
        display_progress=True
    )

    test_score = evaluator(optimized)
    logger.info("Test set score: %.3f", test_score)

    return optimized, test_score
