#!/usr/bin/env python3
"""
DSPy Prompt Optimization CLI.

Optimizes the pronunciation assessment prompt using existing labeled examples
to improve score differentiation and correlation with baseline scores.

Usage:
    python optimize_prompt.py                    # Run full optimization
    python optimize_prompt.py --test-only        # Only test current prompt
    python optimize_prompt.py --quick            # Quick optimization (fewer candidates)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import dspy

from src.assessment.dspy_optimization import (
    PronunciationAssessor,
    compute_assessment_metric,
    load_training_data,
    optimize_prompt,
    load_optimized_program,
)
from src.config import OPENAI_API_KEY


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_baseline(testset: list[dspy.Example]) -> dict:
    """Evaluate the baseline (unoptimized) program."""
    lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
    dspy.configure(lm=lm)

    program = PronunciationAssessor()

    logger.info("Evaluating baseline on %d examples...", len(testset))

    scores = []
    predictions = []

    for i, example in enumerate(testset):
        try:
            pred = program(
                transcription=example.transcription,
                target_language=example.target_language
            )
            score = compute_assessment_metric(example, pred)
            scores.append(score)

            # Calculate predicted final score
            pred_final = (
                int(pred.accuracy_score) +
                int(pred.fluency_score) +
                int(pred.pronunciation_score) +
                int(pred.prosody_score)
            ) / 4

            predictions.append({
                "name": getattr(example, "name", f"example_{i}"),
                "expected": example.expected_score,
                "predicted": pred_final,
                "metric_score": score
            })

            logger.info(
                "[%d/%d] %s: expected=%.1f, predicted=%.1f, metric=%.3f",
                i + 1, len(testset),
                predictions[-1]["name"],
                example.expected_score,
                pred_final,
                score
            )

        except Exception as e:
            logger.error("Failed on example %d: %s", i, e)
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    logger.info("Baseline average metric score: %.3f", avg_score)

    return {
        "avg_score": avg_score,
        "predictions": predictions,
        "scores": scores
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize pronunciation assessment prompt")
    parser.add_argument(
        "--results-dir",
        default="data/batch_assessment/results",
        help="Directory containing assessment results"
    )
    parser.add_argument(
        "--output",
        default="data/optimized_prompt.json",
        help="Path to save optimized program"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only evaluate baseline, don't optimize"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick optimization with fewer candidates"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load and evaluate a previously optimized program"
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Load training data
    logger.info("Loading training data from %s...", args.results_dir)
    examples = load_training_data(args.results_dir)

    if len(examples) < 5:
        logger.error("Need at least 5 examples, got %d", len(examples))
        sys.exit(1)

    logger.info("Loaded %d examples", len(examples))

    # Split data
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    testset = examples[split_idx:]

    logger.info("Train: %d, Test: %d", len(trainset), len(testset))

    # Configure DSPy
    lm = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
    dspy.configure(lm=lm)

    if args.load:
        # Load and evaluate existing optimized program
        logger.info("Loading optimized program from %s...", args.load)
        optimized = load_optimized_program(args.load)

        evaluator = dspy.Evaluate(
            devset=testset,
            metric=compute_assessment_metric,
            num_threads=4,
            display_progress=True
        )
        eval_result = evaluator(optimized)
        test_score = eval_result.score if hasattr(eval_result, 'score') else float(eval_result)
        logger.info("Optimized program test score: %.3f", test_score)

    elif args.test_only:
        # Just evaluate baseline
        results = evaluate_baseline(testset)
        print(f"\nBaseline Results:")
        print(f"  Average Metric Score: {results['avg_score']:.3f}")
        print(f"\nPredictions:")
        for p in results["predictions"]:
            diff = p["predicted"] - p["expected"]
            print(f"  {p['name']}: expected={p['expected']:.0f}, predicted={p['predicted']:.1f} (diff={diff:+.1f})")

    else:
        # Run optimization
        logger.info("Starting optimization...")

        # Evaluate baseline first
        logger.info("Evaluating baseline...")
        baseline_results = evaluate_baseline(testset[:5])  # Quick baseline check

        # Optimize
        auto_level = "light" if args.quick else "medium"
        optimized = optimize_prompt(
            trainset,
            auto=auto_level,
            save_path=args.output
        )

        # Evaluate optimized
        logger.info("Evaluating optimized program...")
        evaluator = dspy.Evaluate(
            devset=testset,
            metric=compute_assessment_metric,
            num_threads=4,
            display_progress=True
        )
        eval_result = evaluator(optimized)

        # Handle both EvaluationResult object and float
        if hasattr(eval_result, 'score'):
            optimized_score = eval_result.score
        else:
            optimized_score = float(eval_result)

        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Baseline Score:  {baseline_results['avg_score']:.3f}")
        print(f"Optimized Score: {optimized_score:.3f}")
        print(f"Improvement:     {optimized_score - baseline_results['avg_score']:+.3f}")
        print(f"\nOptimized program saved to: {args.output}")
        print(f"{'='*60}")

        # Print the optimized prompt
        print("\nOptimized Prompt Configuration:")
        print(json.dumps(optimized.dump_state(), indent=2))


if __name__ == "__main__":
    main()
