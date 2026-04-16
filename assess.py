#!/usr/bin/env python3
"""
Single File Pronunciation Assessment CLI.

Assess pronunciation from a single audio file using OpenAI GPT-4o or Azure Speech.

Usage:
    python assess.py <audio_file> [--provider openai|azure] [--language en-US]

Examples:
    python assess.py audio_1.wav
    python assess.py audio_1.wav --provider azure
    python assess.py interview.wav --language en-GB
"""

import argparse
import json
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.assessment.openai_assessment import assess_pronunciation_openai, print_assessment as print_openai
from src.assessment.azure_assessment import assess_pronunciation_azure, print_assessment as print_azure
from src.utils.logging_utils import log_error_result, setup_logging


logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Pronunciation Assessment CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('audio_file', help='Path to audio file (WAV recommended)')
    parser.add_argument('--provider', choices=['openai', 'azure'], default='openai',
                        help='Assessment provider (default: openai)')
    parser.add_argument('--language', default='en-US',
                        help='Language code (default: en-US)')
    parser.add_argument('--output', '-o', help='Output JSON file path')

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: Audio file not found: {args.audio_file}")
        sys.exit(1)

    try:
        # Run assessment
        print(f"Assessing pronunciation using {args.provider.upper()}...")
        print(f"Audio file: {args.audio_file}")
        print(f"Language: {args.language}")
        print("-" * 50)

        if args.provider == 'openai':
            result = assess_pronunciation_openai(args.audio_file, args.language)
            print_openai(result)
        else:
            result = assess_pronunciation_azure(args.audio_file, args.language)
            print_azure(result)

        if "error" in result:
            log_error_result(
                logger,
                "Assessment failed",
                {
                    "error": result.get("error", "Unknown error"),
                    "error_stage": result.get("error_stage", "assessment"),
                    "error_type": result.get("error_type", "AssessmentError"),
                    "error_context": result.get("error_context", {"audio_file": args.audio_file, "provider": args.provider}),
                },
            )

        # Save JSON output
        output_file = args.output or args.audio_file.rsplit(".", 1)[0] + "_assessment.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception:
        logger.exception("Unhandled error while assessing %s with provider %s", args.audio_file, args.provider)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Assess CLI crashed")
        raise
