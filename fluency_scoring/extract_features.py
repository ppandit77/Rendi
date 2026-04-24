#!/usr/bin/env python3
"""
Stage 1: Extract speech features from audio files.

This script extracts:
1. WavLM embeddings (1024-dim) - state-of-the-art speech representations
2. Whisper encoder features (optional)
3. Prosody features: pitch stats, jitter, shimmer, speaking rate

Usage:
    python extract_features.py [--limit N] [--model wavlm|whisper]
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path(__file__).parent / 'data'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'feature_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
MANIFEST_PATH = Path(__file__).parent / "data" / "download_manifest.csv"
LABELS_PATH = Path(__file__).parent / "data" / "labels.csv"
FEATURE_MANIFEST_PATH = Path(__file__).parent / "data" / "feature_manifest.csv"

# Feature extraction settings
SAMPLE_RATE = 16000  # Expected sample rate for WavLM/Whisper
MAX_AUDIO_SECONDS = 180  # Truncate longer audio


def load_audio(audio_path: Path, target_sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Load audio file and resample if needed."""
    try:
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

        # Truncate if too long
        max_samples = MAX_AUDIO_SECONDS * target_sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None


def extract_prosody_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract prosody features: pitch statistics, speaking rate, energy.

    Returns dict with ~20 features.
    """
    try:
        import librosa

        features = {}

        # Pitch (F0) analysis using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

        # Filter to voiced frames only
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            features['pitch_mean'] = float(np.mean(f0_voiced))
            features['pitch_std'] = float(np.std(f0_voiced))
            features['pitch_min'] = float(np.min(f0_voiced))
            features['pitch_max'] = float(np.max(f0_voiced))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = float(np.median(f0_voiced))

            # Pitch variability (coefficient of variation)
            features['pitch_cv'] = features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0

            # Voiced ratio (fluency indicator)
            features['voiced_ratio'] = len(f0_voiced) / len(f0)
        else:
            # No voiced frames detected
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
                       'pitch_range', 'pitch_median', 'pitch_cv', 'voiced_ratio']:
                features[key] = 0.0

        # Energy/intensity features
        rms = librosa.feature.rms(y=audio)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        features['energy_max'] = float(np.max(rms))

        # Zero crossing rate (correlates with fricatives/noise)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))

        # Speaking rate proxy: number of energy peaks per second
        # (crude but effective without forced alignment)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        # Count syllable-like energy peaks
        energy_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
        peaks = np.where((energy_smooth[1:-1] > energy_smooth[:-2]) &
                        (energy_smooth[1:-1] > energy_smooth[2:]))[0]
        duration_sec = len(audio) / sr
        features['syllable_rate'] = len(peaks) / duration_sec if duration_sec > 0 else 0

        # Audio duration
        features['duration_seconds'] = duration_sec

        return features

    except Exception as e:
        logger.warning(f"Prosody extraction failed: {e}")
        return {}


def extract_wavlm_embeddings(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Extract WavLM embeddings from audio.

    Returns mean-pooled 1024-dim embedding vector.
    """
    try:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        # Load model (cached after first call)
        if not hasattr(extract_wavlm_embeddings, 'model'):
            logger.info("Loading WavLM model...")
            extract_wavlm_embeddings.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus"
            )
            extract_wavlm_embeddings.model = WavLMModel.from_pretrained(
                "microsoft/wavlm-base-plus"
            )
            extract_wavlm_embeddings.model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                extract_wavlm_embeddings.model = extract_wavlm_embeddings.model.cuda()
                logger.info("WavLM running on GPU")
            else:
                logger.info("WavLM running on CPU")

        processor = extract_wavlm_embeddings.processor
        model = extract_wavlm_embeddings.model
        device = next(model.parameters()).device

        # Process audio
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Mean pool across time dimension
            # outputs.last_hidden_state: (batch, time, 768)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings

    except ImportError as e:
        logger.error(f"Missing dependency for WavLM: {e}")
        logger.error("Install with: pip install transformers torch")
        return None
    except Exception as e:
        logger.error(f"WavLM extraction failed: {e}")
        return None


def extract_whisper_embeddings(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Extract Whisper encoder embeddings from audio.

    Returns mean-pooled embedding vector.
    """
    try:
        import torch
        import whisper

        # Load model (cached after first call)
        if not hasattr(extract_whisper_embeddings, 'model'):
            logger.info("Loading Whisper model...")
            extract_whisper_embeddings.model = whisper.load_model("base")
            logger.info("Whisper loaded")

        model = extract_whisper_embeddings.model

        # Pad/trim to 30 seconds as Whisper expects
        audio_padded = whisper.pad_or_trim(audio)

        # Convert to log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

        # Get encoder output
        with torch.no_grad():
            # Add batch dimension
            mel = mel.unsqueeze(0)
            embeddings = model.encoder(mel)

            # Mean pool across time
            embeddings = embeddings.mean(dim=1).squeeze().cpu().numpy()

        return embeddings

    except ImportError:
        logger.warning("Whisper not installed. Install with: pip install openai-whisper")
        return None
    except Exception as e:
        logger.error(f"Whisper extraction failed: {e}")
        return None


def load_manifest() -> List[Dict]:
    """Load download manifest and labels."""
    records = []

    # Load labels for scores and emails
    labels_map = {}
    if LABELS_PATH.exists():
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels_map[row['record_id']] = {
                    'score': float(row['score']),
                    'email': row.get('email', ''),
                    'name': row.get('name', '')
                }

    # Load manifest
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        return []

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] in ('success', 'cached'):
                audio_path = Path(row['audio_path'])
                if audio_path.exists():
                    record_id = row['record_id']
                    label_info = labels_map.get(record_id, {})

                    records.append({
                        'record_id': record_id,
                        'audio_path': audio_path,
                        'score': label_info.get('score', 0),
                        'email': label_info.get('email', ''),
                        'name': label_info.get('name', ''),
                        'audio_duration': float(row.get('audio_duration_seconds', 0))
                    })

    logger.info(f"Loaded {len(records)} records with valid audio")
    return records


def process_record(
    record: Dict,
    model: str = "wavlm",
    save_individual: bool = True
) -> Dict:
    """
    Process a single record: extract all features.

    Returns feature manifest entry.
    """
    record_id = record['record_id']
    audio_path = record['audio_path']

    result = {
        'record_id': record_id,
        'score': record['score'],
        'email': record['email'],
        'status': 'unknown',
        'embedding_path': '',
        'prosody_features': '',
        'error': '',
        'timestamp': datetime.now().isoformat()
    }

    # Check if already processed
    embedding_path = FEATURES_DIR / f"{record_id}_embedding.npy"
    prosody_path = FEATURES_DIR / f"{record_id}_prosody.npy"

    if save_individual and embedding_path.exists() and prosody_path.exists():
        result['status'] = 'cached'
        result['embedding_path'] = str(embedding_path)
        result['prosody_features'] = str(prosody_path)
        logger.debug(f"[{record_id}] Using cached features")
        return result

    # Load audio
    audio = load_audio(audio_path)
    if audio is None:
        result['status'] = 'load_failed'
        result['error'] = 'Failed to load audio'
        return result

    # Extract prosody features
    prosody = extract_prosody_features(audio)

    # Extract embeddings
    if model == "wavlm":
        embeddings = extract_wavlm_embeddings(audio)
    elif model == "whisper":
        embeddings = extract_whisper_embeddings(audio)
    else:
        embeddings = extract_wavlm_embeddings(audio)

    if embeddings is None:
        result['status'] = 'embedding_failed'
        result['error'] = f'{model} embedding extraction failed'
        return result

    # Save features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if save_individual:
        np.save(embedding_path, embeddings)
        np.save(prosody_path, np.array(list(prosody.values())))
        result['embedding_path'] = str(embedding_path)
        result['prosody_features'] = str(prosody_path)

    result['status'] = 'success'
    result['embedding_dim'] = len(embeddings)
    result['prosody_dim'] = len(prosody)

    logger.info(f"[{record_id}] Extracted {model}:{len(embeddings)}d + prosody:{len(prosody)}d")

    return result


def save_combined_features(records: List[Dict], model: str = "wavlm"):
    """
    Save all features as combined numpy arrays for efficient training.

    Outputs:
    - X_embeddings.npy: (N, embedding_dim) matrix
    - X_prosody.npy: (N, prosody_dim) matrix
    - y_scores.npy: (N,) score vector
    - groups.npy: (N,) group IDs for GroupKFold
    - record_ids.npy: (N,) record IDs for reference
    """
    logger.info("Combining features into matrices...")

    embeddings_list = []
    prosody_list = []
    scores = []
    groups = []
    record_ids = []

    # Map emails to group IDs
    email_to_group = {}
    group_counter = 0

    for record in records:
        record_id = record['record_id']
        embedding_path = FEATURES_DIR / f"{record_id}_embedding.npy"
        prosody_path = FEATURES_DIR / f"{record_id}_prosody.npy"

        if not embedding_path.exists() or not prosody_path.exists():
            continue

        embeddings_list.append(np.load(embedding_path))
        prosody_list.append(np.load(prosody_path))
        scores.append(record['score'])
        record_ids.append(record_id)

        # Assign group ID based on email
        email = record.get('email', '')
        if email not in email_to_group:
            email_to_group[email] = group_counter
            group_counter += 1
        groups.append(email_to_group[email])

    if not embeddings_list:
        logger.error("No features to combine!")
        return

    # Stack into matrices
    X_embeddings = np.stack(embeddings_list)
    X_prosody = np.stack(prosody_list)
    y_scores = np.array(scores)
    groups = np.array(groups)
    record_ids = np.array(record_ids)

    # Normalize scores to [0, 1]
    y_normalized = y_scores / 100.0

    # Save
    np.save(FEATURES_DIR / "X_embeddings.npy", X_embeddings)
    np.save(FEATURES_DIR / "X_prosody.npy", X_prosody)
    np.save(FEATURES_DIR / "y_scores.npy", y_scores)
    np.save(FEATURES_DIR / "y_normalized.npy", y_normalized)
    np.save(FEATURES_DIR / "groups.npy", groups)
    np.save(FEATURES_DIR / "record_ids.npy", record_ids)

    logger.info(f"Saved combined features:")
    logger.info(f"  X_embeddings: {X_embeddings.shape}")
    logger.info(f"  X_prosody: {X_prosody.shape}")
    logger.info(f"  y_scores: {y_scores.shape} (range: {y_scores.min()}-{y_scores.max()})")
    logger.info(f"  groups: {len(email_to_group)} unique speakers")

    # Save feature info
    info = {
        'n_samples': len(scores),
        'embedding_dim': X_embeddings.shape[1],
        'prosody_dim': X_prosody.shape[1],
        'n_groups': len(email_to_group),
        'score_min': float(y_scores.min()),
        'score_max': float(y_scores.max()),
        'score_mean': float(y_scores.mean()),
        'model': model
    }

    import json
    with open(FEATURES_DIR / "feature_info.json", 'w') as f:
        json.dump(info, f, indent=2)


def print_summary(results: List[Dict]):
    """Print extraction summary."""
    total = len(results)
    by_status = {}
    for r in results:
        status = r['status']
        by_status[status] = by_status.get(status, 0) + 1

    success_count = by_status.get('success', 0) + by_status.get('cached', 0)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal records processed: {total}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(by_status.items()):
        pct = 100 * count / total
        print(f"  {status:<20}: {count:5d} ({pct:5.1f}%)")

    print(f"\nSuccess rate: {100 * success_count / total:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Extract speech features from audio')
    parser.add_argument('--limit', type=int, default=None, help='Max records to process')
    parser.add_argument('--model', type=str, default='wavlm',
                       choices=['wavlm', 'whisper'], help='Embedding model')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip cached features')
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 1: FEATURE EXTRACTION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit or 'all'}")
    print("=" * 60)

    # Ensure directories exist
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load records
    records = load_manifest()

    if args.limit:
        records = records[:args.limit]

    if not records:
        logger.error("No records to process!")
        sys.exit(1)

    print(f"\nProcessing {len(records)} records...")

    # Process records
    results = []
    start_time = time.time()

    for i, record in enumerate(records):
        result = process_record(record, model=args.model)
        results.append(result)

        # Progress update
        if (i + 1) % 25 == 0 or i + 1 == len(records):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            success = sum(1 for r in results if r['status'] in ('success', 'cached'))
            print(f"  Progress: {i+1}/{len(records)} ({100*(i+1)/len(records):.1f}%) - {rate:.2f}/sec - {success} success")

    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Combine features
    save_combined_features(records, model=args.model)

    # Save feature manifest
    with open(FEATURE_MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['record_id', 'score', 'email', 'status', 'embedding_path',
                     'prosody_features', 'error', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved feature manifest to {FEATURE_MANIFEST_PATH}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
