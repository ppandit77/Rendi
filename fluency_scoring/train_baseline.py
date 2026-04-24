#!/usr/bin/env python3
"""
Stage 2: Train baseline models for fluency scoring.

This script:
1. Loads extracted features (WavLM embeddings + prosody)
2. Trains Ridge and XGBoost regressors
3. Runs cross-validation with both KFold and GroupKFold
4. Reports Spearman (primary), Pearson, MAE, and per-decade MAE

Usage:
    python train_baseline.py [--n-folds N] [--model ridge|xgboost|both]
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

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
        logging.FileHandler(log_dir / 'training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
FEATURES_DIR = Path(__file__).parent / "data" / "features"
MODELS_DIR = Path(__file__).parent / "data" / "models"
RESULTS_DIR = Path(__file__).parent / "data" / "results"

# Decade score buckets for per-decade MAE
DECADE_BUCKETS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load combined features from disk.

    Returns:
        X: Feature matrix (concatenated embeddings + prosody)
        y: Scores normalized to [0, 1]
        y_raw: Raw scores (0-100)
        groups: Group IDs for GroupKFold
    """
    X_embeddings = np.load(FEATURES_DIR / "X_embeddings.npy")
    X_prosody = np.load(FEATURES_DIR / "X_prosody.npy")
    y_normalized = np.load(FEATURES_DIR / "y_normalized.npy")
    y_raw = np.load(FEATURES_DIR / "y_scores.npy")
    groups = np.load(FEATURES_DIR / "groups.npy")

    # Concatenate embeddings and prosody
    X = np.concatenate([X_embeddings, X_prosody], axis=1)

    logger.info(f"Loaded features: X={X.shape}, y={y_normalized.shape}, groups={len(np.unique(groups))} unique")

    return X, y_normalized, y_raw, groups


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_normalized: bool = True) -> Dict:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth (normalized or raw)
        y_pred: Predictions (same scale as y_true)
        is_normalized: Whether scores are in [0,1] range

    Returns:
        Dict with spearman, pearson, mae, rmse
    """
    # Convert to raw scores for interpretable metrics
    if is_normalized:
        y_true_raw = y_true * 100
        y_pred_raw = y_pred * 100
    else:
        y_true_raw = y_true
        y_pred_raw = y_pred

    # Clip predictions to valid range
    y_pred_raw = np.clip(y_pred_raw, 0, 100)

    # Correlation metrics (computed on raw scores)
    spearman_r, spearman_p = spearmanr(y_true_raw, y_pred_raw)
    pearson_r, pearson_p = pearsonr(y_true_raw, y_pred_raw)

    # Error metrics (on raw scores)
    mae = np.mean(np.abs(y_true_raw - y_pred_raw))
    rmse = np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2))

    # Per-decade MAE (to catch mean-prediction failure mode)
    decade_mae = {}
    for decade in DECADE_BUCKETS:
        mask = y_true_raw == decade
        if mask.sum() > 0:
            decade_mae[decade] = float(np.mean(np.abs(y_true_raw[mask] - y_pred_raw[mask])))

    return {
        'spearman': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson': float(pearson_r),
        'pearson_p': float(pearson_p),
        'mae': float(mae),
        'rmse': float(rmse),
        'decade_mae': decade_mae
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
    model_type: str = "ridge",
    n_folds: int = 5
) -> Tuple[Dict, Dict]:
    """
    Run cross-validation with both KFold and GroupKFold.

    Args:
        X: Feature matrix
        y: Normalized scores [0,1]
        y_raw: Raw scores [0,100]
        groups: Group IDs
        model_type: "ridge" or "xgboost"
        n_folds: Number of CV folds

    Returns:
        Tuple of (kfold_results, groupkfold_results)
    """
    from sklearn.model_selection import KFold, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    # Try to import XGBoost
    try:
        from xgboost import XGBRegressor
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        if model_type == "xgboost":
            logger.warning("XGBoost not installed, falling back to Ridge")
            model_type = "ridge"

    def get_model():
        if model_type == "ridge":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ])
        elif model_type == "xgboost" and has_xgboost:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                ))
            ])
        else:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ])

    results = {'kfold': [], 'groupkfold': []}

    # Standard KFold
    logger.info(f"Running {n_folds}-fold KFold CV with {model_type}...")
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_y_true_kf = []
    all_y_pred_kf = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        model = get_model()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        all_y_true_kf.extend(y_raw[test_idx])
        all_y_pred_kf.extend(y_pred * 100)

        fold_metrics = compute_metrics(y[test_idx], y_pred)
        fold_metrics['fold'] = fold
        results['kfold'].append(fold_metrics)

        logger.debug(f"  KFold {fold+1}: Spearman={fold_metrics['spearman']:.3f}, MAE={fold_metrics['mae']:.1f}")

    # Compute overall KFold metrics
    kfold_overall = compute_metrics(
        np.array(all_y_true_kf),
        np.array(all_y_pred_kf),
        is_normalized=False
    )

    # GroupKFold (prevent speaker leakage)
    logger.info(f"Running {n_folds}-fold GroupKFold CV with {model_type}...")

    # Ensure we have enough groups
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)

    if actual_folds < n_folds:
        logger.warning(f"Only {n_groups} unique groups, using {actual_folds} folds for GroupKFold")

    groupkfold = GroupKFold(n_splits=actual_folds)

    all_y_true_gkf = []
    all_y_pred_gkf = []

    for fold, (train_idx, test_idx) in enumerate(groupkfold.split(X, y, groups)):
        model = get_model()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        all_y_true_gkf.extend(y_raw[test_idx])
        all_y_pred_gkf.extend(y_pred * 100)

        fold_metrics = compute_metrics(y[test_idx], y_pred)
        fold_metrics['fold'] = fold
        fold_metrics['n_train_groups'] = len(np.unique(groups[train_idx]))
        fold_metrics['n_test_groups'] = len(np.unique(groups[test_idx]))
        results['groupkfold'].append(fold_metrics)

        logger.debug(f"  GroupKFold {fold+1}: Spearman={fold_metrics['spearman']:.3f}, MAE={fold_metrics['mae']:.1f}")

    # Compute overall GroupKFold metrics
    groupkfold_overall = compute_metrics(
        np.array(all_y_true_gkf),
        np.array(all_y_pred_gkf),
        is_normalized=False
    )

    return {
        'model_type': model_type,
        'n_folds': n_folds,
        'kfold': {
            'folds': results['kfold'],
            'overall': kfold_overall,
            'mean_spearman': np.mean([r['spearman'] for r in results['kfold']]),
            'std_spearman': np.std([r['spearman'] for r in results['kfold']]),
            'mean_mae': np.mean([r['mae'] for r in results['kfold']]),
            'std_mae': np.std([r['mae'] for r in results['kfold']]),
        },
        'groupkfold': {
            'folds': results['groupkfold'],
            'overall': groupkfold_overall,
            'mean_spearman': np.mean([r['spearman'] for r in results['groupkfold']]),
            'std_spearman': np.std([r['spearman'] for r in results['groupkfold']]),
            'mean_mae': np.mean([r['mae'] for r in results['groupkfold']]),
            'std_mae': np.std([r['mae'] for r in results['groupkfold']]),
            'n_groups': n_groups
        }
    }


def print_results(results: Dict, model_type: str):
    """Print formatted results."""
    kf = results['kfold']
    gkf = results['groupkfold']

    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATION RESULTS: {model_type.upper()}")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'KFold':>20} {'GroupKFold':>20}")
    print("-" * 70)

    # Primary metric: Spearman
    print(f"{'Spearman (primary)':<25} {kf['overall']['spearman']:>15.3f}     {gkf['overall']['spearman']:>15.3f}")
    print(f"{'  (mean ± std)':<25} {kf['mean_spearman']:>7.3f} ± {kf['std_spearman']:<7.3f} {gkf['mean_spearman']:>7.3f} ± {gkf['std_spearman']:<7.3f}")

    # Secondary metrics
    print(f"\n{'Pearson':<25} {kf['overall']['pearson']:>15.3f}     {gkf['overall']['pearson']:>15.3f}")
    print(f"{'MAE':<25} {kf['overall']['mae']:>15.1f}     {gkf['overall']['mae']:>15.1f}")
    print(f"{'  (mean ± std)':<25} {kf['mean_mae']:>7.1f} ± {kf['std_mae']:<7.1f} {gkf['mean_mae']:>7.1f} ± {gkf['std_mae']:<7.1f}")
    print(f"{'RMSE':<25} {kf['overall']['rmse']:>15.1f}     {gkf['overall']['rmse']:>15.1f}")

    # Gap analysis (important for speaker leakage detection)
    spearman_gap = kf['overall']['spearman'] - gkf['overall']['spearman']
    mae_gap = gkf['overall']['mae'] - kf['overall']['mae']

    print("\n" + "-" * 70)
    print("LEAKAGE ANALYSIS (KFold - GroupKFold gap)")
    print("-" * 70)
    print(f"{'Spearman gap:':<25} {spearman_gap:>+.3f}", end="")
    if spearman_gap > 0.05:
        print("  ⚠️  WARNING: Possible speaker leakage!")
    else:
        print("  ✓ Acceptable")

    print(f"{'MAE gap:':<25} {mae_gap:>+.1f}", end="")
    if mae_gap > 2:
        print("  ⚠️  WARNING: GroupKFold MAE significantly higher")
    else:
        print("  ✓ Acceptable")

    # Per-decade MAE (to catch mean-prediction failure)
    print("\n" + "-" * 70)
    print("PER-DECADE MAE (GroupKFold)")
    print("-" * 70)

    decade_mae_gkf = gkf['overall'].get('decade_mae', {})
    for decade in DECADE_BUCKETS:
        if decade in decade_mae_gkf:
            mae = decade_mae_gkf[decade]
            warning = " ⚠️ High" if mae > 15 else ""
            print(f"  Score {decade:>3}: MAE = {mae:>6.1f}{warning}")

    print("\n" + "=" * 70)

    # Interpretation
    spearman = gkf['overall']['spearman']
    if spearman > 0.7:
        interp = "EXCELLENT: Strong correlation with human scores"
    elif spearman > 0.5:
        interp = "GOOD: Moderate correlation with human scores"
    elif spearman > 0.3:
        interp = "FAIR: Weak but positive correlation"
    else:
        interp = "POOR: Model struggles to capture fluency patterns"

    print(f"\nInterpretation: {interp}")
    print(f"GroupKFold Spearman (primary metric): {spearman:.3f}")
    print("=" * 70)


def save_results(results: Dict, model_type: str):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f"cv_results_{model_type}_{timestamp}.json"

    # Add metadata
    results['timestamp'] = datetime.now().isoformat()
    results['model_type'] = model_type

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Also save latest results
    latest_path = RESULTS_DIR / f"cv_results_{model_type}_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train baseline fluency scoring models')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--model', type=str, default='both',
                       choices=['ridge', 'xgboost', 'both'], help='Model type to train')
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 2: BASELINE MODEL TRAINING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {args.model}")
    print(f"CV Folds: {args.n_folds}")
    print("=" * 70)

    # Load features
    try:
        X, y, y_raw, groups = load_features()
    except FileNotFoundError as e:
        logger.error(f"Features not found: {e}")
        logger.error("Run extract_features.py first!")
        sys.exit(1)

    # Check data
    logger.info(f"Data summary:")
    logger.info(f"  Samples: {len(y)}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Groups: {len(np.unique(groups))}")
    logger.info(f"  Score range: {y_raw.min():.0f} - {y_raw.max():.0f}")
    logger.info(f"  Score mean: {y_raw.mean():.1f} ± {y_raw.std():.1f}")

    # Score distribution
    print("\nScore distribution:")
    for decade in DECADE_BUCKETS:
        count = np.sum(y_raw == decade)
        if count > 0:
            print(f"  Score {decade:>3}: {count:>4} samples ({100*count/len(y_raw):.1f}%)")

    # Train models
    models_to_train = ['ridge', 'xgboost'] if args.model == 'both' else [args.model]

    all_results = {}

    for model_type in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}...")
        print(f"{'='*70}")

        results = cross_validate(X, y, y_raw, groups, model_type=model_type, n_folds=args.n_folds)

        print_results(results, model_type)
        save_results(results, model_type)

        all_results[model_type] = results

    # Final summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Model':<15} {'GKF Spearman':>15} {'GKF MAE':>10} {'Leakage Gap':>12}")
        print("-" * 55)

        for model_type, results in all_results.items():
            gkf_spearman = results['groupkfold']['overall']['spearman']
            gkf_mae = results['groupkfold']['overall']['mae']
            kf_spearman = results['kfold']['overall']['spearman']
            gap = kf_spearman - gkf_spearman

            print(f"{model_type:<15} {gkf_spearman:>15.3f} {gkf_mae:>10.1f} {gap:>+12.3f}")

        print("=" * 70)


if __name__ == "__main__":
    main()
