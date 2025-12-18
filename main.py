"""
Main execution script for Sleep Apnea Detection
Trains and evaluates both ML and DL models
Supports Optuna hyperparameter tuning and ensemble methods
"""
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    RANDOM_STATE, DEVICE, EPOCHS, MODELS_DIR, RESULTS_DIR,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, ENSEMBLE_TOP_K, ENSEMBLE_METHOD
)
from src.data_loader import load_training_data, load_test_data, get_dataset_info
from src.preprocessing import preprocess_batch
from src.feature_extraction import extract_features_batch
from src.ml_models import get_all_ml_models, train_and_evaluate_all_models
from src.dl_models import get_all_dl_models, train_and_evaluate_all_dl_models, compute_class_weights
from src.train_utils import (
    subject_wise_split, plot_confusion_matrix, plot_roc_curve,
    plot_training_history, save_results, print_results_table,
    ExperimentLogger
)
from src.hyperparameter_tuning import tune_all_ml_models, tune_all_dl_models
from src.ensemble import create_ensemble_from_results, evaluate_ensemble_methods


def set_random_seeds(seed: int = RANDOM_STATE):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def run_ml_pipeline(X_train_features: np.ndarray, y_train: np.ndarray,
                    X_test_features: np.ndarray, y_test: np.ndarray,
                    train_record_ids, logger: ExperimentLogger,
                    tuned_params=None) -> dict:
    """
    Run ML pipeline with feature-based classification
    
    Args:
        tuned_params: Optional dictionary of tuned hyperparameters per model
    """
    logger.log("Starting ML Pipeline...")
    logger.log(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Convert record_ids to group array for CV
    unique_records = list(set(train_record_ids))
    record_to_idx = {r: i for i, r in enumerate(unique_records)}
    groups = np.array([record_to_idx[r] for r in train_record_ids])
    
    # Train and evaluate all ML models
    ml_results = train_and_evaluate_all_models(
        X_train_features, y_train,
        X_test_features, y_test,
        groups_train=groups,
        verbose=True,
        tuned_params=tuned_params
    )
    
    # Log results
    for name, result in ml_results.items():
        logger.log_metrics(name, result['test_metrics'], 'test')
    
    return ml_results


def run_dl_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    epochs: int, logger: ExperimentLogger,
                    tuned_params=None) -> dict:
    """
    Run DL pipeline with raw ECG segments
    
    Args:
        tuned_params: Optional dictionary of tuned hyperparameters per model
    """
    logger.log("Starting DL Pipeline...")
    logger.log(f"Training: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    logger.log(f"Using device: {DEVICE}")
    
    # Train and evaluate all DL models
    dl_results = train_and_evaluate_all_dl_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        epochs=epochs,
        verbose=True,
        tuned_params=tuned_params
    )
    
    # Log results
    for name, result in dl_results.items():
        logger.log_metrics(name, result['test_metrics'], 'test')
    
    return dl_results


def main(run_ml: bool = True, run_dl: bool = True, epochs: int = EPOCHS,
         use_test_set: bool = False, quick_test: bool = False,
         tune: bool = False, n_trials: int = OPTUNA_N_TRIALS,
         ensemble: bool = True, top_k: int = ENSEMBLE_TOP_K):
    """
    Main function to run the complete sleep apnea detection pipeline
    
    Args:
        run_ml: Whether to run ML models
        run_dl: Whether to run DL models
        epochs: Number of epochs for DL training
        use_test_set: Whether to use separate test set (x01-x35)
        quick_test: Run quick test with limited data
        tune: Whether to run Optuna hyperparameter tuning
        n_trials: Number of Optuna trials per model
        ensemble: Whether to create ensemble from best models
        top_k: Number of top models to include in ensemble
    """
    set_random_seeds()
    
    # Initialize logger
    logger = ExperimentLogger()
    logger.log("="*60)
    logger.log("Sleep Apnea Detection - Experiment Started")
    logger.log("="*60)
    
    # Print dataset info
    info = get_dataset_info()
    logger.log(f"Dataset info: {info}")
    
    # Load training data
    logger.log("\nLoading training data...")
    X_raw, y, record_ids = load_training_data(verbose=True)
    logger.log(f"Loaded {len(y)} segments from {len(set(record_ids))} records")
    logger.log(f"Class distribution: Normal={np.sum(y==0)}, Apnea={np.sum(y==1)}")
    
    # Quick test mode - use subset of data
    if quick_test:
        logger.log("\n⚠️ QUICK TEST MODE - Using limited data")
        indices = np.random.choice(len(y), min(1000, len(y)), replace=False)
        X_raw = X_raw[indices]
        y = y[indices]
        record_ids = [record_ids[i] for i in indices]
        epochs = min(5, epochs)
    
    # Preprocess raw data
    logger.log("\nPreprocessing ECG signals...")
    X_preprocessed = preprocess_batch(X_raw)
    logger.log(f"Preprocessed data shape: {X_preprocessed.shape}")
    
    # Split data
    if use_test_set:
        # Load separate test set
        logger.log("\nLoading test data (x01-x35)...")
        X_test_raw, y_test, test_record_ids = load_test_data(verbose=True)
        X_test_preprocessed = preprocess_batch(X_test_raw)
        
        # Split training data into train/val
        logger.log("Splitting training data into train/val...")
        (X_train, X_val, _, y_train, y_val, _,
         train_ids, val_ids, _) = subject_wise_split(
            X_preprocessed, y, record_ids, test_size=0, val_size=0.15
        )
        
        X_test = X_test_preprocessed
    else:
        # Use subject-wise split from training data
        logger.log("\nSplitting data (subject-wise)...")
        (X_train, X_val, X_test, y_train, y_val, y_test,
         train_ids, val_ids, test_ids) = subject_wise_split(
            X_preprocessed, y, record_ids, test_size=0.2, val_size=0.1
        )
    
    logger.log(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    logger.log(f"Train class dist: Normal={np.sum(y_train==0)}, Apnea={np.sum(y_train==1)}")
    
    all_results = {}
    ml_tuned_params = None
    dl_tuned_params = None
    
    # Store features for ensemble (will be set after extraction)
    X_val_features = None
    X_test_features = None
    
    # ================== ML PIPELINE ==================
    if run_ml:
        logger.log("\n" + "="*60)
        logger.log("MACHINE LEARNING MODELS")
        logger.log("="*60)
        
        # Extract features
        logger.log("\nExtracting features for ML models...")
        X_train_features, feature_names = extract_features_batch(X_train, verbose=True)
        X_val_features, _ = extract_features_batch(X_val, verbose=True)
        X_test_features, _ = extract_features_batch(X_test, verbose=True)
        
        logger.log(f"Extracted {len(feature_names)} features")
        logger.log(f"Feature names: {feature_names[:10]}...")  # First 10
        
        # Combine train and val for ML (will use CV)
        X_train_ml = np.vstack([X_train_features, X_val_features])
        y_train_ml = np.concatenate([y_train, y_val])
        train_ids_ml = train_ids + val_ids
        
        # Optuna hyperparameter tuning for ML
        if tune:
            logger.log("\n" + "="*60)
            logger.log("HYPERPARAMETER TUNING - ML MODELS")
            logger.log("="*60)
            
            # Convert record_ids to group array for tuning CV
            unique_records = list(set(train_ids_ml))
            record_to_idx = {r: i for i, r in enumerate(unique_records)}
            groups = np.array([record_to_idx[r] for r in train_ids_ml])
            
            ml_tuning_results = tune_all_ml_models(
                X_train_ml, y_train_ml, groups=groups,
                n_trials=n_trials, verbose=True
            )
            ml_tuned_params = {name: res['params'] for name, res in ml_tuning_results.items()}
            
            logger.log("\n📊 ML Tuning Results:")
            for name, res in ml_tuning_results.items():
                logger.log(f"   {name}: AUC={res['score']:.4f}")
        
        # Run ML pipeline
        ml_results = run_ml_pipeline(
            X_train_ml, y_train_ml,
            X_test_features, y_test,
            train_ids_ml, logger,
            tuned_params=ml_tuned_params
        )
        all_results.update(ml_results)
    
    # ================== DL PIPELINE ==================
    if run_dl:
        logger.log("\n" + "="*60)
        logger.log("DEEP LEARNING MODELS")
        logger.log("="*60)
        
        # Optuna hyperparameter tuning for DL
        if tune:
            logger.log("\n" + "="*60)
            logger.log("HYPERPARAMETER TUNING - DL MODELS")
            logger.log("="*60)
            
            class_weights = compute_class_weights(y_train)
            dl_tuning_results = tune_all_dl_models(
                X_train, y_train, X_val, y_val,
                class_weights=class_weights,
                n_trials=n_trials, verbose=True
            )
            dl_tuned_params = {name: res['params'] for name, res in dl_tuning_results.items()}
            
            logger.log("\n📊 DL Tuning Results:")
            for name, res in dl_tuning_results.items():
                logger.log(f"   {name}: AUC={res['score']:.4f}")
        
        # Run DL pipeline
        dl_results = run_dl_pipeline(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            epochs, logger,
            tuned_params=dl_tuned_params
        )
        all_results.update(dl_results)
        
        # Save training history plots
        for name, result in dl_results.items():
            if 'history' in result:
                plot_training_history(
                    result['history'], name,
                    logger.log_dir / f'{name}_training_history.png'
                )
    
    # ================== ENSEMBLE ==================
    if ensemble and run_ml and run_dl and len(all_results) >= 2:
        logger.log("\n" + "="*60)
        logger.log("ENSEMBLE MODELS")
        logger.log("="*60)
        
        # Create ensemble from best models
        try:
            ensemble_results = evaluate_ensemble_methods(
                all_results,
                X_val_features, X_val, y_val,
                X_test_features, X_test, y_test,
                top_k=top_k,
                verbose=True
            )
            
            # Add ensemble results
            all_results.update(ensemble_results)
            
            # Log ensemble results
            for name, result in ensemble_results.items():
                if 'test_metrics' in result:
                    logger.log_metrics(name, result['test_metrics'], 'test')
                    
        except Exception as e:
            logger.log(f"⚠️ Ensemble creation failed: {e}")
    
    # ================== FINAL RESULTS ==================
    logger.log("\n" + "="*60)
    logger.log("FINAL RESULTS")
    logger.log("="*60)
    
    # Print results table
    print_results_table(all_results)
    
    # Generate plots for all models
    logger.log("\nGenerating evaluation plots...")
    for name, result in all_results.items():
        if 'model' in result:
            model = result['model']
            
            # Get predictions
            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test if run_dl and name in ['CNN1D', 'LSTM', 'CNN_LSTM'] else X_test_features)
                    y_proba = model.predict_proba(X_test if run_dl and name in ['CNN1D', 'LSTM', 'CNN_LSTM'] else X_test_features)[:, 1]
                    
                    # Confusion matrix
                    plot_confusion_matrix(
                        y_test, y_pred, name,
                        logger.log_dir / f'{name}_confusion_matrix.png'
                    )
                    
                    # ROC curve
                    plot_roc_curve(
                        y_test, y_proba, name,
                        logger.log_dir / f'{name}_roc_curve.png'
                    )
                except Exception as e:
                    logger.log(f"Could not generate plots for {name}: {e}")
    
    # Save all results
    logger.log("\nSaving results...")
    logger.results = all_results
    logger.save()
    
    # Find best model (by AUC for better comparison)
    best_model = None
    best_auc = 0
    for name, result in all_results.items():
        if 'test_metrics' in result:
            auc = result['test_metrics'].get('auc', 0)
            if auc > best_auc:
                best_auc = auc
                best_model = name
    
    logger.log("\n" + "="*60)
    logger.log(f"🏆 BEST MODEL: {best_model}")
    if best_model and 'test_metrics' in all_results[best_model]:
        metrics = all_results[best_model]['test_metrics']
        logger.log(f"   AUC: {metrics.get('auc', 'N/A'):.4f}")
        logger.log(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        logger.log(f"   F1: {metrics.get('f1', 'N/A'):.4f}")
        logger.log(f"   Sensitivity: {metrics.get('recall', 'N/A'):.4f}")
        logger.log(f"   Specificity: {metrics.get('specificity', 'N/A'):.4f}")
    logger.log("="*60)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sleep Apnea Detection with Optuna Tuning and Ensemble')
    parser.add_argument('--ml-only', action='store_true', help='Run only ML models')
    parser.add_argument('--dl-only', action='store_true', help='Run only DL models')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for DL')
    parser.add_argument('--use-test-set', action='store_true', 
                        help='Use x01-x35 as test set')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with limited data')
    parser.add_argument('--tune', action='store_true',
                        help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of Optuna trials per model')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable ensemble creation')
    parser.add_argument('--top-k', type=int, default=ENSEMBLE_TOP_K,
                        help='Number of top models for ensemble')
    
    args = parser.parse_args()
    
    run_ml = not args.dl_only
    run_dl = not args.ml_only
    
    results = main(
        run_ml=run_ml,
        run_dl=run_dl,
        epochs=args.epochs,
        use_test_set=args.use_test_set,
        quick_test=args.quick_test,
        tune=args.tune,
        n_trials=args.n_trials,
        ensemble=not args.no_ensemble,
        top_k=args.top_k
    )
