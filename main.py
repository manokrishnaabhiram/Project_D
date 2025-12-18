"""
Main execution script for Sleep Apnea Detection
Trains and evaluates both ML and DL models
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
    OPTUNA_N_TRIALS
)
from src.data_loader import load_training_data, load_test_data, get_dataset_info
from src.preprocessing import preprocess_batch
from src.feature_extraction import extract_features_batch
from src.ml_models import (
    get_all_ml_models, train_and_evaluate_all_models,
    SVMModel, RandomForestModel, GradientBoostingModel, XGBoostModel
)
from src.dl_models import (
    get_all_dl_models, train_and_evaluate_all_dl_models, compute_class_weights,
    CNN1D, LSTMModel, CNNLSTM, DLModelWrapper
)
from src.train_utils import (
    subject_wise_split, plot_confusion_matrix, plot_roc_curve,
    plot_training_history, save_results, print_results_table,
    ExperimentLogger
)
from src.hyperparameter_tuning import optimize_ml_models, optimize_dl_models
from src.ensemble import (
    create_voting_ensemble, create_stacking_ensemble,
    create_weighted_ensemble, create_hybrid_ensemble,
    select_top_models
)


def set_random_seeds(seed: int = RANDOM_STATE):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def run_ml_pipeline(X_train_features: np.ndarray, y_train: np.ndarray,
                    X_test_features: np.ndarray, y_test: np.ndarray,
                    train_record_ids: list, logger: ExperimentLogger) -> dict:
    """
    Run ML pipeline with feature-based classification
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
        verbose=True
    )
    
    # Log results
    for name, result in ml_results.items():
        logger.log_metrics(name, result['test_metrics'], 'test')
    
    return ml_results


def run_dl_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    epochs: int, logger: ExperimentLogger) -> dict:
    """
    Run DL pipeline with raw ECG segments
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
        verbose=True
    )
    
    # Log results
    for name, result in dl_results.items():
        logger.log_metrics(name, result['test_metrics'], 'test')
    
    return dl_results


def run_ml_hyperparameter_tuning(X_train_features: np.ndarray, y_train: np.ndarray,
                                  X_val_features: np.ndarray, y_val: np.ndarray,
                                  train_record_ids: list, logger: ExperimentLogger,
                                  n_trials: int = OPTUNA_N_TRIALS) -> dict:
    """
    Run hyperparameter optimization for ML models
    """
    logger.log("\n" + "="*60)
    logger.log("ML HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    logger.log("="*60)
    
    # Convert record_ids to groups for CV
    unique_records = list(set(train_record_ids))
    record_to_idx = {r: i for i, r in enumerate(unique_records)}
    groups = np.array([record_to_idx[r] for r in train_record_ids])
    
    # Optimize hyperparameters
    optimization_results = optimize_ml_models(
        X_train_features, y_train,
        X_val_features, y_val,
        groups=groups,
        n_trials=n_trials,
        verbose=True
    )
    
    logger.log("\nOptimization Results:")
    for name, result in optimization_results.items():
        logger.log(f"{name}: Best AUC={result['score']:.4f}, Params={result['params']}")
    
    return optimization_results


def run_dl_hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  input_length: int, logger: ExperimentLogger,
                                  n_trials: int = OPTUNA_N_TRIALS,
                                  epochs_per_trial: int = 20) -> dict:
    """
    Run hyperparameter optimization for DL models
    """
    logger.log("\n" + "="*60)
    logger.log("DL HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    logger.log("="*60)
    
    # Optimize hyperparameters
    optimization_results = optimize_dl_models(
        X_train, y_train,
        X_val, y_val,
        input_length=input_length,
        n_trials=n_trials,
        epochs=epochs_per_trial,
        verbose=True
    )
    
    logger.log("\nOptimization Results:")
    for name, result in optimization_results.items():
        logger.log(f"{name}: Best AUC={result['score']:.4f}, Params={result['params']}")
    
    return optimization_results


def train_optimized_ml_models(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              optimization_results: dict,
                              logger: ExperimentLogger) -> dict:
    """
    Train ML models with optimized hyperparameters
    """
    logger.log("\nTraining ML models with optimized hyperparameters...")
    
    results = {}
    model_classes = {
        'SVM': SVMModel,
        'RandomForest': RandomForestModel,
        'GradientBoosting': GradientBoostingModel,
        'XGBoost': XGBoostModel
    }
    
    for name, opt_result in optimization_results.items():
        if name not in model_classes:
            continue
        
        logger.log(f"\nTraining optimized {name}...")
        
        # Create model with optimized parameters
        model_class = model_classes[name]
        model = model_class(**opt_result['params'])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = model.evaluate(X_test, y_test)
        
        results[f"{name}_Optimized"] = {
            'model': model,
            'test_metrics': test_metrics
        }
        
        logger.log(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.log(f"  Test AUC: {test_metrics['auc']:.4f}")
    
    return results


def train_optimized_dl_models(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              optimization_results: dict,
                              input_length: int, epochs: int,
                              logger: ExperimentLogger) -> dict:
    """
    Train DL models with optimized hyperparameters
    """
    logger.log("\nTraining DL models with optimized hyperparameters...")
    
    results = {}
    model_classes = {
        'CNN1D': CNN1D,
        'LSTM': LSTMModel,
        'CNN_LSTM': CNNLSTM
    }
    
    class_weights = compute_class_weights(y_train)
    
    for name, opt_result in optimization_results.items():
        if name not in model_classes:
            continue
        
        logger.log(f"\nTraining optimized {name}...")
        
        # Extract parameters
        params = opt_result['params']
        learning_rate = params.pop('learning_rate', 0.001)
        batch_size = params.pop('batch_size', 32)
        
        # Create model kwargs
        model_kwargs = {'input_length': input_length}
        if name == 'CNN1D' and 'dropout' in params:
            model_kwargs['dropout'] = params['dropout']
        elif name == 'LSTM':
            if 'hidden_size' in params:
                model_kwargs['hidden_size'] = params['hidden_size']
            if 'num_layers' in params:
                model_kwargs['num_layers'] = params['num_layers']
            if 'dropout' in params:
                model_kwargs['dropout'] = params['dropout']
        elif name == 'CNN_LSTM':
            if 'lstm_hidden' in params:
                model_kwargs['lstm_hidden'] = params['lstm_hidden']
            if 'lstm_layers' in params:
                model_kwargs['lstm_layers'] = params['lstm_layers']
            if 'dropout' in params:
                model_kwargs['dropout'] = params['dropout']
        
        # Create and train model
        model_class = model_classes[name]
        model = DLModelWrapper(model_class, f"{name}_Optimized", **model_kwargs)
        model.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            class_weights=class_weights,
            verbose=True
        )
        
        # Evaluate
        test_metrics = model.evaluate(X_test, y_test)
        
        results[f"{name}_Optimized"] = {
            'model': model,
            'test_metrics': test_metrics,
            'history': model.history
        }
        
        logger.log(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.log(f"  Test AUC: {test_metrics['auc']:.4f}")
    
    return results


def create_and_evaluate_ensembles(ml_results: dict, dl_results: dict,
                                  X_train_ml: np.ndarray, y_train: np.ndarray,
                                  X_test_ml: np.ndarray, X_test_dl: np.ndarray,
                                  y_test: np.ndarray,
                                  logger: ExperimentLogger) -> dict:
    """
    Create and evaluate ensemble models
    """
    logger.log("\n" + "="*60)
    logger.log("ENSEMBLE MODELS")
    logger.log("="*60)
    
    ensemble_results = {}
    
    # Select top ML models
    if ml_results:
        logger.log("\nCreating ML ensembles...")
        top_ml_models, top_ml_scores = select_top_models(ml_results, top_n=3, metric='auc')
        
        if len(top_ml_models) >= 2:
            # Voting ensemble
            logger.log("Creating ML Voting Ensemble...")
            voting = create_voting_ensemble(top_ml_models, voting='soft')
            voting.fit(X_train_ml, y_train)
            voting_metrics = voting.evaluate(X_test_ml, y_test)
            ensemble_results['ML_VotingEnsemble'] = {
                'model': voting,
                'test_metrics': voting_metrics
            }
            logger.log(f"  Voting Ensemble - Accuracy: {voting_metrics['accuracy']:.4f}, AUC: {voting_metrics['auc']:.4f}")
            
            # Weighted ensemble
            logger.log("Creating ML Weighted Ensemble...")
            weighted = create_weighted_ensemble(top_ml_models, top_ml_scores)
            weighted_metrics = weighted.evaluate(X_test_ml, y_test)
            ensemble_results['ML_WeightedEnsemble'] = {
                'model': weighted,
                'test_metrics': weighted_metrics
            }
            logger.log(f"  Weighted Ensemble - Accuracy: {weighted_metrics['accuracy']:.4f}, AUC: {weighted_metrics['auc']:.4f}")
    
    # Select top DL models
    if dl_results:
        logger.log("\nCreating DL ensemble...")
        top_dl_models, top_dl_scores = select_top_models(dl_results, top_n=3, metric='auc')
        
        if len(top_dl_models) >= 2:
            # Weighted DL ensemble
            logger.log("Creating DL Weighted Ensemble...")
            dl_weighted = create_weighted_ensemble(top_dl_models, top_dl_scores)
            dl_weighted_metrics = dl_weighted.evaluate(X_test_dl, y_test)
            ensemble_results['DL_WeightedEnsemble'] = {
                'model': dl_weighted,
                'test_metrics': dl_weighted_metrics
            }
            logger.log(f"  DL Weighted Ensemble - Accuracy: {dl_weighted_metrics['accuracy']:.4f}, AUC: {dl_weighted_metrics['auc']:.4f}")
    
    # Hybrid ensemble (ML + DL)
    if ml_results and dl_results:
        logger.log("\nCreating Hybrid Ensemble (ML + DL)...")
        top_ml_models, top_ml_scores = select_top_models(ml_results, top_n=2, metric='auc')
        top_dl_models, top_dl_scores = select_top_models(dl_results, top_n=2, metric='auc')
        
        if top_ml_models and top_dl_models:
            hybrid = create_hybrid_ensemble(
                top_ml_models, top_dl_models,
                top_ml_scores, top_dl_scores,
                ml_dl_ratio=0.5
            )
            hybrid_metrics = hybrid.evaluate(X_test_ml, X_test_dl, y_test)
            ensemble_results['HybridEnsemble'] = {
                'model': hybrid,
                'test_metrics': hybrid_metrics
            }
            logger.log(f"  Hybrid Ensemble - Accuracy: {hybrid_metrics['accuracy']:.4f}, AUC: {hybrid_metrics['auc']:.4f}")
    
    return ensemble_results


def main(run_ml: bool = True, run_dl: bool = True, epochs: int = EPOCHS,
         use_test_set: bool = False, quick_test: bool = False,
         use_optuna: bool = False, optuna_trials: int = OPTUNA_N_TRIALS,
         use_ensemble: bool = False):
    """
    Main function to run the complete sleep apnea detection pipeline
    
    Args:
        run_ml: Whether to run ML models
        run_dl: Whether to run DL models
        epochs: Number of epochs for DL training
        use_test_set: Whether to use separate test set (x01-x35)
        quick_test: Run quick test with limited data
        use_optuna: Whether to use Optuna for hyperparameter tuning
        optuna_trials: Number of Optuna trials
        use_ensemble: Whether to create ensemble models
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
    ml_results = {}
    dl_results = {}
    
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
        
        # Hyperparameter tuning with Optuna
        if use_optuna:
            ml_opt_results = run_ml_hyperparameter_tuning(
                X_train_features, y_train,
                X_val_features, y_val,
                train_ids, logger,
                n_trials=optuna_trials
            )
            
            # Train models with optimized hyperparameters
            X_train_ml = np.vstack([X_train_features, X_val_features])
            y_train_ml = np.concatenate([y_train, y_val])
            
            optimized_ml_results = train_optimized_ml_models(
                X_train_ml, y_train_ml,
                X_test_features, y_test,
                ml_opt_results, logger
            )
            ml_results.update(optimized_ml_results)
        else:
            # Combine train and val for ML (will use CV)
            X_train_ml = np.vstack([X_train_features, X_val_features])
            y_train_ml = np.concatenate([y_train, y_val])
            train_ids_ml = train_ids + val_ids
            
            # Run ML pipeline with default hyperparameters
            ml_results = run_ml_pipeline(
                X_train_ml, y_train_ml,
                X_test_features, y_test,
                train_ids_ml, logger
            )
        
        all_results.update(ml_results)
    
    # ================== DL PIPELINE ==================
    if run_dl:
        logger.log("\n" + "="*60)
        logger.log("DEEP LEARNING MODELS")
        logger.log("="*60)
        
        # Hyperparameter tuning with Optuna
        if use_optuna:
            dl_opt_results = run_dl_hyperparameter_tuning(
                X_train, y_train,
                X_val, y_val,
                X_train.shape[1], logger,
                n_trials=optuna_trials,
                epochs_per_trial=min(20, epochs)
            )
            
            # Train models with optimized hyperparameters
            optimized_dl_results = train_optimized_dl_models(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                dl_opt_results,
                X_train.shape[1], epochs, logger
            )
            dl_results.update(optimized_dl_results)
        else:
            # Run DL pipeline with default hyperparameters
            dl_results = run_dl_pipeline(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                epochs, logger
            )
        
        all_results.update(dl_results)
        
        # Save training history plots
        for name, result in dl_results.items():
            if 'history' in result:
                plot_training_history(
                    result['history'], name,
                    logger.log_dir / f'{name}_training_history.png'
                )
    
    # ================== ENSEMBLE MODELS ==================
    if use_ensemble and (run_ml or run_dl):
        # Create ensemble models
        if run_ml:
            X_train_ml_combined = np.vstack([X_train_features, X_val_features])
            y_train_combined = np.concatenate([y_train, y_val])
        else:
            X_train_ml_combined = None
            y_train_combined = None
        
        ensemble_results = create_and_evaluate_ensembles(
            ml_results if run_ml else {},
            dl_results if run_dl else {},
            X_train_ml_combined if run_ml else None,
            y_train_combined if run_ml else None,
            X_test_features if run_ml else None,
            X_test if run_dl else None,
            y_test, logger
        )
        all_results.update(ensemble_results)
    
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
    
    # Find best model
    best_model = None
    best_accuracy = 0
    for name, result in all_results.items():
        if 'test_metrics' in result:
            acc = result['test_metrics'].get('accuracy', 0)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = name
    
    logger.log("\n" + "="*60)
    logger.log(f"🏆 BEST MODEL: {best_model}")
    logger.log(f"   Accuracy: {best_accuracy:.4f}")
    if best_model and 'test_metrics' in all_results[best_model]:
        logger.log(f"   AUC: {all_results[best_model]['test_metrics'].get('auc', 'N/A'):.4f}")
    logger.log("="*60)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sleep Apnea Detection with Optuna and Ensemble')
    parser.add_argument('--ml-only', action='store_true', help='Run only ML models')
    parser.add_argument('--dl-only', action='store_true', help='Run only DL models')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for DL')
    parser.add_argument('--use-test-set', action='store_true', 
                        help='Use x01-x35 as test set')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with limited data')
    parser.add_argument('--use-optuna', action='store_true',
                        help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--optuna-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of Optuna trials for hyperparameter optimization')
    parser.add_argument('--use-ensemble', action='store_true',
                        help='Create and evaluate ensemble models')
    
    args = parser.parse_args()
    
    run_ml = not args.dl_only
    run_dl = not args.ml_only
    
    results = main(
        run_ml=run_ml,
        run_dl=run_dl,
        epochs=args.epochs,
        use_test_set=args.use_test_set,
        quick_test=args.quick_test,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials,
        use_ensemble=args.use_ensemble
    )
