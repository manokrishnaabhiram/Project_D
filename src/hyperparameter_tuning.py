"""
Hyperparameter tuning module using Optuna
Provides objective functions for optimizing ML and DL models
"""
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, Callable
import warnings
import copy

from .config import (
    RANDOM_STATE, DEVICE, BATCH_SIZE, LEARNING_RATE,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_STUDY_DIR,
    EARLY_STOPPING_PATIENCE, SEGMENT_SAMPLES
)
from .dl_models import ECGDataset, CNN1D, LSTMModel, CNNLSTM, DLModelWrapper

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# ML Model Objective Functions
# ============================================================

class MLObjective:
    """Base class for ML model optimization"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 groups: Optional[np.ndarray] = None, n_splits: int = 5):
        self.X = X
        self.y = y
        self.groups = groups
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
    
    def __call__(self, trial: optuna.Trial) -> float:
        raise NotImplementedError


class SVMObjective(MLObjective):
    """Optuna objective for SVM hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Hyperparameter search space
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3
        
        model = SVC(
            C=C, gamma=gamma, kernel=kernel, degree=degree,
            class_weight='balanced', probability=True,
            random_state=RANDOM_STATE
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()


class RandomForestObjective(MLObjective):
    """Optuna objective for Random Forest hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()


class GradientBoostingObjective(MLObjective):
    """Optuna objective for Gradient Boosting hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            random_state=RANDOM_STATE
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()


class XGBoostObjective(MLObjective):
    """Optuna objective for XGBoost hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            warnings.warn("XGBoost not available, skipping")
            return 0.0
        
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()


class LogisticRegressionObjective(MLObjective):
    """Optuna objective for Logistic Regression hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        C = trial.suggest_float('C', 1e-4, 100, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        
        model = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()


# ============================================================
# DL Model Objective Functions
# ============================================================

class DLObjective:
    """Base class for DL model optimization"""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 class_weights: Optional[np.ndarray] = None,
                 max_epochs: int = 30):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.class_weights = class_weights
        self.max_epochs = max_epochs
        self.input_length = X_train.shape[1]
    
    def _train_and_evaluate(self, model: nn.Module, trial: optuna.Trial,
                            learning_rate: float, batch_size: int) -> float:
        """Train model and return validation AUC"""
        model = model.to(DEVICE)
        
        # Data loaders
        train_dataset = ECGDataset(self.X_train, self.y_train)
        val_dataset = ECGDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        if self.class_weights is not None:
            weights = torch.FloatTensor(self.class_weights).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Training
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_proba = []
            val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    outputs = model(batch_X)
                    proba = torch.softmax(outputs, dim=1)[:, 1]
                    val_proba.extend(proba.cpu().numpy())
                    val_labels.extend(batch_y.numpy())
            
            val_auc = roc_auc_score(val_labels, val_proba)
            scheduler.step(val_auc)
            
            # Early stopping and pruning
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 5:
                break
            
            # Optuna pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_auc


class CNN1DObjective(DLObjective):
    """Optuna objective for 1D CNN hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Architecture hyperparameters
        dropout = trial.suggest_float('dropout', 0.2, 0.6)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        model = CNN1D(
            input_length=self.input_length,
            num_classes=2,
            dropout=dropout
        )
        
        return self._train_and_evaluate(model, trial, learning_rate, batch_size)


class LSTMObjective(DLObjective):
    """Optuna objective for LSTM hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Architecture hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.2, 0.6)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        model = LSTMModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        return self._train_and_evaluate(model, trial, learning_rate, batch_size)


class CNNLSTMObjective(DLObjective):
    """Optuna objective for CNN-LSTM hybrid hyperparameter tuning"""
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Architecture hyperparameters
        cnn_filters_base = trial.suggest_categorical('cnn_filters_base', [16, 32, 64])
        cnn_filters = [cnn_filters_base, cnn_filters_base * 2, cnn_filters_base * 4]
        lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 128, 256])
        lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.2, 0.6)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        model = CNNLSTM(
            input_length=self.input_length,
            num_classes=2,
            cnn_filters=cnn_filters,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        
        return self._train_and_evaluate(model, trial, learning_rate, batch_size)


# ============================================================
# Tuning Functions
# ============================================================

def tune_ml_model(model_name: str, X: np.ndarray, y: np.ndarray,
                  groups: Optional[np.ndarray] = None,
                  n_trials: int = OPTUNA_N_TRIALS,
                  timeout: Optional[int] = OPTUNA_TIMEOUT,
                  verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Tune a single ML model using Optuna
    
    Args:
        model_name: Name of the model ('SVM', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LogisticRegression')
        X: Feature matrix
        y: Labels
        groups: Optional group labels for grouped CV
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        verbose: Whether to show progress
    
    Returns:
        best_params: Best hyperparameters
        best_score: Best cross-validation AUC score
    """
    objective_map = {
        'SVM': SVMObjective,
        'RandomForest': RandomForestObjective,
        'GradientBoosting': GradientBoostingObjective,
        'XGBoost': XGBoostObjective,
        'LogisticRegression': LogisticRegressionObjective
    }
    
    if model_name not in objective_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    objective = objective_map[model_name](X, y, groups)
    
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f'{model_name}_tuning'
    )
    
    if verbose:
        print(f"\n🔧 Tuning {model_name}...")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )
    
    if verbose:
        print(f"   Best AUC: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def tune_dl_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  class_weights: Optional[np.ndarray] = None,
                  n_trials: int = OPTUNA_N_TRIALS,
                  timeout: Optional[int] = OPTUNA_TIMEOUT,
                  verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Tune a single DL model using Optuna
    
    Args:
        model_name: Name of the model ('CNN1D', 'LSTM', 'CNN_LSTM')
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        class_weights: Optional class weights
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        verbose: Whether to show progress
    
    Returns:
        best_params: Best hyperparameters
        best_score: Best validation AUC score
    """
    objective_map = {
        'CNN1D': CNN1DObjective,
        'LSTM': LSTMObjective,
        'CNN_LSTM': CNNLSTMObjective
    }
    
    if model_name not in objective_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    objective = objective_map[model_name](X_train, y_train, X_val, y_val, class_weights)
    
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'{model_name}_tuning'
    )
    
    if verbose:
        print(f"\n🔧 Tuning {model_name}...")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )
    
    if verbose:
        print(f"   Best AUC: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def tune_all_ml_models(X: np.ndarray, y: np.ndarray,
                       groups: Optional[np.ndarray] = None,
                       n_trials: int = OPTUNA_N_TRIALS,
                       timeout: Optional[int] = OPTUNA_TIMEOUT,
                       verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Tune all ML models
    
    Returns:
        Dictionary with model names as keys and {'params': best_params, 'score': best_score} as values
    """
    model_names = ['SVM', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LogisticRegression']
    results = {}
    
    for name in model_names:
        try:
            params, score = tune_ml_model(name, X, y, groups, n_trials, timeout, verbose)
            results[name] = {'params': params, 'score': score}
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to tune {name}: {e}")
            results[name] = {'params': {}, 'score': 0.0}
    
    return results


def tune_all_dl_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       class_weights: Optional[np.ndarray] = None,
                       n_trials: int = OPTUNA_N_TRIALS,
                       timeout: Optional[int] = OPTUNA_TIMEOUT,
                       verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Tune all DL models
    
    Returns:
        Dictionary with model names as keys and {'params': best_params, 'score': best_score} as values
    """
    model_names = ['CNN1D', 'LSTM', 'CNN_LSTM']
    results = {}
    
    for name in model_names:
        try:
            params, score = tune_dl_model(name, X_train, y_train, X_val, y_val, 
                                          class_weights, n_trials, timeout, verbose)
            results[name] = {'params': params, 'score': score}
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to tune {name}: {e}")
            results[name] = {'params': {}, 'score': 0.0}
    
    return results
