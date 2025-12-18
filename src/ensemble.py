"""
Ensemble module for combining ML and DL models
Supports voting and stacking ensemble methods for hybrid ML+DL ensembles
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from pathlib import Path

from .config import (
    RANDOM_STATE, MODELS_DIR, ENSEMBLE_TOP_K, 
    ENSEMBLE_METHOD, ENSEMBLE_VOTING_TYPE
)


class HybridEnsemble:
    """
    Hybrid Ensemble that combines ML (feature-based) and DL (raw ECG) models.
    
    Supports two ensemble methods:
    - Voting: Combines predictions using soft or hard voting
    - Stacking: Uses a meta-learner trained on base model probabilities
    
    Handles the dual input requirement:
    - ML models receive extracted features (X_features)
    - DL models receive raw/preprocessed ECG segments (X_raw)
    """
    
    def __init__(self, method: str = ENSEMBLE_METHOD, 
                 voting_type: str = ENSEMBLE_VOTING_TYPE):
        """
        Initialize ensemble
        
        Args:
            method: 'voting' or 'stacking'
            voting_type: 'soft' or 'hard' (only for voting method)
        """
        self.method = method
        self.voting_type = voting_type
        
        self.ml_models: Dict[str, Any] = {}  # {name: model}
        self.dl_models: Dict[str, Any] = {}  # {name: model}
        self.model_weights: Dict[str, float] = {}  # Optional weights per model
        
        # Meta-learner for stacking
        self.meta_learner = None
        self.is_fitted = False
    
    def add_ml_model(self, name: str, model: Any, weight: float = 1.0):
        """Add an ML model to the ensemble"""
        self.ml_models[name] = model
        self.model_weights[name] = weight
    
    def add_dl_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a DL model to the ensemble"""
        self.dl_models[name] = model
        self.model_weights[name] = weight
    
    def add_models_from_results(self, results: Dict[str, Dict], 
                                top_k: int = ENSEMBLE_TOP_K,
                                ml_models_list: Optional[List[str]] = None,
                                dl_models_list: Optional[List[str]] = None):
        """
        Add top-k models from results dictionary based on AUC score
        
        Args:
            results: Results dictionary from main pipeline
            top_k: Number of top models to add
            ml_models_list: List of ML model names to consider
            dl_models_list: List of DL model names to consider
        """
        if ml_models_list is None:
            ml_models_list = ['SVM', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LogisticRegression']
        if dl_models_list is None:
            dl_models_list = ['CNN1D', 'LSTM', 'CNN_LSTM']
        
        # Collect all models with their AUC scores
        model_scores = []
        for name, result in results.items():
            if 'model' in result and 'test_metrics' in result:
                auc = result['test_metrics'].get('auc', 0.0)
                model_type = 'ml' if name in ml_models_list else 'dl' if name in dl_models_list else None
                if model_type:
                    model_scores.append((name, result['model'], auc, model_type))
        
        # Sort by AUC and select top-k
        model_scores.sort(key=lambda x: x[2], reverse=True)
        selected = model_scores[:top_k]
        
        print(f"\n📦 Ensemble: Selected top-{top_k} models:")
        for name, model, auc, model_type in selected:
            print(f"   - {name} (AUC: {auc:.4f}, Type: {model_type.upper()})")
            if model_type == 'ml':
                self.add_ml_model(name, model, weight=auc)
            else:
                self.add_dl_model(name, model, weight=auc)
    
    def _get_all_probabilities(self, X_features: np.ndarray, 
                                X_raw: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from all models
        
        Args:
            X_features: Feature matrix for ML models
            X_raw: Raw ECG segments for DL models
        
        Returns:
            probabilities: Array of shape (n_samples, n_models) with apnea probabilities
        """
        all_proba = []
        
        # ML models
        for name, model in self.ml_models.items():
            proba = model.predict_proba(X_features)[:, 1]
            all_proba.append(proba)
        
        # DL models
        for name, model in self.dl_models.items():
            proba = model.predict_proba(X_raw)[:, 1]
            all_proba.append(proba)
        
        return np.column_stack(all_proba)
    
    def _get_model_weights_array(self) -> np.ndarray:
        """Get weights array in same order as probabilities"""
        weights = []
        for name in self.ml_models.keys():
            weights.append(self.model_weights.get(name, 1.0))
        for name in self.dl_models.keys():
            weights.append(self.model_weights.get(name, 1.0))
        return np.array(weights)
    
    def fit(self, X_features: np.ndarray, X_raw: np.ndarray, y: np.ndarray):
        """
        Fit the ensemble (only needed for stacking method)
        
        Args:
            X_features: Feature matrix for ML models
            X_raw: Raw ECG segments for DL models
            y: Labels
        """
        if self.method == 'stacking':
            # Get base model probabilities
            base_proba = self._get_all_probabilities(X_features, X_raw)
            
            # Train meta-learner
            self.meta_learner = LogisticRegression(
                class_weight='balanced',
                random_state=RANDOM_STATE,
                max_iter=1000
            )
            self.meta_learner.fit(base_proba, y)
            print(f"   Meta-learner trained on {base_proba.shape[1]} base models")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X_features: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using ensemble
        
        Args:
            X_features: Feature matrix for ML models
            X_raw: Raw ECG segments for DL models
        
        Returns:
            probabilities: Array of shape (n_samples, 2) with [normal_prob, apnea_prob]
        """
        base_proba = self._get_all_probabilities(X_features, X_raw)
        
        if self.method == 'voting':
            if self.voting_type == 'soft':
                # Weighted average of probabilities
                weights = self._get_model_weights_array()
                weights = weights / weights.sum()  # Normalize
                ensemble_proba = np.average(base_proba, axis=1, weights=weights)
            else:
                # Hard voting - average of binary predictions
                predictions = (base_proba > 0.5).astype(float)
                ensemble_proba = predictions.mean(axis=1)
            
            return np.column_stack([1 - ensemble_proba, ensemble_proba])
        
        elif self.method == 'stacking':
            if self.meta_learner is None:
                raise ValueError("Stacking ensemble not fitted. Call fit() first.")
            return self.meta_learner.predict_proba(base_proba)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def predict(self, X_features: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
        """
        Predict class labels using ensemble
        
        Args:
            X_features: Feature matrix for ML models
            X_raw: Raw ECG segments for DL models
        
        Returns:
            predictions: Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X_features, X_raw)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def evaluate(self, X_features: np.ndarray, X_raw: np.ndarray, 
                 y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        
        Args:
            X_features: Feature matrix for ML models
            X_raw: Raw ECG segments for DL models
            y: True labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_features, X_raw)
        y_proba = self.predict_proba(X_features, X_raw)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        return metrics
    
    def save(self, path: Optional[Path] = None):
        """Save ensemble to disk"""
        if path is None:
            path = MODELS_DIR / "ensemble_model.joblib"
        
        data = {
            'method': self.method,
            'voting_type': self.voting_type,
            'ml_models': self.ml_models,
            'dl_models': self.dl_models,
            'model_weights': self.model_weights,
            'meta_learner': self.meta_learner,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, path)
    
    def load(self, path: Optional[Path] = None):
        """Load ensemble from disk"""
        if path is None:
            path = MODELS_DIR / "ensemble_model.joblib"
        
        data = joblib.load(path)
        self.method = data['method']
        self.voting_type = data['voting_type']
        self.ml_models = data['ml_models']
        self.dl_models = data['dl_models']
        self.model_weights = data['model_weights']
        self.meta_learner = data['meta_learner']
        self.is_fitted = data['is_fitted']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble configuration"""
        return {
            'method': self.method,
            'voting_type': self.voting_type if self.method == 'voting' else None,
            'n_ml_models': len(self.ml_models),
            'n_dl_models': len(self.dl_models),
            'ml_model_names': list(self.ml_models.keys()),
            'dl_model_names': list(self.dl_models.keys()),
            'model_weights': self.model_weights,
            'is_fitted': self.is_fitted
        }


def create_ensemble_from_results(results: Dict[str, Dict],
                                 X_features_val: np.ndarray,
                                 X_raw_val: np.ndarray,
                                 y_val: np.ndarray,
                                 top_k: int = ENSEMBLE_TOP_K,
                                 method: str = ENSEMBLE_METHOD,
                                 voting_type: str = ENSEMBLE_VOTING_TYPE,
                                 verbose: bool = True) -> HybridEnsemble:
    """
    Create and fit an ensemble from pipeline results
    
    Args:
        results: Results dictionary from main pipeline
        X_features_val: Validation features for ML models
        X_raw_val: Validation raw data for DL models
        y_val: Validation labels
        top_k: Number of top models to include
        method: Ensemble method ('voting' or 'stacking')
        voting_type: Voting type for voting ensemble
        verbose: Whether to print progress
    
    Returns:
        Fitted HybridEnsemble
    """
    if verbose:
        print(f"\n🔗 Creating {method.upper()} ensemble from top-{top_k} models...")
    
    ensemble = HybridEnsemble(method=method, voting_type=voting_type)
    ensemble.add_models_from_results(results, top_k=top_k)
    
    if method == 'stacking':
        if verbose:
            print("   Training meta-learner...")
        ensemble.fit(X_features_val, X_raw_val, y_val)
    else:
        ensemble.is_fitted = True
    
    return ensemble


def evaluate_ensemble_methods(results: Dict[str, Dict],
                              X_features_val: np.ndarray,
                              X_raw_val: np.ndarray,
                              y_val: np.ndarray,
                              X_features_test: np.ndarray,
                              X_raw_test: np.ndarray,
                              y_test: np.ndarray,
                              top_k: int = ENSEMBLE_TOP_K,
                              verbose: bool = True) -> Dict[str, Dict]:
    """
    Evaluate different ensemble methods and return best one
    
    Returns:
        Dictionary with ensemble results for each method
    """
    ensemble_results = {}
    
    for method in ['voting', 'stacking']:
        for voting_type in (['soft', 'hard'] if method == 'voting' else [None]):
            name = f"Ensemble_{method}" + (f"_{voting_type}" if voting_type else "")
            
            try:
                ensemble = create_ensemble_from_results(
                    results, X_features_val, X_raw_val, y_val,
                    top_k=top_k, method=method,
                    voting_type=voting_type or 'soft',
                    verbose=False
                )
                
                metrics = ensemble.evaluate(X_features_test, X_raw_test, y_test)
                
                ensemble_results[name] = {
                    'model': ensemble,
                    'test_metrics': metrics,
                    'info': ensemble.get_model_info()
                }
                
                if verbose:
                    print(f"   {name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ {name} failed: {e}")
    
    return ensemble_results
