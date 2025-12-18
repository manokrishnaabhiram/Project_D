"""
Ensemble methods for combining ML and DL models
"""
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .config import RANDOM_STATE, ENSEMBLE_VOTING_TYPE, ENSEMBLE_TOP_N_MODELS


class EnsembleModel:
    """Base class for ensemble models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        return metrics


class VotingEnsemble(EnsembleModel):
    """Voting ensemble for ML models"""
    
    def __init__(self, estimators: List[Tuple[str, Any]], voting: str = 'soft'):
        """
        Initialize voting ensemble
        
        Args:
            estimators: List of (name, model) tuples
            voting: 'hard' or 'soft' voting
        """
        super().__init__(f'VotingEnsemble_{voting}')
        self.estimators = estimators
        self.voting = voting
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """Fit the ensemble"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble for ML models"""
    
    def __init__(self, estimators: List[Tuple[str, Any]], 
                 final_estimator: Optional[Any] = None):
        """
        Initialize stacking ensemble
        
        Args:
            estimators: List of (name, model) tuples for base models
            final_estimator: Meta-learner (default: LogisticRegression)
        """
        super().__init__('StackingEnsemble')
        self.estimators = estimators
        
        if final_estimator is None:
            final_estimator = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            )
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit the ensemble"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class WeightedAverageEnsemble(EnsembleModel):
    """Weighted average ensemble for combining predictions"""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted average ensemble
        
        Args:
            models: Dictionary of {name: model}
            weights: Dictionary of {name: weight} (default: equal weights)
        """
        super().__init__('WeightedAverageEnsemble')
        self.models = models
        
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {name: w / total_weight for name, w in weights.items()}
        
        self.weights = weights
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted average"""
        weighted_proba = None
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weight = self.weights[name]
            
            if weighted_proba is None:
                weighted_proba = proba * weight
            else:
                weighted_proba += proba * weight
        
        return weighted_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class HybridEnsemble:
    """
    Hybrid ensemble that combines ML and DL models
    Can handle different input formats (features for ML, raw data for DL)
    """
    
    def __init__(self, ml_models: Dict[str, Any], dl_models: Dict[str, Any],
                 ml_weights: Optional[Dict[str, float]] = None,
                 dl_weights: Optional[Dict[str, float]] = None,
                 ml_dl_weight_ratio: float = 0.5):
        """
        Initialize hybrid ensemble
        
        Args:
            ml_models: Dictionary of {name: ML model}
            dl_models: Dictionary of {name: DL model}
            ml_weights: Weights for ML models (default: equal)
            dl_weights: Weights for DL models (default: equal)
            ml_dl_weight_ratio: Weight ratio between ML and DL (0-1, default 0.5 for equal)
        """
        self.ml_models = ml_models
        self.dl_models = dl_models
        self.ml_dl_weight_ratio = ml_dl_weight_ratio
        
        # Normalize ML weights
        if ml_weights is None:
            ml_weights = {name: 1.0 / len(ml_models) for name in ml_models.keys()}
        else:
            total = sum(ml_weights.values())
            ml_weights = {name: w / total for name, w in ml_weights.items()}
        
        # Normalize DL weights
        if dl_weights is None:
            dl_weights = {name: 1.0 / len(dl_models) for name in dl_models.keys()}
        else:
            total = sum(dl_weights.values())
            dl_weights = {name: w / total for name, w in dl_weights.items()}
        
        self.ml_weights = ml_weights
        self.dl_weights = dl_weights
        self.name = 'HybridEnsemble'
        self.is_fitted = True
    
    def predict_proba(self, X_ml: np.ndarray, X_dl: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using hybrid ensemble
        
        Args:
            X_ml: Features for ML models
            X_dl: Raw data for DL models
        
        Returns:
            Averaged probabilities
        """
        # Get ML predictions
        ml_proba = None
        if self.ml_models:
            for name, model in self.ml_models.items():
                proba = model.predict_proba(X_ml)
                weight = self.ml_weights[name] * self.ml_dl_weight_ratio
                
                if ml_proba is None:
                    ml_proba = proba * weight
                else:
                    ml_proba += proba * weight
        
        # Get DL predictions
        dl_proba = None
        if self.dl_models:
            for name, model in self.dl_models.items():
                proba = model.predict_proba(X_dl)
                weight = self.dl_weights[name] * (1 - self.ml_dl_weight_ratio)
                
                if dl_proba is None:
                    dl_proba = proba * weight
                else:
                    dl_proba += proba * weight
        
        # Combine predictions
        if ml_proba is not None and dl_proba is not None:
            return ml_proba + dl_proba
        elif ml_proba is not None:
            return ml_proba
        elif dl_proba is not None:
            return dl_proba
        else:
            raise ValueError("No models available for prediction")
    
    def predict(self, X_ml: np.ndarray, X_dl: np.ndarray) -> np.ndarray:
        """Predict labels"""
        proba = self.predict_proba(X_ml, X_dl)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X_ml: np.ndarray, X_dl: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_ml, X_dl)
        y_proba = self.predict_proba(X_ml, X_dl)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        return metrics


def create_voting_ensemble(models: Dict[str, Any], voting: str = ENSEMBLE_VOTING_TYPE) -> VotingEnsemble:
    """
    Create voting ensemble from trained ML models
    
    Args:
        models: Dictionary of {name: trained model}
        voting: 'hard' or 'soft' voting
    
    Returns:
        VotingEnsemble instance
    """
    estimators = [(name, model.model) for name, model in models.items()]
    return VotingEnsemble(estimators, voting)


def create_stacking_ensemble(models: Dict[str, Any]) -> StackingEnsemble:
    """
    Create stacking ensemble from trained ML models
    
    Args:
        models: Dictionary of {name: trained model}
    
    Returns:
        StackingEnsemble instance
    """
    estimators = [(name, model.model) for name, model in models.items()]
    return StackingEnsemble(estimators)


def create_weighted_ensemble(models: Dict[str, Any], 
                             validation_scores: Optional[Dict[str, float]] = None) -> WeightedAverageEnsemble:
    """
    Create weighted ensemble based on validation performance
    
    Args:
        models: Dictionary of {name: trained model}
        validation_scores: Dictionary of {name: validation score}
    
    Returns:
        WeightedAverageEnsemble instance
    """
    weights = None
    if validation_scores is not None:
        weights = validation_scores
    
    return WeightedAverageEnsemble(models, weights)


def create_hybrid_ensemble(ml_models: Dict[str, Any], dl_models: Dict[str, Any],
                          ml_scores: Optional[Dict[str, float]] = None,
                          dl_scores: Optional[Dict[str, float]] = None,
                          ml_dl_ratio: float = 0.5) -> HybridEnsemble:
    """
    Create hybrid ensemble combining ML and DL models
    
    Args:
        ml_models: Dictionary of {name: ML model}
        dl_models: Dictionary of {name: DL model}
        ml_scores: Validation scores for ML models
        dl_scores: Validation scores for DL models
        ml_dl_ratio: Weight ratio between ML and DL (0-1)
    
    Returns:
        HybridEnsemble instance
    """
    return HybridEnsemble(ml_models, dl_models, ml_scores, dl_scores, ml_dl_ratio)


def select_top_models(results: Dict[str, Dict], top_n: int = ENSEMBLE_TOP_N_MODELS,
                     metric: str = 'auc') -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Select top N models based on validation performance
    
    Args:
        results: Dictionary of model results with 'model' and 'test_metrics'
        top_n: Number of top models to select
        metric: Metric to use for selection
    
    Returns:
        Tuple of (top_models dict, top_scores dict)
    """
    # Extract scores
    scores = {}
    for name, result in results.items():
        if 'test_metrics' in result and metric in result['test_metrics']:
            scores[name] = result['test_metrics'][metric]
        elif 'cv_metrics' in result and metric in result['cv_metrics']:
            scores[name] = result['cv_metrics'][metric]
    
    # Sort and select top N
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    top_models = {}
    top_scores = {}
    
    for name, score in sorted_models:
        if name in results and 'model' in results[name]:
            top_models[name] = results[name]['model']
            top_scores[name] = score
    
    return top_models, top_scores


if __name__ == "__main__":
    # Test ensemble methods
    print("Testing ensemble methods...")
    
    from .ml_models import RandomForestModel, SVMModel
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    X_train = np.random.randn(500, 20)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(200, 20)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    # Train base models
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    
    svm = SVMModel()
    svm.fit(X_train, y_train)
    
    models = {'RandomForest': rf, 'SVM': svm}
    
    # Test voting ensemble
    print("\nTesting voting ensemble...")
    voting = create_voting_ensemble(models, voting='soft')
    voting.fit(X_train, y_train)
    metrics = voting.evaluate(X_test, y_test)
    print(f"Voting ensemble metrics: {metrics}")
    
    # Test weighted ensemble
    print("\nTesting weighted ensemble...")
    scores = {'RandomForest': 0.9, 'SVM': 0.85}
    weighted = create_weighted_ensemble(models, scores)
    metrics = weighted.evaluate(X_test, y_test)
    print(f"Weighted ensemble metrics: {metrics}")
    
    print("\nEnsemble methods test completed!")
