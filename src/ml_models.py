"""
Machine Learning models for sleep apnea detection
Includes SVM, Random Forest, XGBoost, and Gradient Boosting
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path

from .config import RANDOM_STATE, MODELS_DIR


class MLModelBase:
    """Base class for ML models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLModelBase':
        """Fit the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba, use decision function
            decision = self.model.decision_function(X_scaled)
            # Convert to probabilities using sigmoid
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),  # Sensitivity
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        return metrics
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk"""
        if path is None:
            path = MODELS_DIR / f"{self.name}_model.joblib"
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path: Optional[Path] = None):
        """Load model from disk"""
        if path is None:
            path = MODELS_DIR / f"{self.name}_model.joblib"
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True


class SVMModel(MLModelBase):
    """Support Vector Machine classifier"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 class_weight: str = 'balanced'):
        super().__init__('SVM')
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=RANDOM_STATE
        )


class RandomForestModel(MLModelBase):
    """Random Forest classifier"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, class_weight: str = 'balanced'):
        super().__init__('RandomForest')
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class GradientBoostingModel(MLModelBase):
    """Gradient Boosting classifier"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        super().__init__('GradientBoosting')
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class XGBoostModel(MLModelBase):
    """XGBoost classifier (if available, falls back to GradientBoosting)"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6):
        super().__init__('XGBoost')
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        except ImportError:
            print("XGBoost not available, using GradientBoosting instead")
            self.name = 'GradientBoosting_fallback'
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=RANDOM_STATE
            )
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class LogisticRegressionModel(MLModelBase):
    """Logistic Regression classifier (baseline)"""
    
    def __init__(self, C: float = 1.0, class_weight: str = 'balanced'):
        super().__init__('LogisticRegression')
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            max_iter=1000,
            n_jobs=-1
        )


def get_all_ml_models(tuned_params: Optional[Dict[str, Dict]] = None) -> Dict[str, MLModelBase]:
    """
    Get dictionary of all ML models
    
    Args:
        tuned_params: Optional dictionary of tuned hyperparameters per model
                     e.g., {'SVM': {'C': 10, 'kernel': 'rbf'}, ...}
    """
    params = tuned_params or {}
    
    return {
        'SVM': SVMModel(**params.get('SVM', {})),
        'RandomForest': RandomForestModel(**params.get('RandomForest', {})),
        'GradientBoosting': GradientBoostingModel(**params.get('GradientBoosting', {})),
        'XGBoost': XGBoostModel(**params.get('XGBoost', {})),
        'LogisticRegression': LogisticRegressionModel(**params.get('LogisticRegression', {}))
    }


def cross_validate_model(model: MLModelBase, X: np.ndarray, y: np.ndarray,
                         groups: Optional[np.ndarray] = None,
                         n_splits: int = 5) -> Dict[str, float]:
    """
    Cross-validate a model
    
    Args:
        model: ML model instance
        X: Feature matrix
        y: Labels
        groups: Group labels for leave-one-group-out CV
        n_splits: Number of CV folds (if groups is None)
    
    Returns:
        Dictionary of mean metrics
    """
    from sklearn.model_selection import cross_validate
    
    # Create pipeline with scaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model.model)
    ])
    
    # Choose CV strategy
    if groups is not None:
        cv = LeaveOneGroupOut()
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'auc': 'roc_auc'
    }
    
    # Perform cross-validation
    if groups is not None:
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, groups=groups)
    else:
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    
    # Calculate mean metrics
    metrics = {
        'accuracy': np.mean(cv_results['test_accuracy']),
        'accuracy_std': np.std(cv_results['test_accuracy']),
        'precision': np.mean(cv_results['test_precision']),
        'precision_std': np.std(cv_results['test_precision']),
        'recall': np.mean(cv_results['test_recall']),
        'recall_std': np.std(cv_results['test_recall']),
        'f1': np.mean(cv_results['test_f1']),
        'f1_std': np.std(cv_results['test_f1']),
        'auc': np.mean(cv_results['test_auc']),
        'auc_std': np.std(cv_results['test_auc'])
    }
    
    return metrics


def train_and_evaluate_all_models(X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  groups_train: Optional[np.ndarray] = None,
                                  verbose: bool = True,
                                  tuned_params: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    """
    Train and evaluate all ML models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        groups_train: Group labels for CV
        verbose: Whether to print progress
        tuned_params: Optional dictionary of tuned hyperparameters per model
    
    Returns:
        Dictionary of results for each model
    """
    models = get_all_ml_models(tuned_params)
    results = {}
    
    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        # Cross-validation on training set
        cv_metrics = cross_validate_model(model, X_train, y_train, groups=groups_train)
        
        results[name] = {
            'model': model,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics
        }
        
        if verbose:
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test AUC: {test_metrics['auc']:.4f}")
            print(f"  CV Accuracy: {cv_metrics['accuracy']:.4f} (+/- {cv_metrics['accuracy_std']:.4f})")
    
    return results


if __name__ == "__main__":
    # Test ML models
    print("Testing ML models...")
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Train and evaluate
    results = train_and_evaluate_all_models(X_train, y_train, X_test, y_test)
    
    print("\n=== Summary ===")
    for name, result in results.items():
        print(f"{name}: Accuracy={result['test_metrics']['accuracy']:.4f}, AUC={result['test_metrics']['auc']:.4f}")
