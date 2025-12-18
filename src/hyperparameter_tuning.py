"""
Hyperparameter optimization using Optuna for ML and DL models
"""
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from .config import (
    RANDOM_STATE, ML_PARAM_SPACE, DL_PARAM_SPACE, 
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, DEVICE, BATCH_SIZE
)
from .ml_models import (
    SVMModel, RandomForestModel, GradientBoostingModel, 
    XGBoostModel, cross_validate_model
)
from .dl_models import CNN1D, LSTMModel, CNNLSTM, DLModelWrapper


class MLOptunaOptimizer:
    """Optuna optimizer for ML models"""
    
    def __init__(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, 
                 groups: Optional[np.ndarray] = None):
        """
        Initialize optimizer
        
        Args:
            model_name: Name of the model (SVM, RandomForest, etc.)
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            groups: Group labels for cross-validation
        """
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.groups = groups
        self.param_space = ML_PARAM_SPACE.get(model_name, {})
        
    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters based on the model type"""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple):
                if len(param_config) == 2:
                    # Integer parameter
                    params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                elif len(param_config) == 3:
                    if param_config[2] == 'log':
                        # Float parameter with log scale
                        params[param_name] = trial.suggest_float(
                            param_name, param_config[0], param_config[1], log=True
                        )
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(
                            param_name, param_config[0], param_config[1]
                        )
        
        return params
    
    def _create_model(self, params: Dict):
        """Create model instance with given parameters"""
        if self.model_name == 'SVM':
            return SVMModel(**params)
        elif self.model_name == 'RandomForest':
            return RandomForestModel(**params)
        elif self.model_name == 'GradientBoosting':
            return GradientBoostingModel(**params)
        elif self.model_name == 'XGBoost':
            return XGBoostModel(**params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        # Create and train model
        model = self._create_model(params)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        metrics = model.evaluate(self.X_val, self.y_val)
        
        # Return metric to optimize (AUC)
        return metrics['auc']
    
    def optimize(self, n_trials: int = OPTUNA_N_TRIALS, 
                 timeout: Optional[int] = OPTUNA_TIMEOUT,
                 verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            verbose: Whether to show progress
        
        Returns:
            Tuple of (best_params, best_score)
        """
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        
        # Optimize
        if verbose:
            print(f"\nOptimizing {self.model_name} hyperparameters...")
        
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            timeout=timeout,
            show_progress_bar=verbose
        )
        
        if verbose:
            print(f"Best {self.model_name} AUC: {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value


class DLOptunaOptimizer:
    """Optuna optimizer for DL models"""
    
    def __init__(self, model_name: str, model_class: type,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 input_length: int, epochs: int = 30):
        """
        Initialize optimizer
        
        Args:
            model_name: Name of the model (CNN1D, LSTM, CNN_LSTM)
            model_class: Model class
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            input_length: Length of input sequences
            epochs: Number of epochs for training
        """
        self.model_name = model_name
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_length = input_length
        self.epochs = epochs
        self.param_space = DL_PARAM_SPACE.get(model_name, {})
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters based on the model type"""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple):
                if len(param_config) == 2:
                    # Integer parameter
                    params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                elif len(param_config) == 3:
                    if param_config[2] == 'log':
                        # Float parameter with log scale
                        params[param_name] = trial.suggest_float(
                            param_name, param_config[0], param_config[1], log=True
                        )
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(
                            param_name, param_config[0], param_config[1]
                        )
        
        return params
    
    def _create_model_kwargs(self, params: Dict) -> Dict:
        """Create model kwargs from suggested parameters"""
        model_kwargs = {'input_length': self.input_length}
        
        # Model-specific parameter mapping
        if self.model_name == 'CNN1D':
            if 'dropout' in params:
                model_kwargs['dropout'] = params['dropout']
        elif self.model_name == 'LSTM':
            if 'hidden_size' in params:
                model_kwargs['hidden_size'] = params['hidden_size']
            if 'num_layers' in params:
                model_kwargs['num_layers'] = params['num_layers']
            if 'dropout' in params:
                model_kwargs['dropout'] = params['dropout']
        elif self.model_name == 'CNN_LSTM':
            if 'lstm_hidden' in params:
                model_kwargs['lstm_hidden'] = params['lstm_hidden']
            if 'lstm_layers' in params:
                model_kwargs['lstm_layers'] = params['lstm_layers']
            if 'dropout' in params:
                model_kwargs['dropout'] = params['dropout']
        
        return model_kwargs
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        # Extract training hyperparameters
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', BATCH_SIZE)
        
        # Create model kwargs
        model_kwargs = self._create_model_kwargs(params)
        
        # Create and train model
        model = DLModelWrapper(self.model_class, f"{self.model_name}_optuna", **model_kwargs)
        model.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=self.epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=False
        )
        
        # Evaluate on validation set
        metrics = model.evaluate(self.X_val, self.y_val)
        
        # Return metric to optimize (AUC)
        return metrics['auc']
    
    def optimize(self, n_trials: int = OPTUNA_N_TRIALS,
                 timeout: Optional[int] = OPTUNA_TIMEOUT,
                 verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            verbose: Whether to show progress
        
        Returns:
            Tuple of (best_params, best_score)
        """
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        
        # Optimize
        if verbose:
            print(f"\nOptimizing {self.model_name} hyperparameters...")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose
        )
        
        if verbose:
            print(f"Best {self.model_name} AUC: {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value


def optimize_ml_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       groups: Optional[np.ndarray] = None,
                       models_to_optimize: Optional[list] = None,
                       n_trials: int = OPTUNA_N_TRIALS,
                       verbose: bool = True) -> Dict[str, Dict]:
    """
    Optimize hyperparameters for all ML models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        groups: Group labels for cross-validation
        models_to_optimize: List of model names to optimize (None for all)
        n_trials: Number of trials per model
        verbose: Whether to show progress
    
    Returns:
        Dictionary of {model_name: {'params': best_params, 'score': best_score}}
    """
    if models_to_optimize is None:
        models_to_optimize = ['SVM', 'RandomForest', 'GradientBoosting', 'XGBoost']
    
    results = {}
    
    for model_name in models_to_optimize:
        try:
            optimizer = MLOptunaOptimizer(
                model_name, X_train, y_train, X_val, y_val, groups
            )
            best_params, best_score = optimizer.optimize(n_trials=n_trials, verbose=verbose)
            results[model_name] = {
                'params': best_params,
                'score': best_score
            }
        except Exception as e:
            if verbose:
                print(f"Error optimizing {model_name}: {e}")
            results[model_name] = {
                'params': {},
                'score': 0.0
            }
    
    return results


def optimize_dl_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       input_length: int,
                       models_to_optimize: Optional[list] = None,
                       n_trials: int = OPTUNA_N_TRIALS,
                       epochs: int = 30,
                       verbose: bool = True) -> Dict[str, Dict]:
    """
    Optimize hyperparameters for all DL models
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        input_length: Length of input sequences
        models_to_optimize: List of model names to optimize (None for all)
        n_trials: Number of trials per model
        epochs: Number of epochs for training each trial
        verbose: Whether to show progress
    
    Returns:
        Dictionary of {model_name: {'params': best_params, 'score': best_score}}
    """
    if models_to_optimize is None:
        models_to_optimize = ['CNN1D', 'LSTM', 'CNN_LSTM']
    
    model_classes = {
        'CNN1D': CNN1D,
        'LSTM': LSTMModel,
        'CNN_LSTM': CNNLSTM
    }
    
    results = {}
    
    for model_name in models_to_optimize:
        try:
            model_class = model_classes.get(model_name)
            if model_class is None:
                if verbose:
                    print(f"Unknown model: {model_name}")
                continue
            
            optimizer = DLOptunaOptimizer(
                model_name, model_class,
                X_train, y_train, X_val, y_val,
                input_length, epochs
            )
            best_params, best_score = optimizer.optimize(n_trials=n_trials, verbose=verbose)
            results[model_name] = {
                'params': best_params,
                'score': best_score
            }
        except Exception as e:
            if verbose:
                print(f"Error optimizing {model_name}: {e}")
            results[model_name] = {
                'params': {},
                'score': 0.0
            }
    
    return results


if __name__ == "__main__":
    # Test hyperparameter optimization
    print("Testing hyperparameter optimization...")
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    X_train = np.random.randn(500, 20)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_val = np.random.randn(200, 20)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
    
    # Test ML optimization (with fewer trials for testing)
    print("\nTesting ML optimization...")
    ml_results = optimize_ml_models(
        X_train, y_train, X_val, y_val,
        models_to_optimize=['RandomForest'],
        n_trials=5
    )
    print(f"ML Results: {ml_results}")
    
    print("\nHyperparameter optimization test completed!")
