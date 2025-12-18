"""
Deep Learning models for sleep apnea detection
Includes 1D-CNN, LSTM, and CNN-LSTM hybrid architectures
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import copy

from .config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    EARLY_STOPPING_PATIENCE, SEGMENT_SAMPLES, RANDOM_STATE, MODELS_DIR
)


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG segments"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
        # Add channel dimension if needed
        if self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # (N, 1, L)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for ECG classification"""
    
    def __init__(self, input_length: int = SEGMENT_SAMPLES, num_classes: int = 2,
                 dropout: float = 0.5):
        super(CNN1D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Block 5
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class LSTMModel(nn.Module):
    """LSTM model for ECG sequence classification"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.5,
                 bidirectional: bool = True):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, seq_len) -> need (batch, seq_len, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * num_directions)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.fc(context)
        return out


class CNNLSTM(nn.Module):
    """CNN-LSTM Hybrid model for ECG classification (Best performing)"""
    
    def __init__(self, input_length: int = SEGMENT_SAMPLES, num_classes: int = 2,
                 cnn_filters: List[int] = [32, 64, 128], lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.5):
        super(CNNLSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, cnn_filters[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, channels, seq_len)
        
        # Reshape for LSTM: (batch, seq_len, channels)
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, hidden*2)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.classifier(context)
        return out


class DLModelWrapper:
    """Wrapper class for deep learning models with training utilities"""
    
    def __init__(self, model_class, model_name: str, **model_kwargs):
        self.model_class = model_class
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = None
        self.device = DEVICE
        self.is_fitted = False
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def _create_model(self):
        """Create a new model instance"""
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        return self.model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
            learning_rate: float = LEARNING_RATE,
            class_weights: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'DLModelWrapper':
        """
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            class_weights: Optional class weights for imbalanced data
            verbose: Whether to show progress
        
        Returns:
            self
        """
        # Create model
        self._create_model()
        
        # Create data loaders
        train_dataset = ECGDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = ECGDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Loss function with class weights
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        iterator = tqdm(range(epochs), desc=f"Training {self.model_name}") if verbose else range(epochs)
        
        for epoch in iterator:
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_loader(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
                
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        return self
    
    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on a data loader"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        return total_loss / total, correct / total
    
    def predict(self, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        dataset = ECGDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        dataset = ECGDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = softmax(outputs)
                probabilities.append(probs.cpu().numpy())
        
        return np.vstack(probabilities)
    
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
    
    def save(self, path: Optional[str] = None):
        """Save model to disk"""
        if path is None:
            path = MODELS_DIR / f"{self.model_name}_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_kwargs': self.model_kwargs,
            'history': self.history
        }, path)
    
    def load(self, path: Optional[str] = None):
        """Load model from disk"""
        if path is None:
            path = MODELS_DIR / f"{self.model_name}_model.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model_kwargs = checkpoint['model_kwargs']
        self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.is_fitted = True


def get_all_dl_models(input_length: int = SEGMENT_SAMPLES,
                      tuned_params: Optional[Dict[str, Dict]] = None) -> Dict[str, DLModelWrapper]:
    """
    Get dictionary of all DL models
    
    Args:
        input_length: Length of input ECG segments
        tuned_params: Optional dictionary of tuned hyperparameters per model
                     e.g., {'CNN1D': {'dropout': 0.3}, ...}
    """
    params = tuned_params or {}
    
    # Extract model-specific kwargs (architecture params)
    cnn_kwargs = {k: v for k, v in params.get('CNN1D', {}).items() 
                  if k in ['dropout']}
    lstm_kwargs = {k: v for k, v in params.get('LSTM', {}).items() 
                   if k in ['hidden_size', 'num_layers', 'dropout', 'bidirectional']}
    cnn_lstm_kwargs = {k: v for k, v in params.get('CNN_LSTM', {}).items() 
                       if k in ['lstm_hidden', 'lstm_layers', 'dropout']}
    
    # Handle cnn_filters for CNN_LSTM
    if 'cnn_filters_base' in params.get('CNN_LSTM', {}):
        base = params['CNN_LSTM']['cnn_filters_base']
        cnn_lstm_kwargs['cnn_filters'] = [base, base * 2, base * 4]
    
    return {
        'CNN1D': DLModelWrapper(CNN1D, 'CNN1D', input_length=input_length, **cnn_kwargs),
        'LSTM': DLModelWrapper(LSTMModel, 'LSTM', **lstm_kwargs),
        'CNN_LSTM': DLModelWrapper(CNNLSTM, 'CNN_LSTM', input_length=input_length, **cnn_lstm_kwargs)
    }


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute class weights for imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return weights


def train_and_evaluate_all_dl_models(X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray, y_val: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray,
                                      epochs: int = EPOCHS,
                                      verbose: bool = True,
                                      tuned_params: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    """
    Train and evaluate all DL models
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        X_test: Test data
        y_test: Test labels
        epochs: Number of training epochs
        verbose: Whether to show progress
        tuned_params: Optional dictionary of tuned hyperparameters per model
    
    Returns:
        Dictionary of results for each model
    """
    models = get_all_dl_models(input_length=X_train.shape[1], tuned_params=tuned_params)
    results = {}
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    for name, model in models.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
        
        # Get training hyperparameters from tuned_params if available
        train_kwargs = {}
        if tuned_params and name in tuned_params:
            params = tuned_params[name]
            if 'learning_rate' in params:
                train_kwargs['learning_rate'] = params['learning_rate']
            if 'batch_size' in params:
                train_kwargs['batch_size'] = params['batch_size']
        
        # Train model
        model.fit(X_train, y_train, X_val, y_val, 
                  epochs=epochs, class_weights=class_weights, verbose=verbose,
                  **train_kwargs)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        results[name] = {
            'model': model,
            'test_metrics': test_metrics,
            'history': model.history
        }
        
        if verbose:
            print(f"\n{name} Test Results:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall/Sensitivity: {test_metrics['recall']:.4f}")
            print(f"  Specificity: {test_metrics['specificity']:.4f}")
            print(f"  F1 Score: {test_metrics['f1']:.4f}")
            print(f"  AUC: {test_metrics['auc']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test DL models
    print(f"Using device: {DEVICE}")
    print("Testing DL models...")
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    n_samples = 500
    X = np.random.randn(n_samples, SEGMENT_SAMPLES).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
    
    # Test CNN-LSTM model
    model = DLModelWrapper(CNNLSTM, 'CNN_LSTM_test', input_length=SEGMENT_SAMPLES)
    model.fit(X_train, y_train, X_val, y_val, epochs=5, verbose=True)
    metrics = model.evaluate(X_test, y_test)
    print(f"\nCNN-LSTM Test metrics: {metrics}")
