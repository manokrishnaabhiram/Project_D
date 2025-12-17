"""
Training and evaluation utilities for sleep apnea detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from .config import RANDOM_STATE, RESULTS_DIR, MODELS_DIR


def subject_wise_split(X: np.ndarray, y: np.ndarray, record_ids: List[str],
                       test_size: float = 0.2, val_size: float = 0.1,
                       random_state: int = RANDOM_STATE) -> Tuple:
    """
    Split data by subjects to prevent data leakage
    
    Args:
        X: Feature matrix or raw segments
        y: Labels
        record_ids: Subject/record identifiers for each sample
        test_size: Fraction of subjects for test set
        val_size: Fraction of subjects for validation set
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids
    """
    unique_records = list(set(record_ids))
    np.random.seed(random_state)
    np.random.shuffle(unique_records)
    
    n_records = len(unique_records)
    n_test = max(1, int(n_records * test_size))
    n_val = max(1, int(n_records * val_size))
    
    test_records = set(unique_records[:n_test])
    val_records = set(unique_records[n_test:n_test + n_val])
    train_records = set(unique_records[n_test + n_val:])
    
    # Create masks
    record_ids_array = np.array(record_ids)
    train_mask = np.isin(record_ids_array, list(train_records))
    val_mask = np.isin(record_ids_array, list(val_records))
    test_mask = np.isin(record_ids_array, list(test_records))
    
    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask],
        record_ids_array[train_mask].tolist(),
        record_ids_array[val_mask].tolist(),
        record_ids_array[test_mask].tolist()
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, save_path: Optional[Path] = None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Apnea'],
                yticklabels=['Normal', 'Apnea'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   model_name: str, save_path: Optional[Path] = None):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: Dict, model_name: str, 
                          save_path: Optional[Path] = None):
    """Plot training history for DL models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'accuracy',
                          save_path: Optional[Path] = None):
    """Plot comparison of all models"""
    model_names = []
    values = []
    
    for name, result in results.items():
        model_names.append(name)
        if 'test_metrics' in result:
            values.append(result['test_metrics'].get(metric, 0))
        else:
            values.append(0)
    
    # Sort by value
    sorted_indices = np.argsort(values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = plt.barh(model_names, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)
    
    plt.xlabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xlim([0, max(values) * 1.15])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_results_summary(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a summary DataFrame of all results"""
    rows = []
    
    for name, result in results.items():
        if 'test_metrics' in result:
            row = {'Model': name}
            row.update({f'Test_{k}': v for k, v in result['test_metrics'].items()})
            
            if 'cv_metrics' in result:
                for k, v in result['cv_metrics'].items():
                    if not k.endswith('_std'):
                        row[f'CV_{k}'] = v
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by test accuracy
    if 'Test_accuracy' in df.columns:
        df = df.sort_values('Test_accuracy', ascending=False)
    
    return df


def save_results(results: Dict[str, Dict], experiment_name: str = None):
    """Save all results to files"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_dir = RESULTS_DIR / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    summary_df = create_results_summary(results)
    summary_df.to_csv(results_dir / 'results_summary.csv', index=False)
    
    # Save detailed metrics as JSON
    metrics_dict = {}
    for name, result in results.items():
        if 'test_metrics' in result:
            metrics_dict[name] = {
                'test_metrics': result['test_metrics'],
                'cv_metrics': result.get('cv_metrics', {})
            }
    
    with open(results_dir / 'detailed_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Generate plots
    plot_model_comparison(results, 'accuracy', results_dir / 'comparison_accuracy.png')
    plot_model_comparison(results, 'auc', results_dir / 'comparison_auc.png')
    plot_model_comparison(results, 'f1', results_dir / 'comparison_f1.png')
    
    print(f"\nResults saved to: {results_dir}")
    
    return results_dir


def print_results_table(results: Dict[str, Dict]):
    """Print formatted results table"""
    summary_df = create_results_summary(results)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Format for printing
    metrics_to_show = ['Model', 'Test_accuracy', 'Test_precision', 'Test_recall', 
                       'Test_specificity', 'Test_f1', 'Test_auc']
    available_metrics = [m for m in metrics_to_show if m in summary_df.columns]
    
    print(summary_df[available_metrics].to_string(index=False))
    print("="*80)
    
    # Best model
    if 'Test_accuracy' in summary_df.columns:
        best_model = summary_df.iloc[0]['Model']
        best_acc = summary_df.iloc[0]['Test_accuracy']
        best_auc = summary_df.iloc[0].get('Test_auc', 'N/A')
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {best_acc:.4f}")
        print(f"   AUC: {best_auc:.4f}" if isinstance(best_auc, float) else f"   AUC: {best_auc}")


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str) -> str:
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Apnea'],
                                   digits=4)
    
    header = f"\n{'='*50}\n{model_name} - Classification Report\n{'='*50}\n"
    return header + report


class ExperimentLogger:
    """Logger for tracking experiments"""
    
    def __init__(self, experiment_name: str = None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.log_dir = RESULTS_DIR / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'experiment.log'
        self.results = {}
    
    def log(self, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def log_metrics(self, model_name: str, metrics: Dict[str, float], phase: str = 'test'):
        """Log metrics for a model"""
        self.log(f"{model_name} - {phase} metrics:")
        for name, value in metrics.items():
            self.log(f"  {name}: {value:.4f}")
        
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][f'{phase}_metrics'] = metrics
    
    def save(self):
        """Save experiment results"""
        save_results(self.results, self.experiment_name)


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")
    
    # Generate dummy data
    np.random.seed(RANDOM_STATE)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    
    # Test plotting
    plot_confusion_matrix(y_true, y_pred, 'Test Model', RESULTS_DIR / 'test_cm.png')
    plot_roc_curve(y_true, y_proba, 'Test Model', RESULTS_DIR / 'test_roc.png')
    
    print("Utilities test completed!")
