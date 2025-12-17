"""
Data loading module for Apnea-ECG Database
Handles reading ECG signals and annotations using wfdb library
"""
import numpy as np
import wfdb
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

from .config import DATA_DIR, SAMPLING_RATE, SEGMENT_SAMPLES, TRAIN_RECORDS, TEST_RECORDS


def load_record(record_name: str, data_dir: Path = DATA_DIR) -> Tuple[np.ndarray, dict]:
    """
    Load ECG signal from a record
    
    Args:
        record_name: Name of the record (e.g., 'a01')
        data_dir: Path to the data directory
    
    Returns:
        signal: ECG signal array
        info: Dictionary with record information
    """
    record_path = str(data_dir / record_name)
    record = wfdb.rdrecord(record_path)
    
    signal = record.p_signal.flatten()  # Single-lead ECG
    info = {
        'fs': record.fs,
        'n_samples': len(signal),
        'duration_minutes': len(signal) / (record.fs * 60),
        'record_name': record_name
    }
    
    return signal, info


def load_annotations(record_name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    """
    Load apnea annotations for a record
    
    Args:
        record_name: Name of the record (e.g., 'a01')
        data_dir: Path to the data directory
    
    Returns:
        labels: Array of labels (1 for apnea, 0 for normal) per minute
    """
    ann_path = str(data_dir / record_name)
    
    try:
        ann = wfdb.rdann(ann_path, 'apn')
        
        # Convert annotations to per-minute labels
        # 'A' = Apnea, 'N' = Normal
        labels = []
        for symbol in ann.symbol:
            if symbol == 'A':
                labels.append(1)  # Apnea
            elif symbol == 'N':
                labels.append(0)  # Normal
            # Skip other symbols like 'X' (uncertain)
        
        return np.array(labels)
    except Exception as e:
        print(f"Warning: Could not load annotations for {record_name}: {e}")
        return None


def load_qrs_annotations(record_name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    """
    Load QRS (R-peak) annotations for RR interval calculation
    
    Args:
        record_name: Name of the record (e.g., 'a01')
        data_dir: Path to the data directory
    
    Returns:
        r_peaks: Array of R-peak sample indices
    """
    ann_path = str(data_dir / record_name)
    
    try:
        ann = wfdb.rdann(ann_path, 'qrs')
        return ann.sample
    except Exception as e:
        print(f"Warning: Could not load QRS annotations for {record_name}: {e}")
        return None


def segment_signal(signal: np.ndarray, labels: np.ndarray, 
                   segment_samples: int = SEGMENT_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment ECG signal into 1-minute segments with corresponding labels
    
    Args:
        signal: Full ECG signal
        labels: Per-minute labels
        segment_samples: Number of samples per segment (6000 for 1 min at 100Hz)
    
    Returns:
        segments: Array of shape (n_segments, segment_samples)
        segment_labels: Array of labels for each segment
    """
    n_segments = min(len(signal) // segment_samples, len(labels))
    
    segments = []
    segment_labels = []
    
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = signal[start:end]
        
        if len(segment) == segment_samples:
            segments.append(segment)
            segment_labels.append(labels[i])
    
    return np.array(segments), np.array(segment_labels)


def load_dataset(record_list: List[str], 
                 data_dir: Path = DATA_DIR,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load complete dataset from a list of records
    
    Args:
        record_list: List of record names to load
        data_dir: Path to the data directory
        verbose: Whether to show progress bar
    
    Returns:
        X: Array of segments (n_total_segments, segment_samples)
        y: Array of labels
        record_ids: List of record names for each segment (for subject-wise CV)
    """
    all_segments = []
    all_labels = []
    record_ids = []
    
    iterator = tqdm(record_list, desc="Loading records") if verbose else record_list
    
    for record_name in iterator:
        try:
            # Load signal and annotations
            signal, info = load_record(record_name, data_dir)
            labels = load_annotations(record_name, data_dir)
            
            if labels is None:
                continue
            
            # Segment the signal
            segments, seg_labels = segment_signal(signal, labels)
            
            all_segments.append(segments)
            all_labels.append(seg_labels)
            record_ids.extend([record_name] * len(segments))
            
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
            continue
    
    if len(all_segments) == 0:
        raise ValueError("No data loaded successfully")
    
    X = np.vstack(all_segments)
    y = np.concatenate(all_labels)
    
    return X, y, record_ids


def load_training_data(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load training dataset (a01-a20, b01-b05, c01-c10)"""
    return load_dataset(TRAIN_RECORDS, verbose=verbose)


def load_test_data(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test dataset (x01-x35)"""
    return load_dataset(TEST_RECORDS, verbose=verbose)


def get_dataset_info() -> Dict:
    """Get information about the dataset"""
    info = {
        'train_records': TRAIN_RECORDS,
        'test_records': TEST_RECORDS,
        'sampling_rate': SAMPLING_RATE,
        'segment_duration_sec': 60,
        'segment_samples': SEGMENT_SAMPLES
    }
    return info


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    X_train, y_train, record_ids = load_training_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Class distribution: Normal={np.sum(y_train==0)}, Apnea={np.sum(y_train==1)}")
