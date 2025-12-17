"""
Preprocessing module for ECG signals
Includes filtering, normalization, and signal cleaning
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional

from .config import SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple:
    """
    Design Butterworth bandpass filter
    
    Args:
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float = BANDPASS_LOW, 
                    highcut: float = BANDPASS_HIGH, fs: float = SAMPLING_RATE, 
                    order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply bandpass filter to ECG signal
    
    Args:
        data: Input signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def notch_filter(data: np.ndarray, freq: float = 50.0, 
                 fs: float = SAMPLING_RATE, Q: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove power line interference
    
    Args:
        data: Input signal
        freq: Frequency to remove (50Hz or 60Hz)
        fs: Sampling frequency (Hz)
        Q: Quality factor
    
    Returns:
        Filtered signal
    """
    b, a = iirnotch(freq, Q, fs)
    y = filtfilt(b, a, data)
    return y


def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize ECG signal
    
    Args:
        data: Input signal (can be 1D or 2D)
        method: Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized signal
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    normalized = np.zeros_like(data, dtype=np.float32)
    
    for i in range(data.shape[0]):
        segment = data[i]
        
        if method == 'zscore':
            mean = np.mean(segment)
            std = np.std(segment)
            if std > 0:
                normalized[i] = (segment - mean) / std
            else:
                normalized[i] = segment - mean
                
        elif method == 'minmax':
            min_val = np.min(segment)
            max_val = np.max(segment)
            if max_val - min_val > 0:
                normalized[i] = (segment - min_val) / (max_val - min_val)
            else:
                normalized[i] = np.zeros_like(segment)
                
        elif method == 'robust':
            median = np.median(segment)
            q75, q25 = np.percentile(segment, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized[i] = (segment - median) / iqr
            else:
                normalized[i] = segment - median
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    if squeeze:
        return normalized.squeeze()
    return normalized


def remove_baseline_wander(data: np.ndarray, fs: float = SAMPLING_RATE, 
                           cutoff: float = 0.5) -> np.ndarray:
    """
    Remove baseline wander using highpass filter
    
    Args:
        data: Input signal
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency for highpass filter (Hz)
    
    Returns:
        Signal with baseline removed
    """
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype='high')
    y = filtfilt(b, a, data)
    return y


def preprocess_signal(data: np.ndarray, fs: float = SAMPLING_RATE,
                      apply_bandpass: bool = True,
                      apply_notch: bool = True,
                      apply_normalize: bool = True,
                      normalize_method: str = 'zscore') -> np.ndarray:
    """
    Apply full preprocessing pipeline to ECG signal
    
    Args:
        data: Input signal (1D or 2D array)
        fs: Sampling frequency (Hz)
        apply_bandpass: Whether to apply bandpass filter
        apply_notch: Whether to apply notch filter
        apply_normalize: Whether to normalize
        normalize_method: Normalization method
    
    Returns:
        Preprocessed signal
    """
    if data.ndim == 1:
        # Single segment
        processed = data.copy().astype(np.float64)
        
        if apply_bandpass:
            processed = bandpass_filter(processed, fs=fs)
        
        if apply_notch:
            processed = notch_filter(processed, fs=fs)
        
        if apply_normalize:
            processed = normalize_signal(processed, method=normalize_method)
            
    else:
        # Multiple segments
        processed = np.zeros_like(data, dtype=np.float64)
        
        for i in range(data.shape[0]):
            segment = data[i].copy()
            
            if apply_bandpass:
                segment = bandpass_filter(segment, fs=fs)
            
            if apply_notch:
                segment = notch_filter(segment, fs=fs)
            
            processed[i] = segment
        
        if apply_normalize:
            processed = normalize_signal(processed, method=normalize_method)
    
    return processed.astype(np.float32)


def preprocess_batch(segments: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Preprocess a batch of ECG segments efficiently
    
    Args:
        segments: Array of shape (n_segments, segment_length)
        fs: Sampling frequency (Hz)
    
    Returns:
        Preprocessed segments
    """
    return preprocess_signal(segments, fs=fs)


if __name__ == "__main__":
    # Test preprocessing
    import matplotlib.pyplot as plt
    
    # Generate test signal with noise
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 1 * t)  # 1 Hz signal
    noise = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    # Apply preprocessing
    filtered = preprocess_signal(noisy_signal, fs=100)
    
    print("Preprocessing test completed successfully")
