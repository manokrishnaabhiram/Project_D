"""
Feature extraction module for ECG-based sleep apnea detection
Extracts HRV features, time-domain features, and frequency-domain features
"""
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import warnings

from .config import SAMPLING_RATE, SEGMENT_SAMPLES


def detect_r_peaks(ecg_segment: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Detect R-peaks in ECG segment using Pan-Tompkins-like algorithm
    
    Args:
        ecg_segment: ECG signal segment
        fs: Sampling frequency (Hz)
    
    Returns:
        r_peaks: Indices of R-peaks
    """
    # Differentiate
    diff_ecg = np.diff(ecg_segment)
    
    # Square
    squared = diff_ecg ** 2
    
    # Moving window integration
    window_size = int(0.15 * fs)  # 150ms window
    if window_size < 1:
        window_size = 1
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # Find peaks
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    min_distance = int(0.2 * fs)  # Minimum 200ms between peaks (max 300 bpm)
    
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
    
    # Refine peak locations to actual R-peaks in original signal
    refined_peaks = []
    search_window = int(0.05 * fs)  # 50ms search window
    
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(ecg_segment), peak + search_window)
        local_max = np.argmax(ecg_segment[start:end]) + start
        refined_peaks.append(local_max)
    
    return np.array(refined_peaks)


def compute_rr_intervals(r_peaks: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Compute RR intervals from R-peak locations
    
    Args:
        r_peaks: Indices of R-peaks
        fs: Sampling frequency (Hz)
    
    Returns:
        rr_intervals: RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000  # Convert to milliseconds
    
    return rr_ms


def extract_hrv_time_domain(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract time-domain HRV features
    
    Args:
        rr_intervals: RR intervals in milliseconds
    
    Returns:
        Dictionary of HRV features
    """
    features = {}
    
    if len(rr_intervals) < 2:
        # Return NaN for all features if not enough RR intervals
        return {
            'mean_rr': np.nan, 'std_rr': np.nan, 'mean_hr': np.nan, 'std_hr': np.nan,
            'rmssd': np.nan, 'sdnn': np.nan, 'nn50': np.nan, 'pnn50': np.nan,
            'nn20': np.nan, 'pnn20': np.nan, 'cv_rr': np.nan, 'median_rr': np.nan,
            'range_rr': np.nan, 'iqr_rr': np.nan
        }
    
    # Basic statistics
    features['mean_rr'] = np.mean(rr_intervals)
    features['std_rr'] = np.std(rr_intervals)
    features['median_rr'] = np.median(rr_intervals)
    features['range_rr'] = np.max(rr_intervals) - np.min(rr_intervals)
    features['iqr_rr'] = np.percentile(rr_intervals, 75) - np.percentile(rr_intervals, 25)
    
    # Heart rate statistics
    hr = 60000 / rr_intervals  # BPM
    features['mean_hr'] = np.mean(hr)
    features['std_hr'] = np.std(hr)
    
    # RMSSD - Root mean square of successive differences
    rr_diff = np.diff(rr_intervals)
    features['rmssd'] = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else np.nan
    
    # SDNN - Standard deviation of NN intervals
    features['sdnn'] = np.std(rr_intervals)
    
    # NN50 and pNN50
    nn50 = np.sum(np.abs(rr_diff) > 50)
    features['nn50'] = nn50
    features['pnn50'] = (nn50 / len(rr_diff) * 100) if len(rr_diff) > 0 else 0
    
    # NN20 and pNN20
    nn20 = np.sum(np.abs(rr_diff) > 20)
    features['nn20'] = nn20
    features['pnn20'] = (nn20 / len(rr_diff) * 100) if len(rr_diff) > 0 else 0
    
    # Coefficient of variation
    features['cv_rr'] = (features['std_rr'] / features['mean_rr'] * 100) if features['mean_rr'] > 0 else np.nan
    
    return features


def extract_hrv_frequency_domain(rr_intervals: np.ndarray, fs_resample: float = 4.0) -> Dict[str, float]:
    """
    Extract frequency-domain HRV features
    
    Args:
        rr_intervals: RR intervals in milliseconds
        fs_resample: Resampling frequency for spectral analysis
    
    Returns:
        Dictionary of frequency-domain HRV features
    """
    features = {}
    
    if len(rr_intervals) < 10:
        return {
            'vlf_power': np.nan, 'lf_power': np.nan, 'hf_power': np.nan,
            'total_power': np.nan, 'lf_hf_ratio': np.nan, 'lf_norm': np.nan, 'hf_norm': np.nan
        }
    
    try:
        # Create time array for RR intervals
        rr_times = np.cumsum(rr_intervals) / 1000  # Convert to seconds
        rr_times = rr_times - rr_times[0]  # Start from 0
        
        # Interpolate to uniform sampling
        if rr_times[-1] < 1:
            return {
                'vlf_power': np.nan, 'lf_power': np.nan, 'hf_power': np.nan,
                'total_power': np.nan, 'lf_hf_ratio': np.nan, 'lf_norm': np.nan, 'hf_norm': np.nan
            }
        
        interp_func = interp1d(rr_times, rr_intervals, kind='cubic', fill_value='extrapolate')
        t_uniform = np.arange(0, rr_times[-1], 1/fs_resample)
        rr_uniform = interp_func(t_uniform)
        
        # Remove mean
        rr_uniform = rr_uniform - np.mean(rr_uniform)
        
        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(rr_uniform, fs=fs_resample, nperseg=min(256, len(rr_uniform)))
        
        # Define frequency bands
        vlf_band = (0.003, 0.04)  # Very Low Frequency
        lf_band = (0.04, 0.15)    # Low Frequency
        hf_band = (0.15, 0.4)     # High Frequency
        
        # Calculate band powers
        vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
        lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
        
        features['vlf_power'] = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
        features['lf_power'] = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        features['hf_power'] = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        features['total_power'] = features['vlf_power'] + features['lf_power'] + features['hf_power']
        
        # LF/HF ratio
        features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else np.nan
        
        # Normalized powers
        lf_hf_sum = features['lf_power'] + features['hf_power']
        features['lf_norm'] = (features['lf_power'] / lf_hf_sum * 100) if lf_hf_sum > 0 else np.nan
        features['hf_norm'] = (features['hf_power'] / lf_hf_sum * 100) if lf_hf_sum > 0 else np.nan
        
    except Exception as e:
        return {
            'vlf_power': np.nan, 'lf_power': np.nan, 'hf_power': np.nan,
            'total_power': np.nan, 'lf_hf_ratio': np.nan, 'lf_norm': np.nan, 'hf_norm': np.nan
        }
    
    return features


def extract_ecg_morphology_features(ecg_segment: np.ndarray, fs: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract ECG morphology/waveform features
    
    Args:
        ecg_segment: ECG signal segment
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of morphology features
    """
    features = {}
    
    # Statistical features
    features['ecg_mean'] = np.mean(ecg_segment)
    features['ecg_std'] = np.std(ecg_segment)
    features['ecg_var'] = np.var(ecg_segment)
    features['ecg_min'] = np.min(ecg_segment)
    features['ecg_max'] = np.max(ecg_segment)
    features['ecg_range'] = features['ecg_max'] - features['ecg_min']
    features['ecg_median'] = np.median(ecg_segment)
    features['ecg_iqr'] = np.percentile(ecg_segment, 75) - np.percentile(ecg_segment, 25)
    
    # Higher-order statistics
    features['ecg_skewness'] = stats.skew(ecg_segment)
    features['ecg_kurtosis'] = stats.kurtosis(ecg_segment)
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(ecg_segment)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_segment)
    
    # Energy features
    features['ecg_energy'] = np.sum(ecg_segment ** 2)
    features['ecg_rms'] = np.sqrt(np.mean(ecg_segment ** 2))
    
    # Entropy (sample entropy approximation)
    features['ecg_entropy'] = _sample_entropy(ecg_segment, m=2, r=0.2)
    
    # Spectral features of the ECG
    freqs = fftfreq(len(ecg_segment), 1/fs)
    fft_vals = np.abs(fft(ecg_segment))
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Spectral centroid
    if np.sum(pos_fft) > 0:
        features['spectral_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
    else:
        features['spectral_centroid'] = 0
    
    # Spectral bandwidth
    if np.sum(pos_fft) > 0:
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((pos_freqs - features['spectral_centroid']) ** 2) * pos_fft) / np.sum(pos_fft)
        )
    else:
        features['spectral_bandwidth'] = 0
    
    return features


def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy of a signal (simplified version)
    
    Args:
        signal: Input signal
        m: Embedding dimension
        r: Tolerance (fraction of std)
    
    Returns:
        Sample entropy value
    """
    try:
        N = len(signal)
        r = r * np.std(signal)
        
        if r == 0 or N < m + 1:
            return np.nan
        
        # Simplified calculation
        def count_matches(templates, r):
            count = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count
        
        # Subsample for efficiency
        step = max(1, N // 500)
        indices = np.arange(0, N - m, step)
        
        templates_m = np.array([signal[i:i+m] for i in indices if i + m <= N])
        templates_m1 = np.array([signal[i:i+m+1] for i in indices if i + m + 1 <= N])
        
        if len(templates_m) < 2 or len(templates_m1) < 2:
            return np.nan
        
        A = count_matches(templates_m1, r)
        B = count_matches(templates_m, r)
        
        if B == 0 or A == 0:
            return np.nan
        
        return -np.log(A / B)
    except:
        return np.nan


def extract_respiratory_features(ecg_segment: np.ndarray, r_peaks: np.ndarray, 
                                  fs: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract ECG-derived respiratory (EDR) features
    
    Args:
        ecg_segment: ECG signal segment
        r_peaks: Indices of R-peaks
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of respiratory features
    """
    features = {}
    
    if len(r_peaks) < 3:
        return {
            'edr_mean': np.nan, 'edr_std': np.nan, 'edr_range': np.nan,
            'resp_rate_est': np.nan
        }
    
    try:
        # R-peak amplitude modulation (ECG-derived respiration)
        r_amplitudes = ecg_segment[r_peaks]
        
        features['edr_mean'] = np.mean(r_amplitudes)
        features['edr_std'] = np.std(r_amplitudes)
        features['edr_range'] = np.max(r_amplitudes) - np.min(r_amplitudes)
        
        # Estimate respiratory rate from R-peak amplitude modulation
        if len(r_amplitudes) > 10:
            # Interpolate R-peak amplitudes
            r_times = r_peaks / fs
            interp_func = interp1d(r_times, r_amplitudes, kind='linear', fill_value='extrapolate')
            t_uniform = np.linspace(r_times[0], r_times[-1], 100)
            amp_uniform = interp_func(t_uniform)
            
            # Find peaks in amplitude modulation
            peaks, _ = signal.find_peaks(amp_uniform)
            if len(peaks) >= 2:
                resp_rate = len(peaks) / (t_uniform[-1] - t_uniform[0]) * 60  # breaths per minute
                features['resp_rate_est'] = resp_rate
            else:
                features['resp_rate_est'] = np.nan
        else:
            features['resp_rate_est'] = np.nan
            
    except Exception as e:
        features['edr_mean'] = np.nan
        features['edr_std'] = np.nan
        features['edr_range'] = np.nan
        features['resp_rate_est'] = np.nan
    
    return features


def extract_all_features(ecg_segment: np.ndarray, fs: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract all features from an ECG segment
    
    Args:
        ecg_segment: ECG signal segment
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_segment, fs)
    
    # Compute RR intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)
    
    # Extract HRV time-domain features
    hrv_time = extract_hrv_time_domain(rr_intervals)
    features.update(hrv_time)
    
    # Extract HRV frequency-domain features
    hrv_freq = extract_hrv_frequency_domain(rr_intervals)
    features.update(hrv_freq)
    
    # Extract ECG morphology features
    morph = extract_ecg_morphology_features(ecg_segment, fs)
    features.update(morph)
    
    # Extract respiratory features
    resp = extract_respiratory_features(ecg_segment, r_peaks, fs)
    features.update(resp)
    
    # Add R-peak count
    features['n_r_peaks'] = len(r_peaks)
    features['n_rr_intervals'] = len(rr_intervals)
    
    return features


def extract_features_batch(segments: np.ndarray, fs: float = SAMPLING_RATE, 
                           verbose: bool = True) -> np.ndarray:
    """
    Extract features from multiple ECG segments
    
    Args:
        segments: Array of shape (n_segments, segment_length)
        fs: Sampling frequency (Hz)
        verbose: Whether to show progress
    
    Returns:
        Feature matrix of shape (n_segments, n_features)
    """
    from tqdm import tqdm
    
    all_features = []
    feature_names = None
    
    iterator = tqdm(range(len(segments)), desc="Extracting features") if verbose else range(len(segments))
    
    for i in iterator:
        features = extract_all_features(segments[i], fs)
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        feature_vector = [features[name] for name in feature_names]
        all_features.append(feature_vector)
    
    feature_matrix = np.array(all_features, dtype=np.float32)
    
    # Handle NaN values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col_means = np.nanmean(feature_matrix, axis=0)
        nan_indices = np.where(np.isnan(feature_matrix))
        feature_matrix[nan_indices] = np.take(col_means, nan_indices[1])
    
    # Replace any remaining NaN with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    return feature_matrix, feature_names


def get_feature_names() -> List[str]:
    """Get list of all feature names"""
    # Create a dummy segment to extract feature names
    dummy_segment = np.random.randn(SEGMENT_SAMPLES)
    features = extract_all_features(dummy_segment)
    return list(features.keys())


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")
    
    # Generate test ECG-like signal
    t = np.linspace(0, 60, SEGMENT_SAMPLES)
    test_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(SEGMENT_SAMPLES)
    
    features = extract_all_features(test_ecg)
    print(f"Number of features: {len(features)}")
    print(f"Feature names: {list(features.keys())}")
