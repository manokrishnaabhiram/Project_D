"""
Configuration settings for Sleep Apnea Detection project
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_set" / "apnea-ecg-database-1.0.0"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Signal parameters
SAMPLING_RATE = 100  # Hz
SEGMENT_DURATION = 60  # seconds (1 minute)
SEGMENT_SAMPLES = SAMPLING_RATE * SEGMENT_DURATION  # 6000 samples per segment

# Preprocessing parameters
BANDPASS_LOW = 0.5  # Hz
BANDPASS_HIGH = 40  # Hz
FILTER_ORDER = 4

# Training records (with annotations)
TRAIN_RECORDS = [
    # Apnea patients (Class A - high AHI)
    'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
    'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
    # Borderline cases (Class B)
    'b01', 'b02', 'b03', 'b04', 'b05',
    # Control/Normal subjects (Class C)
    'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10'
]

# Test records (for final evaluation)
TEST_RECORDS = [f'x{i:02d}' for i in range(1, 36)]  # x01 to x35

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Deep Learning parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Device for PyTorch
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
