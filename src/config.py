"""
Configuration settings for the Spam Email Detection project.
Centralizes hyperparameters, model parameters, and file paths.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data Generation Config
NUM_SAMPLES = 300           # Total synthetic emails to generate
RANDOM_STATE = 42           # For reproducibility

# Text Preprocessing & Vectorization Config
MAX_TOKENS = 10000          # Vocabulary size
SEQUENCE_LENGTH = 200       # Maximum length of input sequences
BATCH_SIZE = 32             # Batch size for tf.data.Dataset

# Model Architecture Config
EMBEDDING_DIM = 64          # Dimensions of the embedding layer
LSTM_UNITS = 64             # Number of units in the Bi-LSTM layer
DENSE_UNITS = 32            # Number of units in the hidden dense layer
DROPOUT_RATE = 0.3          # Dropout rate for regularization

# Training Config
EPOCHS = 15                 # Maximum number of epochs
LEARNING_RATE = 1e-3        # Adam optimizer learning rate
PATIENCE = 3                # Early stopping patience

# File Paths
DATA_FILEPATH = DATA_DIR / "synthetic_emails.csv"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "spam_detector.keras"
VECTORIZER_PATH = MODELS_DIR / "text_vectorizer.pkl"