"""
Configuration file for the DLBA (Deep Learning Based Analysis) project.
This file centralizes key settings, hyperparameters, and configuration values
used across multiple scripts in the project.
"""

import torch
import os
from datetime import datetime, timedelta

# =====================================================================
# Data Loader Settings
# =====================================================================
# Date range for historical data
END_DATE = "2025-02-28"  # Today's date
START_DATE = "2023-01-03"  # One year before end date

# Path to save/load stock data
CSV_PATH = os.path.join("data", "stock_data.csv")

# List of stock tickers to analyze
TICKERS = ['NVDA', 'AMD', 'INTC', 'AVGO', 'TSM', 'QCOM', 'TXN', 'MU', 'ASML', 'MRVL']

# Maximum number of retries for API calls
MAX_RETRIES = 3

# =====================================================================
# Sentiment Analysis Settings
# =====================================================================
# Pre-trained model for sentiment analysis
MODEL_NAME = "ProsusAI/finbert"

# Topics and associated keywords for topic-specific sentiment analysis
TOPICS = {
    'market_conditions': ['stock', 'stocks', 'investors', 'shares', 'bull', 'bullish'],
    'financial_performance': ['earnings', 'forecast', 'outlook', 'growth', 'performance'],
    'technology': ['technology', 'AI', 'semiconductor', 'chip', 'chips'],
    'regulation': ['lawsuit', 'lawsuits', 'regulatory'],
    'competition': ['market share', 'competition', 'competitive advantage'],
    'geopolitical': ['China', 'Russia', 'Ukraine', 'global', 'trade restrictions', 'tariffs']
}

# Threshold for sentiment classification
SENTIMENT_THRESHOLD = 0.3

# =====================================================================
# Graph Builder Settings
# =====================================================================
# Thresholds for creating edges in the stock relationship graph
PRICE_THRESHOLD = 0.5
VOLUME_THRESHOLD = 0.4
RSI_THRESHOLD = 0.6
MACD_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.7
TOPIC_SENTIMENT_THRESHOLD = 0.6

# =====================================================================
# Model Hyperparameters
# =====================================================================
# Architecture dimensions (balanced for expressivity and stability)
NODE_FEATURES = 9  # Number of input features per node
HIDDEN_DIM = 32  # Further reduced to prevent overfitting
TEMPORAL_DIM = 32  # Further reduced to prevent overfitting
OUTPUT_DIM = 1     # Predicting next-day closing price

# Sequence parameters
FORECAST_HORIZON = 7  # Number of days to forecast
LOOKBACK_WINDOW = 30

# Model architecture
NUM_GNN_LAYERS = 2  # Further reduced for simpler model
NUM_TEMPORAL_LAYERS = 1  # Single temporal layer
EDGE_TYPES = ['price', 'volume', 'rsi', 'macd', 'sentiment']
TEMPORAL_CELL = 'LSTM'

# Regularization
DROPOUT = 0.1  # Reduced dropout for better training signal
USE_BATCH_NORM = False  # Disable batch norm initially
USE_LAYER_NORM = True
USE_RESIDUAL = True
USE_ATTENTION = False  # Disable attention initially
NUM_HEADS = 1

# =====================================================================
# Training Settings
# =====================================================================
# Learning rate settings
INITIAL_LEARNING_RATE = 0.001  # Increased for faster initial learning
MIN_LEARNING_RATE = 1e-5
WARMUP_EPOCHS = 5  # Reduced warmup period

# Batch size for training (balanced for stability and speed)
BATCH_SIZE = 16  # Further reduced for better generalization

# Training duration
EPOCHS = 300
PATIENCE = 30  # Increased patience

# Optimization settings
WEIGHT_DECAY = 1e-4  # Reduced weight decay
GRADIENT_CLIP = 1.0  # Increased gradient clip
LABEL_SMOOTHING = 0.0  # Disabled label smoothing initially
USE_COSINE_SCHEDULE = False  # Disable cosine schedule initially
USE_ONE_CYCLE = False
USE_GRADIENT_ACCUMULATION = False  # Disable gradient accumulation initially
ACCUMULATION_STEPS = 1

# =====================================================================
# Other Configuration Settings
# =====================================================================
# Path for logging
LOG_FILE_PATH = os.path.join("data", "data_loader.log")

# Path for saving data and models
SAVE_PATH = "data"
MODEL_SAVE_PATH = "data/models/stgnn_model.pth"
DATA_SAVE_PATH = "data/processed"

# =====================================================================
# Device Configuration
# =====================================================================
# Determine the best available device for computation
# Priority: MPS (Apple Silicon) > CUDA > CPU
USE_GPU = True
USE_MPS = True  # Enable MPS for Apple Silicon
USE_CUDA = True  # Enable CUDA for NVIDIA GPUs
USE_CPU = True   # Enable CPU fallback

DEVICE = torch.device("mps" if torch.backends.mps.is_available() and USE_MPS else
                     "cuda" if torch.cuda.is_available() and USE_CUDA else
                     "cpu")

print(f"Using device: {DEVICE}") 