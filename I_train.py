import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
import argparse
import importlib.util
import sys
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import json
from pathlib import Path

# Import configuration settings
from M_config import (
    INITIAL_LEARNING_RATE,
    MIN_LEARNING_RATE,
    WARMUP_EPOCHS,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    LABEL_SMOOTHING,
    HIDDEN_DIM,
    TEMPORAL_DIM,
    FORECAST_HORIZON,
    LOOKBACK_WINDOW,
    DROPOUT,
    TEMPORAL_CELL,
    SAVE_PATH,
    LOG_FILE_PATH,
    USE_COSINE_SCHEDULE,
    USE_GRADIENT_ACCUMULATION,
    ACCUMULATION_STEPS,
    MODEL_SAVE_PATH,
    DEVICE,
    TICKERS,
    NODE_FEATURES,
    NUM_GNN_LAYERS,
    NUM_TEMPORAL_LAYERS,
    EDGE_TYPES,
    OUTPUT_DIM
)

# Import the model from H_model.py using importlib.util
spec = importlib.util.spec_from_file_location("model_module", "H_model.py")
model_module = importlib.util.module_from_spec(spec)
sys.modules["model_module"] = model_module
spec.loader.exec_module(model_module)

# Get the STGNN class and device from the module
STGNN = model_module.STGNN
device = model_module.device

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Log training configuration
logger.info("Training Configuration:")
logger.info(f"Device: {device}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Learning Rate: {INITIAL_LEARNING_RATE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Patience: {PATIENCE}")
logger.info(f"Lookback Window: {LOOKBACK_WINDOW}")
logger.info(f"Forecast Horizon: {FORECAST_HORIZON}")
logger.info(f"Model Architecture: {TEMPORAL_CELL} with {NUM_GNN_LAYERS} GNN layers and {NUM_TEMPORAL_LAYERS} temporal layers")
logger.info(f"Hidden Dimensions: {HIDDEN_DIM}")
logger.info(f"Temporal Dimensions: {TEMPORAL_DIM}")
logger.info(f"Node Features: {NODE_FEATURES}")
logger.info(f"Edge Types: {EDGE_TYPES}")
logger.info("=" * 50)

class StockDataset(torch.utils.data.Dataset):
    """
    Dataset class for stock data with graph structure
    """
    def __init__(self, split='train', lookback_window=LOOKBACK_WINDOW, forecast_horizon=FORECAST_HORIZON):
        """
        Initialize the dataset
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            lookback_window (int): Number of days to look back for input
            forecast_horizon (int): Number of days to forecast
        """
        super().__init__()
        self.split = split
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        # Load data
        data = load_data()
        self.stock_data = data['stock_data']
        self.edge_index = data['edge_index']
        self.edge_types = data['edge_types']
        self.edge_weights = data['edge_weights']
        self.node_labels = data['node_labels']
        self.node_features = data['node_features']
        
        # Prepare sequences
        sequences, targets = self._prepare_sequences()
        
        # Split data
        train_ratio, val_ratio = 0.7, 0.15
        n = len(sequences)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        
        if split == 'train':
            self.sequences = sequences[:train_size]
            self.targets = targets[:train_size]
        elif split == 'val':
            self.sequences = sequences[train_size:train_size+val_size]
            self.targets = targets[train_size:train_size+val_size]
        else:  # test
            self.sequences = sequences[train_size+val_size:]
            self.targets = targets[train_size+val_size:]
        
        # Convert to tensors
        self.sequences = torch.FloatTensor(self.sequences)
        self.targets = torch.FloatTensor(self.targets)
        
        # Move to device
        self.sequences = self.sequences.to(DEVICE)
        self.targets = self.targets.to(DEVICE)
        self.edge_index = self.edge_index.to(DEVICE)
        self.edge_weights = self.edge_weights.to(DEVICE)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
        
    def _prepare_sequences(self):
        """
        Prepare sequences for training and testing
        
        This method creates input sequences that combine:
        1. Time series data (closing prices)
        2. Node features from the graph structure
        
        Returns:
            tuple: (sequences, targets) where sequences include both time series and node features
        """
        sequences = []
        targets = []
        
        # Get close price columns for each stock
        close_columns = {}
        for idx, ticker in self.node_labels.items():
            close_columns[idx] = [col for col in self.stock_data.columns if col.startswith(f"{ticker}_Close")]
        
        # Create sequences
        for i in range(len(self.stock_data) - self.lookback_window - self.forecast_horizon + 1):
            # Input sequence for time series data
            time_series_seq = []
            for idx in range(len(self.node_labels)):
                if close_columns[idx]:
                    # Get data for this stock
                    stock_seq = self.stock_data[close_columns[idx][0]].values[i:i+self.lookback_window]
                    time_series_seq.append(stock_seq)
                else:
                    # Fallback if no data
                    time_series_seq.append(np.zeros(self.lookback_window))
            
            # Convert to numpy array and transpose to (lookback_window, num_nodes)
            time_series_seq = np.array(time_series_seq).T
            
            # Get node features for each stock
            # We'll repeat the node features for each time step in the sequence
            node_features_np = self.node_features.cpu().numpy()
            
            # Create a combined sequence with both time series and node features
            # Shape: (lookback_window, num_nodes, features_per_node)
            # Where features_per_node = 1 (close price) + node_features.shape[1]
            combined_seq = np.zeros((self.lookback_window, len(self.node_labels), 1 + node_features_np.shape[1]))
            
            # Fill in the time series data (close prices)
            for t in range(self.lookback_window):
                for n in range(len(self.node_labels)):
                    # First feature is the close price
                    combined_seq[t, n, 0] = time_series_seq[t, n]
                    
                    # Remaining features are the node features
                    combined_seq[t, n, 1:] = node_features_np[n]
            
            # Target sequence
            target = []
            for idx in range(len(self.node_labels)):
                if close_columns[idx]:
                    # Get future prices for this stock
                    stock_target = self.stock_data[close_columns[idx][0]].values[
                        i+self.lookback_window:i+self.lookback_window+self.forecast_horizon
                    ]
                    target.append(stock_target)
                else:
                    # Fallback if no data
                    target.append(np.zeros(self.forecast_horizon))
            
            sequences.append(combined_seq)
            targets.append(np.array(target).T)  # Transpose to (forecast_horizon, num_nodes)
        
        return np.array(sequences), np.array(targets)


class Trainer:
    """
    Trainer class for the ST-GNN model with advanced training features
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        device=DEVICE,
        learning_rate=INITIAL_LEARNING_RATE,
        min_lr=MIN_LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE,
        gradient_clip=GRADIENT_CLIP,
        label_smoothing=LABEL_SMOOTHING
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        
        # Initialize criterion
        self.criterion = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        if USE_COSINE_SCHEDULE:
            self.scheduler = CosineAnnealingLR(
            self.optimizer, 
                T_max=epochs,
                eta_min=min_lr
            )
        else:
            self.scheduler = None
        
        # Initialize early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize gradient accumulation
        self.accumulation_steps = ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            # Get graph data from dataset
            edge_index = self.train_loader.dataset.edge_index
            edge_types = self.train_loader.dataset.edge_types
            edge_weights = self.train_loader.dataset.edge_weights
            
            # Forward pass
            outputs = self.model(X_batch, edge_index, edge_types, edge_weights)
            loss = self.criterion(outputs, y_batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                # Get graph data from dataset
                edge_index = self.val_loader.dataset.edge_index
                edge_types = self.val_loader.dataset.edge_types
                edge_weights = self.val_loader.dataset.edge_weights
                
                # Forward pass
                outputs = self.model(X_batch, edge_index, edge_types, edge_weights)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """
        Train the model with logging
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        # Create directory for training history if it doesn't exist
        history_dir = Path('data/training_history')
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique filename for this training run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = history_dir / f'training_history_{timestamp}.json'
        
        # Initialize training history
        training_history = {
            'config': {
                'device': str(device),
                'batch_size': BATCH_SIZE,
                'learning_rate': INITIAL_LEARNING_RATE,
                'epochs': EPOCHS,
                'patience': PATIENCE,
                'model_type': 'STGNN'
            },
            'metrics': {
                'epochs': [],
                'train_losses': [],
                'val_losses': [],
                'learning_rates': []
            }
        }
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (sequences, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                # Get graph data from dataset
                edge_index = self.train_loader.dataset.edge_index
                edge_types = self.train_loader.dataset.edge_types
                edge_weights = self.train_loader.dataset.edge_weights
                
                # Forward pass
                outputs = self.model(sequences, edge_index, edge_types, edge_weights)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log batch progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            if self.val_loader is not None:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                # Learning rate logging
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Current Learning Rate: {current_lr:.6f}")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'model_params': {
                            'node_features': NODE_FEATURES,
                            'hidden_dim': HIDDEN_DIM,
                            'temporal_dim': TEMPORAL_DIM,
                            'output_dim': OUTPUT_DIM,
                            'forecast_horizon': FORECAST_HORIZON,
                            'num_gnn_layers': NUM_GNN_LAYERS,
                            'num_temporal_layers': NUM_TEMPORAL_LAYERS,
                            'edge_types': EDGE_TYPES,
                            'dropout': DROPOUT,
                            'temporal_cell': TEMPORAL_CELL
                        }
                    }
                    torch.save(checkpoint, MODEL_SAVE_PATH)
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save metrics to history
            training_history['metrics']['epochs'].append(epoch + 1)
            training_history['metrics']['train_losses'].append(avg_train_loss)
            training_history['metrics']['val_losses'].append(val_loss)
            training_history['metrics']['learning_rates'].append(current_lr)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 50)

        # Save training history at the end of training
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=4)
        logger.info(f'Training history saved to {history_file}')

        return self.best_val_loss


class LabelSmoothingMSELoss(nn.Module):
    """MSE Loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Apply label smoothing
        smoothed_target = target * (1 - self.smoothing) + self.smoothing * target.mean()
        return self.mse(pred, smoothed_target)


def get_linear_warmup_scheduler(optimizer, num_warmup_steps):
    """Create a linear warmup scheduler"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data():
    """
    Load graph data and stock prices from saved files
    
    Returns:
        dict: Dictionary with loaded data
    """
    logger.info("Loading data...")
    
    # Load graph structure
    edge_index = torch.load('edge_index.pt').to(device)
    node_features = torch.load('node_features.pt').to(device)
    edge_types = torch.load('edge_types.pt')
    edge_weights = torch.load('edge_weights.pt').to(device)
    node_labels = torch.load('node_labels.pt')
    
    # Load stock data
    stock_data = pd.read_csv(f'{SAVE_PATH}stock_data_preprocessed.csv')
    
    logger.info(f"Loaded data with {len(node_labels)} stocks and {len(stock_data)} days")
    
    return {
        'edge_index': edge_index,
        'node_features': node_features,
        'edge_types': edge_types,
        'edge_weights': edge_weights,
        'node_labels': node_labels,
        'stock_data': stock_data
    }


def main():
    # Load data
    train_dataset = StockDataset(split='train')
    val_dataset = StockDataset(split='val')
    test_dataset = StockDataset(split='test')
    
    # Print data info
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    # Get sample batch to check dimensions
    sample_batch = next(iter(train_dataset))
    if isinstance(sample_batch, tuple):
        X_sample, _ = sample_batch
    else:
        X_sample = sample_batch
    print(f"Sequence shape: {tuple(X_sample.shape)}, Target shape: {tuple(train_dataset[0][1].shape)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = STGNN().to(DEVICE)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=INITIAL_LEARNING_RATE,
        min_lr=MIN_LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE,
        gradient_clip=GRADIENT_CLIP,
        label_smoothing=LABEL_SMOOTHING
    )
    
    print(f"Starting training on {DEVICE}...")
    
    # Train model
    best_val_loss = trainer.train()
    
    if isinstance(best_val_loss, dict):
        print(f"Training completed. Best validation metrics: {best_val_loss}")
    else:
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss


if __name__ == "__main__":
    main() 