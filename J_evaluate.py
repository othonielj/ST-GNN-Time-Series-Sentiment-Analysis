import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import argparse
import importlib.util
import sys
import json
import shap
from tqdm import tqdm

# Import the model from H_model.py using importlib.util
spec = importlib.util.spec_from_file_location("model_module", "H_model.py")
model_module = importlib.util.module_from_spec(spec)
sys.modules["model_module"] = model_module
spec.loader.exec_module(model_module)

# Get the STGNN class and device from the module
STGNN = model_module.STGNN
device = model_module.device

class ModelEvaluator:
    """
    Class for evaluating the ST-GNN model on test set
    """
    def __init__(
        self, 
        model_path='data/models/stgnn_model.pth',  # Updated path
        lookback_window=30,
        forecast_horizon=7,
        output_dir='evaluation',
        test_ratio=0.15  # Match the test ratio from training
    ):
        """
        Initialize the evaluator
        
        Args:
            model_path (str): Path to the trained model
            lookback_window (int): Number of days to look back for input
            forecast_horizon (int): Number of days to forecast
            output_dir (str): Directory to save evaluation results
            test_ratio (float): Ratio of data to use for testing (should match training split)
        """
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.output_dir = output_dir
        self.test_ratio = test_ratio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        self.model, self.model_params = self._load_model()
        
        # Print model information
        print(f"Loaded model from {model_path}")
        print(f"Model parameters: {self.model_params}")
        
        # Initialize metrics dictionary
        self.metrics = {
            'test_set': {
                'overall': {},
                'per_stock': {},
                'per_horizon': {},
                'with_sentiment': {},
                'without_sentiment': {}
            },
            'validation': {
                'best_loss': None,
                'best_epoch': None
            }
        }
        
    def _load_model(self):
        """
        Load the trained model
        
        Returns:
            tuple: (model, model_params)
        """
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=device)
        
        # Create model using configuration from M_config.py
        model = STGNN().to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        # Get model parameters from checkpoint for reference
        model_params = checkpoint.get('model_params', {})
        
        return model, model_params
    
    def load_data(self):
        """
        Load graph data and stock prices from saved files
        
        Returns:
            dict: Dictionary with loaded data
        """
        print("Loading data...")
        
        # Load graph structure
        edge_index = torch.load('edge_index.pt').to(device)
        node_features = torch.load('node_features.pt').to(device)
        edge_types = torch.load('edge_types.pt')
        edge_weights = torch.load('edge_weights.pt').to(device)
        node_labels = torch.load('node_labels.pt')
        
        # Load stock data
        stock_data = pd.read_csv('data/stock_data_preprocessed.csv')
        
        print(f"Loaded data with {len(node_labels)} stocks and {len(stock_data)} days")
        
        return {
            'edge_index': edge_index,
            'node_features': node_features,
            'edge_types': edge_types,
            'edge_weights': edge_weights,
            'node_labels': node_labels,
            'stock_data': stock_data
        }
    
    def prepare_validation_data(self, stock_data, node_labels, node_features):
        """
        Prepare validation data for evaluation
        
        Args:
            stock_data (pd.DataFrame): Stock data with prices and sentiment
            node_labels (dict): Dictionary mapping node indices to stock tickers
            node_features (torch.Tensor): Node features tensor
            
        Returns:
            tuple: (X_val, y_val, close_columns)
        """
        # Get close price columns for each stock
        close_columns = {}
        for idx, ticker in node_labels.items():
            close_columns[idx] = [col for col in stock_data.columns if col.startswith(f"{ticker}_Close")]
        
        # Calculate the number of sequences
        n_sequences = len(stock_data) - self.lookback_window - self.forecast_horizon + 1
        
        # Calculate validation set indices (15% after training set)
        train_ratio = 0.7
        val_ratio = 0.15
        train_size = int(n_sequences * train_ratio)
        val_size = int(n_sequences * val_ratio)
        
        # Get the start and end indices for validation data
        val_start_idx = train_size
        val_end_idx = train_size + val_size
        
        # Create sequences for validation data
        X_val = []
        y_val = []
        
        for i in range(val_start_idx, val_end_idx):
            # Input sequence for time series data
            time_series_seq = []
            for idx in range(len(node_labels)):
                if close_columns[idx]:
                    # Get data for this stock
                    stock_seq = stock_data[close_columns[idx][0]].values[i:i+self.lookback_window]
                    time_series_seq.append(stock_seq)
                else:
                    # Fallback if no data
                    time_series_seq.append(np.zeros(self.lookback_window))
            
            # Convert to numpy array and transpose to (lookback_window, num_nodes)
            time_series_seq = np.array(time_series_seq).T
            
            # Get node features for each stock
            node_features_np = node_features.cpu().numpy()
            
            # Create a combined sequence with both time series and node features
            combined_seq = np.zeros((self.lookback_window, len(node_labels), 1 + node_features_np.shape[1]))
            
            # Fill in the time series data (close prices)
            for t in range(self.lookback_window):
                for n in range(len(node_labels)):
                    # First feature is the close price
                    combined_seq[t, n, 0] = time_series_seq[t, n]
                    
                    # Remaining features are the node features
                    combined_seq[t, n, 1:] = node_features_np[n]
            
            # Target sequence
            target = []
            for idx in range(len(node_labels)):
                if close_columns[idx]:
                    # Get future prices for this stock
                    stock_target = stock_data[close_columns[idx][0]].values[
                        i+self.lookback_window:i+self.lookback_window+self.forecast_horizon
                    ]
                    target.append(stock_target)
                else:
                    # Fallback if no data
                    target.append(np.zeros(self.forecast_horizon))
            
            X_val.append(combined_seq)
            y_val.append(np.array(target).T)  # Transpose to (forecast_horizon, num_nodes)
        
        # Convert to numpy arrays
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        print(f"Prepared validation data: {len(X_val)} sequences")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        return X_val, y_val, close_columns

    def evaluate_validation_set(self, X_val, y_val, edge_index, edge_types, edge_weights):
        """
        Evaluate the model on validation set
        
        Args:
            X_val (np.ndarray): Validation input sequences
            y_val (np.ndarray): Validation target sequences
            edge_index (torch.Tensor): Graph connectivity
            edge_types (list): Edge types
            edge_weights (torch.Tensor): Edge weights
            
        Returns:
            dict: Validation metrics
        """
        print("\nEvaluating model on validation set...")
        
        # Convert validation data to tensors
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_val_pred = self.model(X_val_tensor, edge_index, edge_types, edge_weights)
        
        # Convert predictions to numpy
        y_val_pred = y_val_pred.cpu().numpy()
        y_val_true = y_val_tensor.cpu().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_val_true.flatten(), y_val_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_true.flatten(), y_val_pred.flatten())
        
        print(f"\nValidation Set Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        
        # Store validation metrics
        self.metrics['validation'] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        return self.metrics['validation']
    
    def prepare_test_data(self, stock_data, node_labels, node_features, test_ratio=0.15):
        """
        Prepare test data for evaluation
        
        Args:
            stock_data (pd.DataFrame): Stock data with prices and sentiment
            node_labels (dict): Dictionary mapping node indices to stock tickers
            node_features (torch.Tensor): Node features tensor
            test_ratio (float): Ratio of data to use for testing (default: 0.15 to match training split)
            
        Returns:
            tuple: (X_test, y_test, close_columns)
        """
        # Get close price columns for each stock
        close_columns = {}
        for idx, ticker in node_labels.items():
            close_columns[idx] = [col for col in stock_data.columns if col.startswith(f"{ticker}_Close")]
        
        # Calculate the number of sequences
        n_sequences = len(stock_data) - self.lookback_window - self.forecast_horizon + 1
        
        # Calculate the number of test sequences
        n_test = int(n_sequences * test_ratio)
        
        # Get the start index for test data
        test_start_idx = n_sequences - n_test
        
        # Create sequences for test data
        X_test = []
        y_test = []
        
        for i in range(test_start_idx, n_sequences):
            # Input sequence for time series data
            time_series_seq = []
            for idx in range(len(node_labels)):
                if close_columns[idx]:
                    # Get data for this stock
                    stock_seq = stock_data[close_columns[idx][0]].values[i:i+self.lookback_window]
                    time_series_seq.append(stock_seq)
                else:
                    # Fallback if no data
                    time_series_seq.append(np.zeros(self.lookback_window))
            
            # Convert to numpy array and transpose to (lookback_window, num_nodes)
            time_series_seq = np.array(time_series_seq).T
            
            # Get node features for each stock
            node_features_np = node_features.cpu().numpy()
            
            # Create a combined sequence with both time series and node features
            # Shape: (lookback_window, num_nodes, features_per_node)
            combined_seq = np.zeros((self.lookback_window, len(node_labels), 1 + node_features_np.shape[1]))
            
            # Fill in the time series data (close prices)
            for t in range(self.lookback_window):
                for n in range(len(node_labels)):
                    # First feature is the close price
                    combined_seq[t, n, 0] = time_series_seq[t, n]
                    
                    # Remaining features are the node features
                    combined_seq[t, n, 1:] = node_features_np[n]
            
            # Target sequence
            target = []
            for idx in range(len(node_labels)):
                if close_columns[idx]:
                    # Get future prices for this stock
                    stock_target = stock_data[close_columns[idx][0]].values[
                        i+self.lookback_window:i+self.lookback_window+self.forecast_horizon
                    ]
                    target.append(stock_target)
                else:
                    # Fallback if no data
                    target.append(np.zeros(self.forecast_horizon))
            
            X_test.append(combined_seq)
            y_test.append(np.array(target).T)  # Transpose to (forecast_horizon, num_nodes)
        
        # Convert to numpy arrays
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Prepared test data: {len(X_test)} sequences")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_test, y_test, close_columns
    
    def evaluate_model(self, X_test, y_test, edge_index, edge_types, edge_weights, node_labels, close_columns):
        """
        Evaluate the model on test data
        """
        print("Evaluating model on test set...")
        
        # Store test data as instance variables for plotting
        self.X_test = X_test
        self.y_test = y_test
        
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor, edge_index, edge_types, edge_weights)
        
        # Convert predictions back to numpy and store as instance variable
        self.y_pred = y_pred_tensor.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        # Calculate overall metrics
        mse = mean_squared_error(y_true.flatten(), self.y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), self.y_pred.flatten())
        
        print(f"\nOverall Test Set Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        
        # Store metrics
        self.metrics['test_set']['overall'] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        # Calculate per-stock metrics
        per_stock_metrics = {}
        for i, ticker in node_labels.items():
            stock_mse = mean_squared_error(y_true[:, :, i].flatten(), self.y_pred[:, :, i].flatten())
            stock_rmse = np.sqrt(stock_mse)
            stock_r2 = r2_score(y_true[:, :, i].flatten(), self.y_pred[:, :, i].flatten())
            
            per_stock_metrics[ticker] = {
                'mse': float(stock_mse),
                'rmse': float(stock_rmse),
                'r2': float(stock_r2)
            }
        
        self.metrics['test_set']['per_stock'] = per_stock_metrics
        
        # Calculate per-horizon metrics
        per_horizon_metrics = {}
        for h in range(self.forecast_horizon):
            horizon_mse = mean_squared_error(y_true[:, h, :].flatten(), self.y_pred[:, h, :].flatten())
            horizon_rmse = np.sqrt(horizon_mse)
            horizon_r2 = r2_score(y_true[:, h, :].flatten(), self.y_pred[:, h, :].flatten())
            
            per_horizon_metrics[f'day_{h+1}'] = {
                'mse': float(horizon_mse),
                'rmse': float(horizon_rmse),
                'r2': float(horizon_r2)
            }
        
        self.metrics['test_set']['per_horizon'] = per_horizon_metrics
        
        # Compare with/without sentiment
        sentiment_comparison = self.compare_with_without_sentiment(X_test, y_test, edge_index, edge_types, edge_weights)
        self.metrics['test_set']['sentiment_comparison'] = sentiment_comparison
        
        return self.metrics['test_set']
    
    def compare_with_without_sentiment(self, X_test, y_test, edge_index, edge_types, edge_weights):
        """
        Compare model performance with and without sentiment analysis
        
        Args:
            X_test (np.ndarray): Test input sequences
            y_test (np.ndarray): Test target sequences
            edge_index (torch.Tensor): Graph connectivity
            edge_types (list): Edge types
            edge_weights (torch.Tensor): Edge weights
            
        Returns:
            dict: Comparison metrics
        """
        print("Comparing performance with and without sentiment analysis...")
        
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Make predictions with full model (with sentiment)
        self.model.eval()
        with torch.no_grad():
            y_pred_with_sentiment = self.model(X_test_tensor, edge_index, edge_types, edge_weights)
        
        # Calculate metrics with sentiment
        mse_with = mean_squared_error(y_test.flatten(), y_pred_with_sentiment.cpu().numpy().flatten())
        rmse_with = np.sqrt(mse_with)
        r2_with = r2_score(y_test.flatten(), y_pred_with_sentiment.cpu().numpy().flatten())
        
        # Store metrics with sentiment
        self.metrics['test_set']['with_sentiment'] = {
            'MSE': mse_with,
            'RMSE': rmse_with,
            'R2': r2_with
        }
        
        # Create a version of X_test without sentiment features
        # Identify sentiment feature indices (assuming they are the last features)
        # In our case, the last 2 features are general_sentiment and topic_sentiment
        feature_dim = X_test.shape[3]
        sentiment_indices = list(range(feature_dim - 2, feature_dim))
        
        # Create a mask for non-sentiment features
        non_sentiment_mask = np.ones(feature_dim, dtype=bool)
        non_sentiment_mask[sentiment_indices] = False
        
        # Create X_test without sentiment
        X_test_no_sentiment = X_test.copy()
        # Zero out sentiment features
        for idx in sentiment_indices:
            X_test_no_sentiment[:, :, :, idx] = 0
        
        # Convert to tensor
        X_test_no_sentiment_tensor = torch.tensor(X_test_no_sentiment, dtype=torch.float32).to(device)
        
        # Make predictions without sentiment
        with torch.no_grad():
            y_pred_without_sentiment = self.model(X_test_no_sentiment_tensor, edge_index, edge_types, edge_weights)
        
        # Calculate metrics without sentiment
        mse_without = mean_squared_error(y_test.flatten(), y_pred_without_sentiment.cpu().numpy().flatten())
        rmse_without = np.sqrt(mse_without)
        r2_without = r2_score(y_test.flatten(), y_pred_without_sentiment.cpu().numpy().flatten())
        
        # Store metrics without sentiment
        self.metrics['test_set']['without_sentiment'] = {
            'MSE': mse_without,
            'RMSE': rmse_without,
            'R2': r2_without
        }
        
        # Calculate improvement percentage
        mse_improvement = ((mse_without - mse_with) / mse_without) * 100
        rmse_improvement = ((rmse_without - rmse_with) / rmse_without) * 100
        r2_improvement = ((r2_with - r2_without) / abs(r2_without)) * 100 if r2_without != 0 else float('inf')
        
        # Store improvement metrics
        self.metrics['test_set']['sentiment_improvement'] = {
            'MSE_improvement_percent': mse_improvement,
            'RMSE_improvement_percent': rmse_improvement,
            'R2_improvement_percent': r2_improvement
        }
        
        print(f"With sentiment - MSE: {mse_with:.6f}, RMSE: {rmse_with:.6f}, R²: {r2_with:.6f}")
        print(f"Without sentiment - MSE: {mse_without:.6f}, RMSE: {rmse_without:.6f}, R²: {r2_without:.6f}")
        print(f"Improvement - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%")
        
        return self.metrics['test_set']['with_sentiment']
    
    def analyze_feature_importance(self, X_test, edge_index, edge_types, edge_weights, node_labels):
        """
        Analyze feature importance using a simplified approach
        """
        print("Analyzing feature importance...")
        
        # Feature names
        feature_names = [
            "Close Price",
            "Volume",
            "RSI",
            "MACD",
            "PE Ratio",
            "EPS",
            "General Sentiment",
            "Topic Sentiment"
        ]
        
        # Calculate feature importance based on correlation with prediction error
        feature_importance = {}
        
        # Calculate errors for each prediction
        errors = np.abs(self.y_test - self.y_pred)  # Shape: (batch, horizon, nodes)
        mean_errors = np.mean(errors, axis=(0, 1))  # Average across batch and horizon
        
        # For each feature, calculate its correlation with prediction accuracy
        for i, feature_name in enumerate(feature_names):
            # Get feature values for each node (averaging across time and batch)
            feature_values = np.mean(X_test[:, :, :, i], axis=(0, 1))  # Average across batch and time
            
            # Calculate correlation between feature and prediction error
            correlation = np.corrcoef(feature_values, mean_errors)[0, 1]
            importance = np.abs(correlation)  # Use absolute correlation as importance
            
            feature_importance[feature_name] = float(importance)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Store in metrics
        self.metrics['test_set']['feature_importance'] = {
            'overall': feature_importance,
            'sorted': sorted_features
        }
        
        # Create feature importance plot
        plt.figure(figsize=(12, 6))
        features = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create bar plot
        bars = plt.barh(features, importance_values)
        plt.title('Feature Importance Analysis')
        plt.xlabel('Absolute Correlation with Prediction Error')
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'model_evaluation', 'feature_analysis', 'feature_importance.png'))
        plt.close()
        
        print("\nFeature Importance Rankings:")
        for feature, importance in sorted_features:
            print(f"- {feature}: {importance:.4f}")
        
        return feature_importance
    
    def save_evaluation_results(self):
        """
        Save evaluation results to CSV and JSON files
        """
        print("Saving evaluation results...")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save overall metrics
        overall_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'R2'],
            'Value': [
                self.metrics['test_set']['overall']['mse'],
                self.metrics['test_set']['overall']['rmse'],
                self.metrics['test_set']['overall']['r2']
            ]
        })
        overall_df.to_csv(os.path.join(self.output_dir, 'overall_metrics.csv'), index=False)
        
        # Save per-stock metrics
        per_stock_data = []
        for ticker, metrics in self.metrics['test_set']['per_stock'].items():
            per_stock_data.append({
                'Stock': ticker,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'R2': metrics['r2']
            })
        
        per_stock_df = pd.DataFrame(per_stock_data)
        per_stock_df.to_csv(os.path.join(self.output_dir, 'per_stock_metrics.csv'), index=False)
        
        # Save sentiment comparison
        sentiment_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'R2'],
            'With_Sentiment': [
                self.metrics['test_set']['with_sentiment']['MSE'],
                self.metrics['test_set']['with_sentiment']['RMSE'],
                self.metrics['test_set']['with_sentiment']['R2']
            ],
            'Without_Sentiment': [
                self.metrics['test_set']['without_sentiment']['MSE'],
                self.metrics['test_set']['without_sentiment']['RMSE'],
                self.metrics['test_set']['without_sentiment']['R2']
            ],
            'Improvement_Percent': [
                self.metrics['test_set']['sentiment_improvement']['MSE_improvement_percent'],
                self.metrics['test_set']['sentiment_improvement']['RMSE_improvement_percent'],
                self.metrics['test_set']['sentiment_improvement']['R2_improvement_percent']
            ]
        })
        sentiment_df.to_csv(os.path.join(self.output_dir, 'sentiment_comparison.csv'), index=False)
        
        # Save feature importance
        if 'feature_importance' in self.metrics['test_set']:
            feature_importance_data = []
            for feature, importance in self.metrics['test_set']['feature_importance']['sorted']:
                feature_importance_data.append({
                    'Feature': feature,
                    'Importance': importance
                })
            
            feature_importance_df = pd.DataFrame(feature_importance_data)
            feature_importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        # Save all metrics as JSON
        with open(os.path.join(self.output_dir, 'all_metrics.json'), 'w') as f:
            json.dump(self.metrics['test_set'], f, indent=4)
        
        print(f"Evaluation results saved to {self.output_dir}")
    
    def plot_evaluation_results(self):
        """
        Plot comprehensive evaluation results
        """
        print("Plotting comprehensive evaluation results...")
        
        # Create main output directory and subdirectories
        base_dir = os.path.join('visualizations', 'model_evaluation')
        dirs = {
            'basic_metrics': os.path.join(base_dir, 'basic_metrics'),
            'feature_analysis': os.path.join(base_dir, 'feature_analysis'),
            'prediction_analysis': os.path.join(base_dir, 'prediction_analysis'),
            'performance_analysis': os.path.join(base_dir, 'performance_analysis')
        }
        
        # Create all directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. Basic Metrics (Per-stock and Sentiment)
        stocks = list(self.metrics['test_set']['per_stock'].keys())
        mse_values = [metrics['mse'] for metrics in self.metrics['test_set']['per_stock'].values()]
        rmse_values = [metrics['rmse'] for metrics in self.metrics['test_set']['per_stock'].values()]
        r2_values = [metrics['r2'] for metrics in self.metrics['test_set']['per_stock'].values()]
        
        # Per-stock metrics
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot MSE
        sns.barplot(x=stocks, y=mse_values, ax=axes[0])
        axes[0].set_title('Mean Squared Error (MSE) by Stock')
        axes[0].set_ylabel('MSE')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        # Plot RMSE
        sns.barplot(x=stocks, y=rmse_values, ax=axes[1])
        axes[1].set_title('Root Mean Squared Error (RMSE) by Stock')
        axes[1].set_ylabel('RMSE')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
        # Plot R2
        sns.barplot(x=stocks, y=r2_values, ax=axes[2])
        axes[2].set_title('R² Score by Stock')
        axes[2].set_ylabel('R²')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['basic_metrics'], 'per_stock_metrics.png'))
        plt.close()
        
        # Sentiment comparison
        plt.figure(figsize=(10, 6))
        metrics = ['MSE', 'RMSE', 'R2']
        with_sentiment = [
            self.metrics['test_set']['with_sentiment']['MSE'],
            self.metrics['test_set']['with_sentiment']['RMSE'],
            self.metrics['test_set']['with_sentiment']['R2']
        ]
        without_sentiment = [
            self.metrics['test_set']['without_sentiment']['MSE'],
            self.metrics['test_set']['without_sentiment']['RMSE'],
            self.metrics['test_set']['without_sentiment']['R2']
        ]
        
        sentiment_df = pd.DataFrame({
            'Metric': metrics * 2,
            'Value': with_sentiment + without_sentiment,
            'Type': ['With Sentiment'] * 3 + ['Without Sentiment'] * 3
        })
        
        sns.barplot(x='Metric', y='Value', hue='Type', data=sentiment_df)
        plt.title('Model Performance With vs. Without Sentiment Analysis')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['basic_metrics'], 'sentiment_comparison.png'))
        plt.close()
        
        # 3. Prediction Analysis
        # Prediction vs Actual plots
        plt.figure(figsize=(15, 10))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (stock, metrics) in enumerate(self.metrics['test_set']['per_stock'].items()):
            if idx < 4:  # Plot first 4 stocks
                ax = axes[idx//2, idx%2]
                stock_actual = self.y_test[:, :, idx].flatten()
                stock_pred = self.y_pred[:, :, idx].flatten()
                
                ax.scatter(stock_actual, stock_pred, alpha=0.5)
                ax.plot([stock_actual.min(), stock_actual.max()], 
                       [stock_actual.min(), stock_actual.max()], 
                       'r--', label='Perfect Prediction')
                
                ax.set_title(f'{stock} - Actual vs Predicted')
                ax.set_xlabel('Actual Price')
                ax.set_ylabel('Predicted Price')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['prediction_analysis'], 'prediction_vs_actual.png'))
        plt.close()
        
        # Error distribution
        plt.figure(figsize=(12, 6))
        errors = self.y_test.flatten() - self.y_pred.flatten()
        sns.histplot(errors, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['prediction_analysis'], 'error_distribution.png'))
        plt.close()
        
        # 4. Performance Analysis
        # Horizon-wise performance
        plt.figure(figsize=(12, 6))
        horizons = list(self.metrics['test_set']['per_horizon'].keys())
        rmse_values = [metrics['rmse'] for metrics in self.metrics['test_set']['per_horizon'].values()]
        
        plt.plot(horizons, rmse_values, marker='o')
        plt.title('RMSE Across Different Forecast Horizons')
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['performance_analysis'], 'horizon_performance.png'))
        plt.close()
        
        # Directional accuracy
        plt.figure(figsize=(12, 6))
        directional_accuracies = []
        for idx, ticker in enumerate(stocks):
            stock_actual = self.y_test[:, :, idx].flatten()
            stock_pred = self.y_pred[:, :, idx].flatten()
            
            actual_direction = np.sign(np.diff(stock_actual))
            predicted_direction = np.sign(np.diff(stock_pred))
            accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            directional_accuracies.append(accuracy)
        
        sns.barplot(x=stocks, y=directional_accuracies)
        plt.title('Directional Accuracy by Stock')
        plt.xlabel('Stock')
        plt.ylabel('Directional Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['performance_analysis'], 'directional_accuracy.png'))
        plt.close()
        
        print(f"All evaluation plots saved to {base_dir}")
        print("Organized in the following directories:")
        for dir_name, dir_path in dirs.items():
            print(f"- {dir_name}: {dir_path}")
        print("\nNote: EDA visualizations are stored in visualizations/eda/")


def main(args):
    """
    Main function to evaluate the ST-GNN model
    
    Args:
        args: Command line arguments
    """
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        output_dir=args.output_dir
    )
    
    # Load data
    data = evaluator.load_data()
    
    # Prepare validation data
    X_val, y_val, val_close_columns = evaluator.prepare_validation_data(
        data['stock_data'],
        data['node_labels'],
        data['node_features']
    )
    
    # Evaluate on validation set
    evaluator.evaluate_validation_set(
        X_val,
        y_val,
        data['edge_index'],
        data['edge_types'],
        data['edge_weights']
    )
    
    # Prepare test data
    X_test, y_test, close_columns = evaluator.prepare_test_data(
        data['stock_data'],
        data['node_labels'],
        data['node_features'],
        test_ratio=args.test_ratio
    )
    
    # Evaluate model on test set
    evaluator.evaluate_model(
        X_test,
        y_test,
        data['edge_index'],
        data['edge_types'],
        data['edge_weights'],
        data['node_labels'],
        close_columns
    )
    
    # Analyze feature importance
    evaluator.analyze_feature_importance(
        X_test,
        data['edge_index'],
        data['edge_types'],
        data['edge_weights'],
        data['node_labels']
    )
    
    # Save evaluation results
    evaluator.save_evaluation_results()
    
    # Plot evaluation results
    evaluator.plot_evaluation_results()
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate ST-GNN model for stock price prediction')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='data/models/stgnn_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--lookback_window', type=int, default=30,
                        help='Number of days to look back for input')
    parser.add_argument('--forecast_horizon', type=int, default=7,
                        help='Number of days to forecast')
    
    # Evaluation parameters
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Run main function
    main(args) 