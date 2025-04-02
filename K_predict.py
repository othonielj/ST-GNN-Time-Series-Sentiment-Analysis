import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import importlib.util
from tqdm import tqdm
import joblib
from H_model import STGNN
import seaborn as sns

# Import the model from H_model.py using importlib.util
spec = importlib.util.spec_from_file_location("model_module", "H_model.py")
model_module = importlib.util.module_from_spec(spec)
sys.modules["model_module"] = model_module
spec.loader.exec_module(model_module)

# Get the STGNN class and device from the module
STGNN = model_module.STGNN
device = model_module.device

class StockPredictor:
    """
    Class for making predictions with a trained ST-GNN model
    """
    def __init__(self, model_path='models/stgnn_model.pth', lookback_window=30, data_dir="data"):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the trained model
            lookback_window (int): Number of days to look back for input
            data_dir (str): Directory containing data files
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.data_dir = data_dir
        
        # Load the model
        self.model = self.load_model()
        
        # Load the close price scaler
        self.price_scaler = joblib.load(os.path.join(data_dir, "close_scaler.joblib"))
        print("Loaded close price scaler for denormalization")
        
        # Print model information
        print(f"Loaded model from {model_path}")
        
    def load_model(self):
        """
        Load the trained model
        
        Returns:
            model: Loaded model
        """
        try:
            # Load model parameters
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model instance with the same parameters
            model = STGNN(
                node_features=checkpoint['model_params']['node_features'],
                hidden_dim=checkpoint['model_params']['hidden_dim'],
                temporal_dim=checkpoint['model_params']['temporal_dim'],
                output_dim=checkpoint['model_params']['output_dim'],
                num_gnn_layers=checkpoint['model_params']['num_gnn_layers'],
                num_temporal_layers=checkpoint['model_params']['num_temporal_layers'],
                edge_types=checkpoint['model_params']['edge_types'],
                dropout=checkpoint['model_params']['dropout']
            )
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def load_data(self):
        """
        Load graph data and stock prices from saved files
        
        Returns:
            dict: Dictionary with loaded data
        """
        print("Loading data...")
        
        # Load graph structure
        edge_index = torch.load('edge_index.pt').to(self.device)
        node_features = torch.load('node_features.pt').to(self.device)
        edge_types = torch.load('edge_types.pt')
        edge_weights = torch.load('edge_weights.pt').to(self.device)
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
    
    def prepare_input_sequence(self, stock_data, node_labels, node_features):
        """
        Prepare input sequence for prediction
        
        Args:
            stock_data (pd.DataFrame): Stock data with prices and sentiment
            node_labels (dict): Dictionary mapping node indices to stock tickers
            node_features (torch.Tensor): Node features tensor
            
        Returns:
            torch.Tensor: Input sequence for prediction
        """
        # Get the most recent data for the lookback window
        recent_data = stock_data.iloc[-self.lookback_window:].reset_index(drop=True)
        
        # Get close price columns for each stock
        close_columns = {}
        for idx, ticker in node_labels.items():
            close_columns[idx] = [col for col in stock_data.columns if col.startswith(f"{ticker}_Close")]
        
        # Create input sequence for time series data
        time_series_seq = []
        for idx in range(len(node_labels)):
            if close_columns[idx]:
                # Get data for this stock
                stock_seq = recent_data[close_columns[idx][0]].values
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
        
        # Add batch dimension and convert to tensor
        input_seq = torch.tensor(combined_seq, dtype=torch.float32).unsqueeze(0)
        
        return input_seq
    
    def predict(self, input_seq, edge_index, edge_types, edge_weights):
        """
        Make predictions with the model
        
        Args:
            input_seq (torch.Tensor): Input sequence
            edge_index (torch.Tensor): Graph connectivity
            edge_types (list): Edge types
            edge_weights (torch.Tensor): Edge weights
            
        Returns:
            np.ndarray: Predicted stock prices
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            input_seq = input_seq.to(self.device)
            
            # Forward pass
            predictions = self.model(input_seq, edge_index, edge_types, edge_weights)
            
            # Move predictions to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def format_predictions(self, predictions, dates, node_indices):
        """
        Format predictions into a DataFrame with denormalized prices
        
        Args:
            predictions (np.ndarray): Predicted stock prices with shape (batch_size, forecast_horizon, num_nodes)
            dates (pd.DatetimeIndex): Dates for the predictions
            node_indices (list): Indices of the nodes in the predictions
            
        Returns:
            pd.DataFrame: Formatted predictions
        """
        print(f"Predictions shape: {predictions.shape}")
        print(f"Dates shape: {len(dates)}")
        print(f"Node indices: {node_indices}")
        
        # Convert predictions to numpy if it's a torch tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Get the stock names from node_labels
        node_labels = torch.load('node_labels.pt')
        print(f"Node labels: {node_labels}")
        
        # Create DataFrame with predictions
        df_predictions = pd.DataFrame()
        
        # Generate dates for the forecast horizon
        current_date = pd.to_datetime(dates[0])
        forecast_dates = [current_date + pd.Timedelta(days=i) for i in range(predictions.shape[1])]
        df_predictions['Date'] = forecast_dates
        
        # Denormalize predictions for each stock
        for i in range(len(node_labels)):  # Process all stocks
            stock_name = node_labels[i]
            print(f"Processing stock {stock_name} (index {i})")
            
            # Extract predictions for this stock across all time steps
            stock_predictions = predictions[0, :, i]  # Shape: (forecast_horizon,)
            
            # Create a zero array for each time step
            normalized_predictions = np.zeros((len(stock_predictions), len(node_labels)))
            normalized_predictions[:, i] = stock_predictions  # Fill in the predictions for this stock
            
            # Denormalize predictions
            denormalized_predictions = self.price_scaler.inverse_transform(normalized_predictions)[:, i]
            df_predictions[stock_name] = denormalized_predictions
        
        return df_predictions
    
    def save_predictions(self, df_predictions, output_path='predictions/predicted_stock_prices.csv'):
        """
        Save predictions to a CSV file
        
        Args:
            df_predictions (pd.DataFrame): Formatted predictions
            output_path (str): Path to save the predictions
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save predictions
        df_predictions.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
    
    def plot_predictions(self, predictions_df, output_dir='predictions'):
        """
        Plot predictions for each stock
        
        Args:
            predictions_df (pd.DataFrame): Formatted predictions
            output_dir (str): Directory to save the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better visualization
        plt.style.use('default')
        
        # Create a figure with a 5x2 grid of subplots
        n_stocks = len(predictions_df.columns) - 1  # Subtract 1 for the Date column
        fig, axes = plt.subplots(5, 2, figsize=(15, 25))  # Adjusted figure size for vertical layout
        fig.suptitle('Stock Price Predictions (Next 7 Days)', fontsize=16, y=0.95)
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten()
        
        # Set the color palette
        colors = plt.cm.Set3(np.linspace(0, 1, n_stocks))
        
        for i, stock in enumerate(predictions_df.columns[1:]):  # Skip Date column
            ax = axes_flat[i]
            
            # Plot predictions
            ax.plot(predictions_df['Date'], predictions_df[stock], 
                   color=colors[i], linewidth=2)
            
            # Customize the plot
            ax.set_title(f'{stock} Price Predictions', fontsize=12, pad=10)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Price ($)', fontsize=10)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis to show dollar values
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add current price annotation
            current_price = predictions_df[stock].iloc[0]
            ax.annotate(f'Current: ${current_price:,.2f}',
                       xy=(predictions_df['Date'].iloc[0], current_price),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Add predicted price annotation
            predicted_price = predictions_df[stock].iloc[-1]
            ax.annotate(f'Predicted: ${predicted_price:,.2f}',
                       xy=(predictions_df['Date'].iloc[-1], predicted_price),
                       xytext=(10, -10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Remove the last subplot if we have an odd number of stocks
        if n_stocks < 10:
            axes_flat[-1].remove()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot with high DPI
        plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction plots saved to {output_dir}")


def main(args):
    """
    Main function to predict stock prices
    
    Args:
        args: Command line arguments
    """
    # Create predictor
    predictor = StockPredictor(
        model_path=args.model_path,
        lookback_window=args.lookback_window
    )
    
    # Load data
    data = predictor.load_data()
    
    # Prepare input sequence
    input_seq = predictor.prepare_input_sequence(
        data['stock_data'],
        data['node_labels'],
        data['node_features']
    )
    
    # Make predictions
    predictions = predictor.predict(
        input_seq,
        data['edge_index'],
        data['edge_types'],
        data['edge_weights']
    )
    
    # Use current date for predictions
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Format predictions with all node indices
    df_predictions = predictor.format_predictions(
        predictions,
        pd.date_range(start=current_date, periods=len(predictions)),
        range(len(data['node_labels']))  # Use all node indices
    )
    
    # Save predictions
    predictor.save_predictions(df_predictions, args.output_path)
    
    # Plot predictions
    if args.plot:
        predictor.plot_predictions(df_predictions)
    
    print("Prediction completed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict stock prices using trained ST-GNN model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='models/stgnn_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--lookback_window', type=int, default=30,
                        help='Number of days to look back for input')
    parser.add_argument('--forecast_horizon', type=int, default=7,
                        help='Number of days to forecast')
    
    # Output parameters
    parser.add_argument('--output_path', type=str, default='predictions/predicted_stock_prices.csv',
                        help='Path to save the predictions')
    parser.add_argument('--plot', action='store_true',
                        help='Plot predictions for each stock')
    
    args = parser.parse_args()
    
    # Run main function
    main(args) 