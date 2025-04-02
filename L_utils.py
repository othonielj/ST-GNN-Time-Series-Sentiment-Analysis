import pandas as pd
import numpy as np
import os
import gc
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging
import psutil
import traceback

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_csv(filepath, optimize_dtypes=True, date_column='Date'):
    """
    Load CSV data with optimized dtype management for memory efficiency.
    
    Args:
        filepath (str): Path to the CSV file
        optimize_dtypes (bool): Whether to optimize dtypes for memory efficiency
        date_column (str): Name of the date column to parse as datetime
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            log_message(f"Error: File not found - {filepath}", level="ERROR")
            return None
        
        # Log start of loading
        log_message(f"Loading CSV file: {filepath}")
        
        if optimize_dtypes:
            # First read a small sample to infer dtypes
            sample_df = pd.read_csv(filepath, nrows=1000)
            
            # Create dtype dictionary for optimization
            dtypes = {}
            for col in sample_df.columns:
                if col == date_column:
                    continue  # Skip date column
                
                # Check if column contains only integers
                if pd.api.types.is_integer_dtype(sample_df[col]):
                    # Use smallest integer type that can represent the data
                    min_val = sample_df[col].min()
                    max_val = sample_df[col].max()
                    
                    if min_val >= 0:
                        if max_val < 256:
                            dtypes[col] = 'uint8'
                        elif max_val < 65536:
                            dtypes[col] = 'uint16'
                        else:
                            dtypes[col] = 'uint32'
                    else:
                        if min_val > -128 and max_val < 128:
                            dtypes[col] = 'int8'
                        elif min_val > -32768 and max_val < 32768:
                            dtypes[col] = 'int16'
                        else:
                            dtypes[col] = 'int32'
                
                # Check if column contains floating point numbers
                elif pd.api.types.is_float_dtype(sample_df[col]):
                    # Use float32 instead of float64 to save memory
                    dtypes[col] = 'float32'
            
            # Read the full CSV with optimized dtypes
            parse_dates = [date_column] if date_column in sample_df.columns else None
            df = pd.read_csv(filepath, dtype=dtypes, parse_dates=parse_dates)
        else:
            # Read without dtype optimization
            parse_dates = [date_column] if date_column in pd.read_csv(filepath, nrows=1).columns else None
            df = pd.read_csv(filepath, parse_dates=parse_dates)
        
        # Log success
        log_message(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from {filepath}")
        
        return df
    
    except Exception as e:
        log_message(f"Error loading CSV file {filepath}: {str(e)}", level="ERROR")
        log_message(traceback.format_exc(), level="DEBUG")
        return None

def normalize_data(df, columns=None, scaler=None, return_scaler=False):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to normalize. If None, all numeric columns are normalized.
        scaler (sklearn.preprocessing.MinMaxScaler): Pre-fitted scaler. If None, a new scaler is created.
        return_scaler (bool): Whether to return the scaler along with the normalized data
        
    Returns:
        pd.DataFrame or tuple: Normalized dataframe, or tuple of (normalized dataframe, scaler)
    """
    try:
        # Log start of normalization
        log_message(f"Normalizing data for {len(columns) if columns else 'all numeric'} columns")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_normalized = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out columns that don't exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            log_message("Warning: No valid columns to normalize", level="WARNING")
            return (df_normalized, None) if return_scaler else df_normalized
        
        # Create or use provided scaler
        if scaler is None:
            scaler = MinMaxScaler()
            # Fit the scaler on the data
            scaler.fit(df[columns])
        
        # Transform the data
        df_normalized[columns] = scaler.transform(df[columns])
        
        # Log success
        log_message(f"Successfully normalized {len(columns)} columns")
        
        if return_scaler:
            return df_normalized, scaler
        else:
            return df_normalized
    
    except Exception as e:
        log_message(f"Error normalizing data: {str(e)}", level="ERROR")
        log_message(traceback.format_exc(), level="DEBUG")
        if return_scaler:
            return df, None
        else:
            return df

def log_message(message, level="INFO"):
    """
    Log a message to the pipeline log file and print to console.
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR, DEBUG)
    """
    # Get the current time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Print to console
    print(f"[{timestamp}] {level}: {message}")
    
    # Log to file
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "DEBUG":
        logging.debug(message)

def memory_cleanup(pytorch_cleanup=True):
    """
    Clean up memory after model training for efficient usage on Apple M2.
    
    Args:
        pytorch_cleanup (bool): Whether to clean up PyTorch GPU memory
    
    Returns:
        dict: Memory usage statistics before and after cleanup
    """
    # Get memory usage before cleanup
    before = {
        'ram_used_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3)
    }
    
    log_message(f"Memory cleanup started. RAM usage: {before['ram_used_percent']}% ({before['ram_used_gb']:.2f} GB)")
    
    # Run garbage collection
    gc.collect()
    
    # PyTorch specific cleanup
    if pytorch_cleanup and torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_message("PyTorch CUDA memory cache cleared")
    
    # For Apple M2 with Metal Performance Shaders
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # There's no direct equivalent to cuda.empty_cache() for MPS
        # but we can force some cleanup by creating and deleting a small tensor
        temp = torch.ones(1, device='mps')
        del temp
        log_message("PyTorch MPS memory cleanup attempted")
    
    # Get memory usage after cleanup
    after = {
        'ram_used_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3)
    }
    
    # Calculate difference
    diff = {
        'ram_percent_diff': before['ram_used_percent'] - after['ram_used_percent'],
        'ram_gb_diff': before['ram_used_gb'] - after['ram_used_gb']
    }
    
    log_message(f"Memory cleanup completed. RAM usage: {after['ram_used_percent']}% ({after['ram_used_gb']:.2f} GB)")
    log_message(f"Memory freed: {diff['ram_percent_diff']:.2f}% ({diff['ram_gb_diff']:.2f} GB)")
    
    return {
        'before': before,
        'after': after,
        'difference': diff
    }

def get_column_patterns(df, verbose=True):
    """
    Identify column patterns in the dataframe (e.g., Close, Volume, RSI columns).
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        verbose (bool): Whether to print the results
        
    Returns:
        dict: Dictionary with column patterns
    """
    patterns = {
        'close_columns': [col for col in df.columns if "Close" in col],
        'volume_columns': [col for col in df.columns if "Volume" in col],
        'rsi_columns': [col for col in df.columns if "RSI" in col],
        'macd_columns': [col for col in df.columns if "MACD" in col],
        'pe_columns': [col for col in df.columns if "PE_Ratio" in col],
        'eps_columns': [col for col in df.columns if "EPS" in col],
        'sentiment_columns': [col for col in df.columns if "Sentiment" in col],
        'topic_sentiment_columns': [col for col in df.columns if "Topic_Sentiment" in col]
    }
    
    if verbose:
        log_message("Column patterns found:")
        for pattern_name, columns in patterns.items():
            log_message(f"  {pattern_name}: {len(columns)} columns")
    
    return patterns

def extract_stock_tickers(columns, pattern="Close"):
    """
    Extract unique stock tickers from column names.
    
    Args:
        columns (list): List of column names
        pattern (str): Pattern to look for in column names (e.g., "Close")
        
    Returns:
        list: List of unique stock tickers
    """
    tickers = []
    for col in columns:
        if pattern in col:
            # Extract ticker from column name (assuming format like "AAPL_Close")
            ticker = col.split('_')[0]
            if ticker not in tickers:
                tickers.append(ticker)
    
    return sorted(tickers)

def create_directory_if_not_exists(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
        
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            log_message(f"Created directory: {directory}")
        return True
    except Exception as e:
        log_message(f"Error creating directory {directory}: {str(e)}", level="ERROR")
        return False 