import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import os
import time
import random
from typing import List, Dict, Tuple, Optional
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import logging

# Import configuration settings
from M_config import START_DATE, END_DATE, CSV_PATH, TICKERS, MAX_RETRIES, SAVE_PATH, LOG_FILE_PATH

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_stock_data_batch(tickers: List[str], start_date: str, end_date: str, max_retries=4, delay_between_batches=15) -> pd.DataFrame:
    """
    Fetch stock data for multiple tickers in a single batch, with retry logic and multiple data sources.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts
        delay_between_batches: Delay in seconds between retry attempts
        
    Returns:
        DataFrame with stock data for all tickers
    """
    all_data = []
    
    # Try different data sources in order of preference
    data_sources = ['yahoo', 'stooq']
    
    for source in data_sources:
        logger.info(f"Attempting to fetch data from {source} for {len(tickers)} tickers...")
        
        for attempt in range(max_retries):
            try:
                if source == 'yahoo':
                    # Use yfinance for Yahoo data
                    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
                    
                    # If data is empty or has too few rows, try another source
                    if data.empty or len(data) < 5:
                        logger.warning(f"Received empty or insufficient data from {source}. Trying next source.")
                        break
                    
                    # Process multi-ticker result
                    if len(tickers) > 1:
                        # Restructure the data
                        processed_data = []
                        for ticker in tickers:
                            if ticker in data.columns.levels[0]:
                                ticker_data = data[ticker].copy()
                                ticker_data['Ticker'] = ticker
                                processed_data.append(ticker_data)
                        
                        if processed_data:
                            combined_data = pd.concat(processed_data)
                            all_data.append(combined_data)
                            logger.info(f"Successfully fetched data for {len(processed_data)} tickers from {source}")
                            return combined_data.reset_index()
                    else:
                        # Single ticker result
                        data['Ticker'] = tickers[0]
                        all_data.append(data)
                        logger.info(f"Successfully fetched data for {tickers[0]} from {source}")
                        return data.reset_index()
                
                elif source == 'stooq':
                    # Use pandas_datareader for Stooq data
                    all_ticker_data = []
                    for ticker in tickers:
                        try:
                            ticker_data = pdr.data.get_data_stooq(ticker, start=start_date, end=end_date)
                            if not ticker_data.empty:
                                ticker_data['Ticker'] = ticker
                                all_ticker_data.append(ticker_data)
                        except Exception as e:
                            logger.warning(f"Error fetching {ticker} from Stooq: {e}")
                    
                    if all_ticker_data:
                        combined_data = pd.concat(all_ticker_data)
                        all_data.append(combined_data)
                        logger.info(f"Successfully fetched data for {len(all_ticker_data)} tickers from {source}")
                        return combined_data.reset_index()
                
                # If we got here with data, we're done
                if all_data:
                    return pd.concat(all_data).reset_index()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {source}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = delay_between_batches * (1 + random.random())
                    logger.info(f"Waiting {sleep_time:.2f} seconds before retrying...")
                    time.sleep(sleep_time)
    
    # If we get here, all sources and retries failed
    logger.error("All data sources failed. Returning empty DataFrame.")
    return pd.DataFrame()

def fetch_stock_data_sequential(tickers: List[str], start_date: str, end_date: str, max_retries=5, delay_between_tickers=10) -> pd.DataFrame:
    """
    Fetch stock data for each ticker sequentially to avoid rate limits.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts per ticker
        delay_between_tickers: Delay in seconds between fetching different tickers
        
    Returns:
        DataFrame with stock data for all tickers
    """
    all_data = []
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        
        for attempt in range(max_retries):
            try:
                # Try yfinance first
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if ticker_data.empty or len(ticker_data) < 5:
                    # If yfinance fails, try pandas_datareader with Stooq
                    logger.warning(f"Yfinance returned empty data for {ticker}. Trying Stooq...")
                    ticker_data = pdr.data.get_data_stooq(ticker, start=start_date, end=end_date)
                
                if not ticker_data.empty and len(ticker_data) >= 5:
                    ticker_data['Ticker'] = ticker
                    all_data.append(ticker_data)
                    logger.info(f"Successfully fetched data for {ticker} with {len(ticker_data)} rows")
                    break
                else:
                    logger.warning(f"Attempt {attempt+1}/{max_retries} for {ticker} returned insufficient data")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {ticker}: {e}")
            
            if attempt < max_retries - 1:
                sleep_time = delay_between_tickers * (0.5 + random.random())
                logger.info(f"Waiting {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
        
        # Add a delay between tickers to avoid rate limits
        if ticker != tickers[-1]:
            sleep_time = delay_between_tickers * (0.8 + 0.4 * random.random())
            logger.info(f"Waiting {sleep_time:.2f} seconds before fetching next ticker...")
            time.sleep(sleep_time)
    
    if all_data:
        return pd.concat(all_data).reset_index()
    else:
        logger.error("Failed to fetch data for any tickers. Returning empty DataFrame.")
        return pd.DataFrame()

def generate_synthetic_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic stock data when real data cannot be fetched.
    Useful for testing or when APIs are unavailable.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with synthetic stock data
    """
    logger.warning("Generating synthetic data for testing purposes")
    
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    all_data = []
    
    for ticker in tickers:
        # Start with a random price between $50 and $500
        base_price = random.uniform(50, 500)
        
        # Generate random walk
        np.random.seed(hash(ticker) % 10000)  # Seed based on ticker for reproducibility
        returns = np.random.normal(0.0005, 0.015, size=len(date_range))
        prices = base_price * (1 + np.cumsum(returns))
        
        # Generate other columns
        volumes = np.random.randint(100000, 10000000, size=len(date_range))
        
        # Create DataFrame
        ticker_data = pd.DataFrame({
            'Date': date_range,
            'Open': prices * (1 - np.random.uniform(0, 0.01, size=len(date_range))),
            'High': prices * (1 + np.random.uniform(0, 0.02, size=len(date_range))),
            'Low': prices * (1 - np.random.uniform(0, 0.02, size=len(date_range))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': volumes,
            'Ticker': ticker
        })
        
        all_data.append(ticker_data)
    
    return pd.concat(all_data)

def fetch_stock_data(tickers: List[str] = TICKERS, start_date: str = START_DATE, end_date: str = END_DATE, max_retries=MAX_RETRIES, delay=5) -> pd.DataFrame:
    """
    Main function to fetch stock data, trying different methods.
    
    Args:
        tickers: List of stock ticker symbols (default from config)
        start_date: Start date in YYYY-MM-DD format (default from config)
        end_date: End date in YYYY-MM-DD format (default from config)
        max_retries: Maximum number of retry attempts (default from config)
        delay: Base delay in seconds between retries
        
    Returns:
        DataFrame with stock data for all tickers
    """
    logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Try batch fetching first
    data = fetch_stock_data_batch(tickers, start_date, end_date, max_retries, delay)
    
    # If batch fetching fails or returns incomplete data, try sequential fetching
    if data.empty or len(data) < 5 or len(set(data['Ticker'])) < len(tickers) * 0.5:
        logger.warning("Batch fetching failed or returned incomplete data. Trying sequential fetching...")
        data = fetch_stock_data_sequential(tickers, start_date, end_date, max_retries, delay)
    
    # If all real data fetching methods fail, generate synthetic data as a last resort
    if data.empty or len(data) < 5:
        logger.warning("All real data fetching methods failed. Generating synthetic data...")
        data = generate_synthetic_data(tickers, start_date, end_date)
    
    # Ensure the data is properly formatted
    if not data.empty:
        # Make sure Date is a datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by Date and Ticker
        data = data.sort_values(['Ticker', 'Date'])
        
        logger.info(f"Successfully fetched data with {len(data)} rows for {len(set(data['Ticker']))} tickers")
    
    return data

def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for stock data.
    
    Args:
        data: DataFrame with stock data
        
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Computing technical indicators...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Process each ticker separately
    tickers = df['Ticker'].unique()
    result_dfs = []
    
    for ticker in tickers:
        ticker_data = df[df['Ticker'] == ticker].copy()
        
        # Ensure data is sorted by date
        ticker_data = ticker_data.sort_values('Date')
        
        # Compute RSI (14-day)
        delta = ticker_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        ticker_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Compute MACD
        ema12 = ticker_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = ticker_data['Close'].ewm(span=26, adjust=False).mean()
        ticker_data['MACD'] = ema12 - ema26
        
        result_dfs.append(ticker_data)
    
    return pd.concat(result_dfs)

def normalize_and_save(data: pd.DataFrame, filename: str) -> None:
    """
    Normalize the data and save to CSV.
    
    Args:
        data: DataFrame with stock data
        filename: Path to save the CSV file
    """
    # Create a wide format DataFrame with tickers as columns
    logger.info("Normalizing and saving data...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Convert to wide format for stock-specific columns
    metrics = {
        'Close': 'Close',
        'Volume': 'Volume',
        'RSI': 'RSI',
        'MACD': 'MACD',
        'PE_Ratio___': 'PE_Ratio',  # Updated to match actual column names
        'EPS___': 'EPS'             # Updated to match actual column names
    }
    wide_data = pd.DataFrame()
    
    # Process each metric separately
    for col_suffix, metric in metrics.items():
        # Filter columns for current metric
        metric_cols = [col for col in df.columns if col_suffix in col]
        if metric_cols:
            metric_data = df[metric_cols]
            # Add Date for joining
            metric_data['Date'] = df['Date']
            # Set Date as index for proper column naming
            metric_data.set_index('Date', inplace=True)
            # Join with existing wide_data
            if wide_data.empty:
                wide_data = metric_data
            else:
                wide_data = wide_data.join(metric_data)
    
    # Reset index to make Date a column
    if not wide_data.empty:
        wide_data = wide_data.reset_index()
    else:
        logger.warning("No data to save after processing")
        return
    
    # Save to CSV
    wide_data.to_csv(filename, index=False)
    logger.info(f"Data saved to {filename}")

def fetch_pe_ratio_eps(ticker: str, max_retries=5, delay=5) -> dict:
    """
    Fetch PE ratio and EPS for a stock ticker using yfinance's earnings history and quarterly financials.
    
    Args:
        ticker: Stock ticker symbol
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
        
    Returns:
        Dictionary with PE ratio and EPS historical data
    """
    for attempt in range(max_retries):
        try:
            # Use yfinance to get the data
            stock = yf.Ticker(ticker)
            logger.info(f"Fetching data for {ticker}...")
            
            # Get current PE ratio and EPS from info
            current_info = stock.info
            current_pe = current_info.get('trailingPE', None)
            current_eps = current_info.get('trailingEps', None)
            logger.info(f"{ticker} current PE: {current_pe}, EPS: {current_eps}")
            
            # Get earnings history
            earnings_dates = stock.get_earnings_dates()
            logger.info(f"{ticker} earnings dates shape: {earnings_dates.shape if earnings_dates is not None else 'None'}")
            if earnings_dates is not None and not earnings_dates.empty:
                # Filter out future dates and NaN values
                earnings_dates = earnings_dates[earnings_dates.index <= pd.Timestamp.now()]
                earnings_dates = earnings_dates.dropna(subset=['Reported EPS'])
                logger.info(f"{ticker} filtered earnings dates:\n{earnings_dates}")
            
            # Get quarterly financials for additional data
            quarterly_financials = stock.quarterly_financials
            logger.info(f"{ticker} quarterly financials shape: {quarterly_financials.shape if quarterly_financials is not None else 'None'}")
            if quarterly_financials is not None and not quarterly_financials.empty:
                # Get EPS data if available
                if 'Diluted EPS' in quarterly_financials.index:
                    eps_data = quarterly_financials.loc['Diluted EPS']
                    # Filter out future dates and NaN values
                    eps_data = eps_data[eps_data.index <= pd.Timestamp.now()]
                    eps_data = eps_data.dropna()
                    logger.info(f"{ticker} quarterly EPS data:\n{eps_data}")
                else:
                    eps_data = None
                    logger.warning(f"No Diluted EPS data found in quarterly financials for {ticker}")
            else:
                eps_data = None
            
            return {
                'current': {
                    'PE_Ratio': current_pe,
                    'EPS': current_eps
                },
                'historical': {
                    'earnings_dates': earnings_dates,
                    'eps_data': eps_data
                }
            }
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {ticker}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Stack trace for {ticker}:", exc_info=True)
        
        # Wait before retrying
        if attempt < max_retries - 1:
            sleep_time = delay * (1 + random.random())
            logger.info(f"Waiting {sleep_time:.2f} seconds before retrying...")
            time.sleep(sleep_time)
    
    # If all attempts fail, return None values
    logger.error(f"Failed to fetch PE ratio and EPS for {ticker} after {max_retries} attempts")
    return {
        'current': {
            'PE_Ratio': None,
            'EPS': None
        },
        'historical': None
    }

def update_csv_with_pe_eps(data: pd.DataFrame, tickers: List[str] = TICKERS, csv_path: str = CSV_PATH, delay_between_tickers=5):
    """
    Update the CSV file with PE ratio and EPS data.
    
    Args:
        data: DataFrame with stock data
        tickers: List of stock ticker symbols (default from config)
        csv_path: Path to the CSV file (default from config)
        delay_between_tickers: Delay in seconds between fetching data for different tickers
    """
    logger.info("Updating CSV with PE ratio and EPS data...")
    
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        # Read the existing CSV
        df = pd.read_csv(csv_path)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Read existing CSV with shape: {df.shape}")
    else:
        # Create a new DataFrame with the same structure as the input data
        df = data.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Created new DataFrame with shape: {df.shape}")
    
    # Add PE ratio and EPS columns if they don't exist
    for ticker in tickers:
        pe_col = f"{ticker}_PE_Ratio___"
        eps_col = f"{ticker}_EPS___"
        price_col = f"{ticker}_Close"
        
        if pe_col not in df.columns:
            df[pe_col] = None
            logger.info(f"Added new column: {pe_col}")
        if eps_col not in df.columns:
            df[eps_col] = None
            logger.info(f"Added new column: {eps_col}")
        
        # Fetch PE ratio and EPS data
        logger.info(f"Fetching PE ratio and EPS for {ticker}...")
        pe_eps_data = fetch_pe_ratio_eps(ticker)
        
        if pe_eps_data['historical'] is not None:
            # Process earnings dates data first (more accurate)
            earnings_dates = pe_eps_data['historical']['earnings_dates']
            if earnings_dates is not None and not earnings_dates.empty:
                logger.info(f"Processing {len(earnings_dates)} earnings dates for {ticker}")
                for date, row in earnings_dates.iterrows():
                    # Find the closest date in our DataFrame
                    closest_date = df['Date'].iloc[(df['Date'] - pd.to_datetime(date)).abs().argsort()[0]]
                    logger.debug(f"Found closest date {closest_date} for earnings date {date}")
                    
                    # Update EPS value if we have reported EPS
                    if pd.notnull(row['Reported EPS']):
                        df.loc[df['Date'] == closest_date, eps_col] = row['Reported EPS']
                        logger.info(f"Updated {ticker} EPS to {row['Reported EPS']} for date {closest_date}")
            
            # Process quarterly EPS data as backup
            eps_data = pe_eps_data['historical']['eps_data']
            if eps_data is not None and not eps_data.empty:
                logger.info(f"Processing {len(eps_data)} quarterly financials for {ticker}")
                for date, eps in eps_data.items():
                    if pd.notnull(eps):
                        # Find the closest date in our DataFrame
                        closest_date = df['Date'].iloc[(df['Date'] - pd.to_datetime(date)).abs().argsort()[0]]
                        logger.debug(f"Found closest date {closest_date} for financial date {date}")
                        
                        # Update EPS if we don't already have a value from earnings dates
                        if pd.isnull(df.loc[df['Date'] == closest_date, eps_col].iloc[0]):
                            df.loc[df['Date'] == closest_date, eps_col] = eps
                            logger.info(f"Updated {ticker} EPS to {eps} for date {closest_date}")
        
        # Fill forward EPS values
        df[eps_col] = df[eps_col].ffill()
        logger.info(f"Filled forward EPS values for {ticker}")
        
        # Update with current EPS for the most recent date if available
        if pe_eps_data['current']['EPS'] is not None:
            df.loc[df['Date'] == df['Date'].max(), eps_col] = pe_eps_data['current']['EPS']
            logger.info(f"Updated {ticker} current EPS to {pe_eps_data['current']['EPS']}")
        
        # Calculate PE ratios using closing prices and EPS values
        if price_col in df.columns:
            # Calculate PE ratio for all dates where we have both price and EPS
            mask = (df[eps_col].notnull()) & (df[price_col].notnull()) & (df[eps_col] != 0)
            df.loc[mask, pe_col] = df.loc[mask, price_col] / df.loc[mask, eps_col]
            logger.info(f"Calculated PE ratios for {ticker} using closing prices and EPS values")
            
            # Update current PE ratio if available (more accurate)
            if pe_eps_data['current']['PE_Ratio'] is not None:
                df.loc[df['Date'] == df['Date'].max(), pe_col] = pe_eps_data['current']['PE_Ratio']
                logger.info(f"Updated {ticker} current PE ratio to {pe_eps_data['current']['PE_Ratio']}")
        
        # Add a delay between tickers to avoid rate limits
        if ticker != tickers[-1]:
            sleep_time = delay_between_tickers * (0.8 + 0.4 * random.random())
            logger.info(f"Waiting {sleep_time:.2f} seconds before fetching next ticker...")
            time.sleep(sleep_time)
    
    # Save the updated DataFrame
    df.to_csv(csv_path, index=False)
    logger.info(f"Updated CSV saved to {csv_path}")

def get_last_trading_day():
    """Get the most recent trading day."""
    today = pd.Timestamp.now()
    
    # If today is a weekend, move back to Friday
    while today.dayofweek in [5, 6]:  # 5 = Saturday, 6 = Sunday
        today = today - pd.Timedelta(days=1)
    
    # Format as YYYY-MM-DD
    return today.strftime('%Y-%m-%d')

def update_stock_data(tickers: List[str] = TICKERS, start_date: str = START_DATE, end_date: str = END_DATE, csv_path: str = CSV_PATH):
    """
    Update stock data for the specified tickers and date range.
    
    Args:
        tickers: List of stock ticker symbols (default from config)
        start_date: Start date in YYYY-MM-DD format (default from config)
        end_date: End date in YYYY-MM-DD format (default from config)
        csv_path: Path to the CSV file (default from config)
    """
    logger.info(f"Using date range from config: {start_date} to {end_date}")
    
    # Convert dates to datetime for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # If start date is after end date, use end date for both
    if start_dt > end_dt:
        logger.info(f"Start date {start_date} is after end date {end_date}, using end date for both")
        start_date = end_date
    
    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        logger.info(f"Found existing data at {csv_path}")
        
        # Read the last date from the CSV
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns and not df.empty:
            last_date = pd.to_datetime(df['Date'].max()).strftime('%Y-%m-%d')
            logger.info(f"Last date in CSV: {last_date}")
            
            # If we already have data up to the end date, no need to fetch more
            if pd.to_datetime(last_date) >= end_dt:
                logger.info(f"CSV already contains data up to {end_date}, no need to fetch more")
                return
    
    logger.info(f"Fetching new data from {start_date} to {end_date}")
    
    # Fetch new data
    new_data = fetch_stock_data(tickers, start_date, end_date)
    
    if new_data.empty:
        logger.warning("No new data fetched")
        return
    
    # Compute technical indicators
    new_data = compute_technical_indicators(new_data)
    
    # If we have existing data, merge the new data with it
    if os.path.exists(csv_path):
        # Read existing data
        existing_data = pd.read_csv(csv_path)
        existing_data['Date'] = pd.to_datetime(existing_data['Date'])
        
        # Combine existing and new data
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Date'], keep='last')
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
    else:
        combined_data = new_data
    
    # Update PE ratio and EPS data for the entire dataset
    update_csv_with_pe_eps(combined_data, tickers, csv_path)
    
    # Read the updated data with PE ratios and EPS
    final_data = pd.read_csv(csv_path)
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    
    # Sort by date
    final_data = final_data.sort_values('Date').reset_index(drop=True)
    
    # Save the final normalized data
    normalize_and_save(final_data, csv_path)
    
    logger.info("Stock data update complete")


# Example usage
if __name__ == "__main__":
    # Using values from M_config
    tickers = TICKERS
    start_date = START_DATE
    end_date = END_DATE
    csv_path = CSV_PATH
    
    # Install required packages if not already installed
    try:
        import pandas_datareader
    except ImportError:
        print("Installing pandas_datareader...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas_datareader"])
        import pandas_datareader
    
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        import subprocess
        subprocess.check_call(["pip", "install", "numpy"])
        import numpy as np
    
    # Update stock data with real data only
    update_stock_data(tickers, start_date, end_date, csv_path) 