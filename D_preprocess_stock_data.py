import pandas as pd
import os
import numpy as np
import sys
import joblib

# Import utility functions
import L_utils as utils

# Define file paths
# Use relative paths for better portability
data_dir = "data"
stock_data_file = os.path.join(data_dir, "stock_data.csv")
sentiment_general_file = os.path.join(data_dir, "general_sentiment.csv")
sentiment_topic_file = os.path.join(data_dir, "topic_sentiment.csv")
output_file = os.path.join(data_dir, "stock_data_preprocessed.csv")

# Load datasets
utils.log_message("Loading stock data...")
df_stock = utils.load_csv(stock_data_file)
utils.log_message(f"Stock data loaded with {len(df_stock)} rows and {len(df_stock.columns)} columns")

# Check if sentiment files exist
has_sentiment_data = os.path.exists(sentiment_general_file) and os.path.exists(sentiment_topic_file)

if has_sentiment_data:
    utils.log_message("Loading sentiment data...")
    df_general_sentiment = utils.load_csv(sentiment_general_file)
    df_topic_sentiment = utils.load_csv(sentiment_topic_file)
    utils.log_message(f"Sentiment data loaded: {len(df_general_sentiment)} general records, {len(df_topic_sentiment)} topic records")

    ### **1️⃣ Aggregate Topic Sentiment Data to Ensure One Entry per (Date, Stock)**
    df_topic_sentiment = df_topic_sentiment.groupby(["Date", "Stock"]).agg({"Topic Sentiment Score": "mean"}).reset_index()
    df_topic_sentiment.rename(columns={"Topic Sentiment Score": "Avg_Topic_Sentiment"}, inplace=True)

    ### **2️⃣ Merge Sentiment Data Before Merging with Stock Data**
    df_sentiment = df_general_sentiment.merge(df_topic_sentiment, on=["Date", "Stock"], how="outer")

### **3️⃣ Identify Important Features (Includes Correlation Test)**
# Get column patterns using utility function
utils.log_message("Identifying columns...")
patterns = utils.get_column_patterns(df_stock)

# Extract columns from patterns
close_columns = patterns['close_columns']
volume_columns = patterns['volume_columns']
financial_columns = patterns['pe_columns'] + patterns['eps_columns']
technical_columns = patterns['macd_columns'] + patterns['rsi_columns']

# Define specified_features before using it
specified_features = financial_columns + technical_columns

# Initialize selected_columns with basic columns
selected_columns = ["Date"] + close_columns + volume_columns + specified_features

# Compute correlation, ensuring only numeric columns are considered
numeric_df = df_stock.select_dtypes(include=['number'])  # Exclude 'Date'
if len(close_columns) > 0 and all(col in numeric_df.columns for col in close_columns):
    correlation = numeric_df.corr()[close_columns].mean(axis=1).sort_values(ascending=False)
    
    # Print correlation results
    utils.log_message("Correlation results:")
    for col, corr_val in correlation.head(10).items():
        utils.log_message(f"  {col}: {corr_val:.6f}")
    
    # Apply a correlation threshold (e.g., |0.2|) to keep only important features
    important_features = correlation[correlation.abs() > 0.2].index.tolist()
    
    # Add important features to selected columns if not already included
    selected_columns += [col for col in important_features if col not in selected_columns]
else:
    utils.log_message("Warning: Could not compute correlations. Using all columns.", level="WARNING")
    selected_columns = df_stock.columns.tolist()

# Filter dataset to retain only selected features
# Make sure all selected columns exist in the dataframe
selected_columns = [col for col in selected_columns if col in df_stock.columns]
df_stock = df_stock[selected_columns]

utils.log_message(f"Selected {len(selected_columns)} columns for processing")

if has_sentiment_data:
    ### **4️⃣ Merge Sentiment Data with Stock Data in Wide Format**
    # Pivot sentiment data to wide format
    sentiment_wide = df_sentiment.pivot(index="Date", columns="Stock", values=["General Sentiment Score", "Avg_Topic_Sentiment"])

    # Flatten multi-level columns
    sentiment_wide.columns = [f'{val}_{stock}' for val, stock in sentiment_wide.columns]

    # Reset index to make Date a column
    sentiment_wide = sentiment_wide.reset_index()

    # Merge stock data with sentiment data
    stock_data_with_sentiment = pd.merge(df_stock, sentiment_wide, on="Date", how="left")

    # Debug: Print column names after merging
    utils.log_message("Column names after merging:")
    for i, col in enumerate(stock_data_with_sentiment.columns[:10]):
        utils.log_message(f"  {i}: {col}")
    if len(stock_data_with_sentiment.columns) > 10:
        utils.log_message(f"  ... and {len(stock_data_with_sentiment.columns) - 10} more columns")
else:
    # If no sentiment data, just use the stock data
    stock_data_with_sentiment = df_stock
    utils.log_message("No sentiment data found. Using only stock data.", level="WARNING")

### **4.5️⃣ Handle Missing PE Ratio and EPS Values**
def fill_missing_financial_metrics(df):
    """
    Fill missing PE ratio and EPS values with appropriate defaults or computed values
    """
    utils.log_message("Handling missing PE ratio and EPS values...")
    
    # Identify PE ratio and EPS columns
    pe_ratio_columns = [col for col in df.columns if "PE_Ratio" in col]
    eps_columns = [col for col in df.columns if "EPS" in col]
    
    utils.log_message(f"Found {len(pe_ratio_columns)} PE ratio columns and {len(eps_columns)} EPS columns")
    
    # For each stock, compute mean PE ratio and EPS from available data
    for stock in set([col.split('_')[0] for col in pe_ratio_columns]):
        pe_col = next((col for col in pe_ratio_columns if col.startswith(f"{stock}_")), None)
        eps_col = next((col for col in eps_columns if col.startswith(f"{stock}_")), None)
        
        if pe_col and pe_col in df.columns:
            # Calculate mean PE ratio from non-empty values
            pe_values = df[pe_col].dropna()
            if len(pe_values) > 0:
                # Check if the corresponding EPS is negative
                if eps_col and eps_col in df.columns:
                    eps_values = df[eps_col].dropna()
                    if len(eps_values) > 0 and eps_values.mean() < 0:
                        # If EPS is negative, use -1 for PE ratio
                        utils.log_message(f"Company {stock} has negative EPS, setting PE ratio to -1")
                        df[pe_col] = df[pe_col].fillna(-1)
                    else:
                        pe_mean = pe_values.mean()
                        utils.log_message(f"Mean PE ratio for {stock}: {pe_mean:.2f} (from {len(pe_values)} values)")
                        # Fill missing values with the mean
                        df[pe_col] = df[pe_col].fillna(pe_mean)
                else:
                    pe_mean = pe_values.mean()
                    utils.log_message(f"Mean PE ratio for {stock}: {pe_mean:.2f} (from {len(pe_values)} values)")
                    # Fill missing values with the mean
                    df[pe_col] = df[pe_col].fillna(pe_mean)
            else:
                # If no values available, use -1 to indicate loss
                utils.log_message(f"No PE ratio data for {stock}, using -1 to indicate potential loss")
                df[pe_col] = df[pe_col].fillna(-1)
        
        if eps_col and eps_col in df.columns:
            # Calculate mean EPS from non-empty values
            eps_values = df[eps_col].dropna()
            if len(eps_values) > 0:
                eps_mean = eps_values.mean()
                utils.log_message(f"Mean EPS for {stock}: {eps_mean:.2f} (from {len(eps_values)} values)")
                # Fill missing values with the mean
                df[eps_col] = df[eps_col].fillna(eps_mean)
            else:
                # If no values available, use a reasonable default (-1.0 for loss)
                utils.log_message(f"No EPS data for {stock}, using -1.0 to indicate potential loss")
                df[eps_col] = df[eps_col].fillna(-1.0)
    
    return df

# Apply the function to fill missing financial metrics
stock_data_with_sentiment = fill_missing_financial_metrics(stock_data_with_sentiment)

### **5️⃣ Normalize Stock Prices, Volume, and Sentiment**
# Ensure all columns exist before scaling
existing_close_columns = [col for col in close_columns if col in stock_data_with_sentiment.columns]
existing_volume_columns = [col for col in volume_columns if col in stock_data_with_sentiment.columns]

if existing_close_columns and existing_volume_columns:
    utils.log_message(f"Normalizing {len(existing_close_columns)} close columns and {len(existing_volume_columns)} volume columns")
    
    # Normalize close prices
    stock_data_with_sentiment, close_scaler = utils.normalize_data(
        stock_data_with_sentiment, 
        columns=existing_close_columns,
        return_scaler=True
    )
    # Save the close price scaler
    joblib.dump(close_scaler, os.path.join(data_dir, "close_scaler.joblib"))
    utils.log_message("Saved close price scaler for denormalization")
    
    # Normalize volume
    stock_data_with_sentiment, volume_scaler = utils.normalize_data(
        stock_data_with_sentiment, 
        columns=existing_volume_columns,
        return_scaler=True
    )
    # Save the volume scaler
    joblib.dump(volume_scaler, os.path.join(data_dir, "volume_scaler.joblib"))
    utils.log_message("Saved volume scaler for denormalization")

if has_sentiment_data:
    sentiment_columns = [col for col in stock_data_with_sentiment.columns if "General Sentiment Score" in col or "Avg_Topic_Sentiment" in col]
    if sentiment_columns:  # Check if there are any sentiment columns
        utils.log_message(f"Normalizing {len(sentiment_columns)} sentiment columns")
        # Use the utility function for normalization
        stock_data_with_sentiment = utils.normalize_data(
            stock_data_with_sentiment,
            columns=sentiment_columns
        )

### **6️⃣ Reorder Columns by Stock**
# Extract unique stock tickers from close columns
stocks = utils.extract_stock_tickers(close_columns)

utils.log_message(f"Detected {len(stocks)} unique stocks: {stocks}")

# Define the order of columns for each stock, checking for their presence
ordered_columns = []
for stock in stocks:
    # Look for columns that match this stock
    stock_columns = [col for col in stock_data_with_sentiment.columns if col.startswith(f"{stock}_")]
    # Add only existing columns
    ordered_columns.extend(stock_columns)

# Reorder the DataFrame columns, ensuring Date is first
ordered_columns = ["Date"] + [col for col in ordered_columns if col != "Date"]
# Remove duplicates while preserving order
ordered_columns = list(dict.fromkeys(ordered_columns))
# Ensure all columns exist in the DataFrame
ordered_columns = [col for col in ordered_columns if col in stock_data_with_sentiment.columns]

# Add any remaining columns that weren't included
remaining_columns = [col for col in stock_data_with_sentiment.columns if col not in ordered_columns and col != "Date"]
ordered_columns = ["Date"] + ordered_columns[1:] + remaining_columns

# Reorder the DataFrame columns
stock_data_with_sentiment = stock_data_with_sentiment[ordered_columns]

### **7️⃣ Fill NaN Values**
# Fill missing sentiment scores with 0 (neutral sentiment)
if has_sentiment_data:
    sentiment_columns = [col for col in stock_data_with_sentiment.columns if "General Sentiment Score" in col or "Avg_Topic_Sentiment" in col]
    if sentiment_columns:
        utils.log_message(f"Filling NaN values in {len(sentiment_columns)} sentiment columns")
        stock_data_with_sentiment[sentiment_columns] = stock_data_with_sentiment[sentiment_columns].fillna(0)

# Ensure forward fill and backfill are applied correctly to RSI columns
rsi_columns = [col for col in stock_data_with_sentiment.columns if "RSI" in col]
if rsi_columns:
    utils.log_message(f"Filling NaN values in {len(rsi_columns)} RSI columns")
    stock_data_with_sentiment[rsi_columns] = stock_data_with_sentiment[rsi_columns].ffill().bfill()

# Fill NaN values in other columns
utils.log_message("Filling remaining NaN values")
stock_data_with_sentiment = stock_data_with_sentiment.ffill().bfill()

### **8️⃣ Save the Updated Dataset**
utils.log_message(f"Saving preprocessed data with {len(stock_data_with_sentiment)} rows and {len(stock_data_with_sentiment.columns)} columns")
stock_data_with_sentiment.to_csv(output_file, index=False)

utils.log_message(f"✅ Data preprocessing complete. Saved to: {output_file}")

# Clean up memory
utils.memory_cleanup()

