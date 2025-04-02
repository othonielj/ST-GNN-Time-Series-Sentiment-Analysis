import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create output directory if it doesn't exist
output_dir = 'eda_visualizations'
os.makedirs(output_dir, exist_ok=True)

def load_data():
    """Load and prepare the preprocessed stock data."""
    print("Loading preprocessed stock data...")
    df = pd.read_csv('data/stock_data_preprocessed.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index for time series analysis
    df_indexed = df.set_index('Date')
    
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df, df_indexed

def extract_stock_data(df):
    """Extract stock-specific data from the combined dataframe."""
    # Identify column patterns for different metrics
    close_columns = [col for col in df.columns if "Close_" in col]
    volume_columns = [col for col in df.columns if "Volume_" in col]
    sentiment_columns = [col for col in df.columns if "General Sentiment Score_" in col]
    topic_sentiment_columns = [col for col in df.columns if "Avg_Topic_Sentiment_" in col]
    
    # Extract unique stock tickers (remove duplicates with .1, .2, etc.)
    raw_tickers = [col.split('_')[-1] for col in close_columns]
    # Remove duplicates by taking only the base ticker (without .1, .2, etc.)
    base_tickers = set()
    for ticker in raw_tickers:
        base_ticker = ticker.split('.')[0]  # Get the part before any dot
        base_tickers.add(base_ticker)
    
    tickers = sorted(base_tickers)
    print(f"Found {len(tickers)} unique stocks: {tickers}")
    
    # Create a dictionary to store stock-specific dataframes
    stock_data = {}
    
    for ticker in tickers:
        # For each ticker, find the exact column for each metric
        close_col = None
        volume_col = None
        rsi_col = None
        macd_col = None
        pe_ratio_col = None
        eps_col = None
        sentiment_col = None
        topic_sentiment_col = None
        
        # Find the first matching column for each metric
        for col in close_columns:
            if col.endswith(f"_{ticker}"):
                close_col = col
                break
        
        for col in volume_columns:
            if col.endswith(f"_{ticker}"):
                volume_col = col
                break
        
        for col in df.columns:
            if "RSI__" in col and ticker in col:
                rsi_col = col
                break
        
        for col in df.columns:
            if "MACD__" in col and ticker in col:
                macd_col = col
                break
        
        for col in df.columns:
            if "PE_Ratio__" in col and ticker in col:
                pe_ratio_col = col
                break
        
        for col in df.columns:
            if "EPS__" in col and ticker in col:
                eps_col = col
                break
        
        for col in sentiment_columns:
            if col.endswith(f"_{ticker}"):
                sentiment_col = col
                break
        
        for col in topic_sentiment_columns:
            if col.endswith(f"_{ticker}"):
                topic_sentiment_col = col
                break
        
        # Create a new dataframe with selected columns
        selected_cols = []
        col_mapping = {}
        
        if close_col:
            selected_cols.append(close_col)
            col_mapping[close_col] = 'Close'
        
        if volume_col:
            selected_cols.append(volume_col)
            col_mapping[volume_col] = 'Volume'
        
        if rsi_col:
            selected_cols.append(rsi_col)
            col_mapping[rsi_col] = 'RSI'
        
        if macd_col:
            selected_cols.append(macd_col)
            col_mapping[macd_col] = 'MACD'
        
        if pe_ratio_col:
            selected_cols.append(pe_ratio_col)
            col_mapping[pe_ratio_col] = 'PE_Ratio'
        
        if eps_col:
            selected_cols.append(eps_col)
            col_mapping[eps_col] = 'EPS'
        
        if sentiment_col:
            selected_cols.append(sentiment_col)
            col_mapping[sentiment_col] = 'Sentiment'
        
        if topic_sentiment_col:
            selected_cols.append(topic_sentiment_col)
            col_mapping[topic_sentiment_col] = 'Topic_Sentiment'
        
        # If we found any columns, create a dataframe
        if selected_cols:
            # Add Date column
            selected_cols.append('Date')
            
            # Create dataframe with selected columns
            stock_df = df[selected_cols].copy()
            
            # Rename columns
            stock_df = stock_df.rename(columns=col_mapping)
            
            # Store in dictionary
            stock_data[ticker] = stock_df
        else:
            print(f"Warning: No columns found for ticker {ticker}")
    
    return stock_data, tickers

def calculate_returns(stock_data):
    """Calculate daily returns for each stock."""
    for ticker, df in stock_data.items():
        try:
            if 'Close' in df.columns:
                # Calculate returns
                df['Return'] = df['Close'].pct_change()
                
                # Calculate sentiment change if sentiment data exists
                if 'Sentiment' in df.columns:
                    df['Sentiment_Change'] = df['Sentiment'].diff()
                    
                    # Create sentiment categories
                    df['Sentiment_Category'] = pd.cut(
                        df['Sentiment'],
                        bins=[-float('inf'), -0.33, 0.33, float('inf')],
                        labels=['Negative', 'Neutral', 'Positive']
                    )
        except Exception as e:
            print(f"Error calculating returns for {ticker}: {e}")
    
    return stock_data

def plot_density_of_stock_prices(stock_data, tickers, output_dir):
    """Create density plots of stock prices."""
    plt.figure(figsize=(14, 8))
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
            
        df = stock_data[ticker]
        if 'Close' in df.columns:
            sns.kdeplot(df['Close'].dropna(), label=ticker, fill=True, alpha=0.3)
    
    plt.title('Density Plot of Normalized Stock Prices', fontsize=16)
    plt.xlabel('Normalized Price', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Stocks')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stock_price_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved density plot to {output_dir}/stock_price_density.png")

def plot_histograms_of_stock_prices(stock_data, tickers, output_dir):
    """Create histograms of stock price distributions."""
    # Filter out tickers that don't have data
    valid_tickers = [t for t in tickers if t in stock_data]
    
    # Create a grid of histograms
    n_stocks = len(valid_tickers)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    
    for i, ticker in enumerate(valid_tickers):
        df = stock_data[ticker]
        if 'Close' in df.columns:
            sns.histplot(df['Close'].dropna(), kde=True, ax=axes[i], color=f'C{i}')
            axes[i].set_title(f'{ticker} Price Distribution', fontsize=12)
            axes[i].set_xlabel('Normalized Price')
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stock_price_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram plot to {output_dir}/stock_price_histograms.png")

def plot_boxplots_for_outliers(stock_data, tickers, output_dir):
    """Create boxplots to identify outliers in stock prices."""
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
            
        df = stock_data[ticker]
        if 'Close' in df.columns:
            # Convert to list to ensure it's 1D
            boxplot_data.append(df['Close'].dropna().tolist())
            boxplot_labels.append(ticker)
    
    plt.figure(figsize=(14, 8))
    
    # Use tick_labels instead of labels for newer matplotlib versions
    box = plt.boxplot(boxplot_data, patch_artist=True, tick_labels=boxplot_labels)
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Boxplot of Stock Prices (Outlier Detection)', fontsize=16)
    plt.xlabel('Stock', fontsize=12)
    plt.ylabel('Normalized Price', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stock_price_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved boxplot to {output_dir}/stock_price_boxplots.png")

def plot_sentiment_vs_returns(stock_data, tickers, output_dir):
    """Create histplots of sentiment change vs. stock returns."""
    # Combine data from all stocks
    combined_data = pd.DataFrame()
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
            
        df = stock_data[ticker]
        if all(col in df.columns for col in ['Return', 'Sentiment_Change', 'Sentiment_Category']):
            temp_df = df[['Return', 'Sentiment_Change', 'Sentiment_Category']].copy()
            # Add a unique identifier to avoid duplicate indices
            temp_df['Ticker'] = ticker
            # Reset index to avoid duplicate indices when concatenating
            temp_df = temp_df.reset_index(drop=True)
            combined_data = pd.concat([combined_data, temp_df], ignore_index=True)
    
    # Remove NaN values
    combined_data = combined_data.dropna()
    
    if combined_data.empty:
        print("Warning: No valid data for sentiment vs returns plot")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Use a scatter plot instead of histplot for more reliable plotting
    sns.scatterplot(
        data=combined_data,
        x='Return',
        y='Sentiment_Change',
        hue='Sentiment_Category',
        palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
        alpha=0.7,
        s=50  # Point size
    )
    
    plt.title('Sentiment Change vs. Stock Returns', fontsize=16)
    plt.xlabel('Daily Return', fontsize=12)
    plt.ylabel('Sentiment Change', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_vs_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sentiment vs returns plot to {output_dir}/sentiment_vs_returns.png")
    
    # Create individual plots for each stock
    # Filter valid tickers
    valid_tickers = [t for t in tickers if t in stock_data]
    n_stocks = len(valid_tickers)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    
    for i, ticker in enumerate(valid_tickers):
        df = stock_data[ticker]
        if all(col in df.columns for col in ['Return', 'Sentiment_Change', 'Sentiment_Category']):
            df_clean = df[['Return', 'Sentiment_Change', 'Sentiment_Category']].dropna()
            
            if not df_clean.empty:
                sns.scatterplot(
                    data=df_clean,
                    x='Return',
                    y='Sentiment_Change',
                    hue='Sentiment_Category',
                    palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                    alpha=0.7,
                    ax=axes[i]
                )
                
                axes[i].set_title(f'{ticker}: Sentiment vs Returns', fontsize=12)
                axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[i].legend(title='Sentiment')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_vs_returns_by_stock.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual sentiment vs returns plots to {output_dir}/sentiment_vs_returns_by_stock.png")

def create_correlation_heatmap(df_indexed, output_dir):
    """Create a correlation heatmap of all numeric columns."""
    # Select only numeric columns
    numeric_df = df_indexed.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,  # Too many values to annotate
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    
    plt.title('Correlation Heatmap of Stock Data Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to {output_dir}/correlation_heatmap.png")
    
    # Create a more focused heatmap with just close prices
    close_cols = [col for col in numeric_df.columns if "Close_" in col]
    
    if close_cols:
        close_corr = numeric_df[close_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            close_corr,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            fmt='.2f'
        )
        
        # Clean up the labels
        labels = [col.split('_')[-1] for col in close_cols]
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        
        plt.title('Correlation Heatmap of Stock Prices', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/price_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved price correlation heatmap to {output_dir}/price_correlation_heatmap.png")

def create_time_series_plots(stock_data, tickers, output_dir):
    """Create time series plots of stock prices and sentiment."""
    # Create a figure for prices
    plt.figure(figsize=(14, 8))
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
            
        df = stock_data[ticker]
        if 'Close' in df.columns and 'Date' in df.columns:
            plt.plot(df['Date'], df['Close'], label=ticker)
    
    plt.title('Stock Price Time Series', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price', fontsize=12)
    plt.legend(title='Stocks')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stock_price_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved price time series to {output_dir}/stock_price_time_series.png")
    
    # Create a figure for sentiment
    plt.figure(figsize=(14, 8))
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
            
        df = stock_data[ticker]
        if 'Sentiment' in df.columns and 'Date' in df.columns:
            plt.plot(df['Date'], df['Sentiment'], label=ticker)
    
    plt.title('Stock Sentiment Time Series', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.legend(title='Stocks')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sentiment time series to {output_dir}/sentiment_time_series.png")

def main():
    """Main function to run all EDA visualizations."""
    print("Starting exploratory data analysis...")
    
    # Load data
    df, df_indexed = load_data()
    
    # Extract stock-specific data
    stock_data, tickers = extract_stock_data(df)
    
    # Calculate returns and sentiment changes
    stock_data = calculate_returns(stock_data)
    
    # Create visualizations
    plot_density_of_stock_prices(stock_data, tickers, output_dir)
    plot_histograms_of_stock_prices(stock_data, tickers, output_dir)
    plot_boxplots_for_outliers(stock_data, tickers, output_dir)
    plot_sentiment_vs_returns(stock_data, tickers, output_dir)
    create_correlation_heatmap(df_indexed, output_dir)
    create_time_series_plots(stock_data, tickers, output_dir)
    
    print("\nExploratory data analysis completed successfully!")
    print(f"All visualizations saved to the '{output_dir}' directory")

if __name__ == "__main__":
    main() 