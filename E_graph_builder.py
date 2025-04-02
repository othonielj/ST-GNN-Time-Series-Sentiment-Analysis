import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import logging

# Import configuration settings
from M_config import (
    PRICE_THRESHOLD, 
    VOLUME_THRESHOLD, 
    RSI_THRESHOLD, 
    MACD_THRESHOLD, 
    SENTIMENT_THRESHOLD, 
    TOPIC_SENTIMENT_THRESHOLD,
    SAVE_PATH,
    LOG_FILE_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH
)
logger = logging.getLogger(__name__)

# Load preprocessed stock data (final version)
stock_data = pd.read_csv(f'{SAVE_PATH}stock_data_preprocessed.csv')

# Identify column patterns for different metrics
# Updated to match the actual column naming patterns in the preprocessed data
close_columns = [col for col in stock_data.columns if "Close" in col]
volume_columns = [col for col in stock_data.columns if "Volume" in col]
rsi_columns = [col for col in stock_data.columns if "RSI___" in col]
macd_columns = [col for col in stock_data.columns if "MACD___" in col]
pe_columns = [col for col in stock_data.columns if "PE_Ratio___" in col]
eps_columns = [col for col in stock_data.columns if "EPS___" in col]
gen_sentiment_columns = [col for col in stock_data.columns if "General Sentiment Score_" in col]
topic_sentiment_columns = [col for col in stock_data.columns if "Avg_Topic_Sentiment_" in col]

logger.info("Column patterns found:")
logger.info(f"Close columns: {len(close_columns)}")
logger.info(f"Volume columns: {len(volume_columns)}")
logger.info(f"RSI columns: {len(rsi_columns)}")
logger.info(f"MACD columns: {len(macd_columns)}")
logger.info(f"PE columns: {len(pe_columns)}")
logger.info(f"EPS columns: {len(eps_columns)}")
logger.info(f"General sentiment columns: {len(gen_sentiment_columns)}")
logger.info(f"Topic sentiment columns: {len(topic_sentiment_columns)}")

# Extract unique stock tickers that have all required columns
# Updated to match the actual column naming patterns
raw_tickers = [col.split('_')[0] for col in close_columns]
unique_tickers = []

# Only keep tickers that have all the necessary columns
for ticker in set(raw_tickers):
    has_close = any(f"{ticker}_Close" in col for col in close_columns)
    has_volume = any(f"{ticker}_Volume" in col for col in volume_columns)
    has_sentiment = f"General Sentiment Score_{ticker}" in stock_data.columns
    has_topic = f"Avg_Topic_Sentiment_{ticker}" in stock_data.columns
    
    if has_close and has_volume and has_sentiment and has_topic:
        unique_tickers.append(ticker)

# Sort for consistency
stock_tickers = sorted(unique_tickers)
logger.info(f"Found {len(stock_tickers)} valid stocks: {stock_tickers}")

# Create correlation matrices for different metrics
# We need to be careful with column names here
price_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
volume_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
rsi_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
macd_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
gen_sentiment_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
topic_sentiment_corr = pd.DataFrame(index=stock_tickers, columns=stock_tickers)

# Fill correlation matrices
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        # Find the correct column names for this pair of stocks
        close_i = [col for col in close_columns if col.startswith(f"{ticker_i}_")][0]
        close_j = [col for col in close_columns if col.startswith(f"{ticker_j}_")][0]
        volume_i = [col for col in volume_columns if col.startswith(f"{ticker_i}_")][0]
        volume_j = [col for col in volume_columns if col.startswith(f"{ticker_j}_")][0]
        
        # Calculate correlations
        price_corr.loc[ticker_i, ticker_j] = stock_data[close_i].corr(stock_data[close_j])
        volume_corr.loc[ticker_i, ticker_j] = stock_data[volume_i].corr(stock_data[volume_j])
        
        # For RSI and MACD, we need to be careful with column names
        rsi_i = [col for col in rsi_columns if col.startswith(f"{ticker_i}_")]
        rsi_j = [col for col in rsi_columns if col.startswith(f"{ticker_j}_")]
        if rsi_i and rsi_j:
            rsi_corr.loc[ticker_i, ticker_j] = stock_data[rsi_i[0]].corr(stock_data[rsi_j[0]])
        else:
            rsi_corr.loc[ticker_i, ticker_j] = 0
            
        macd_i = [col for col in macd_columns if col.startswith(f"{ticker_i}_")]
        macd_j = [col for col in macd_columns if col.startswith(f"{ticker_j}_")]
        if macd_i and macd_j:
            macd_corr.loc[ticker_i, ticker_j] = stock_data[macd_i[0]].corr(stock_data[macd_j[0]])
        else:
            macd_corr.loc[ticker_i, ticker_j] = 0
            
        # Sentiment correlations
        gen_sent_i = f"General Sentiment Score_{ticker_i}"
        gen_sent_j = f"General Sentiment Score_{ticker_j}"
        gen_sentiment_corr.loc[ticker_i, ticker_j] = stock_data[gen_sent_i].corr(stock_data[gen_sent_j])
        
        topic_sent_i = f"Avg_Topic_Sentiment_{ticker_i}"
        topic_sent_j = f"Avg_Topic_Sentiment_{ticker_j}"
        topic_sentiment_corr.loc[ticker_i, ticker_j] = stock_data[topic_sent_i].corr(stock_data[topic_sent_j])

# Create a NetworkX graph
G = nx.Graph()

# Add nodes with enhanced features
for ticker in stock_tickers:
    # Find the correct column names for this ticker
    close_cols = [col for col in close_columns if col.startswith(f"{ticker}_")]
    volume_cols = [col for col in volume_columns if col.startswith(f"{ticker}_")]
    rsi_cols = [col for col in rsi_columns if col.startswith(f"{ticker}_")]
    macd_cols = [col for col in macd_columns if col.startswith(f"{ticker}_")]
    pe_cols = [col for col in pe_columns if col.startswith(f"{ticker}_")]
    eps_cols = [col for col in eps_columns if col.startswith(f"{ticker}_")]
    
    # Calculate average values for each metric
    close_avg = stock_data[close_cols].mean().mean() if close_cols else 0
    volume_avg = stock_data[volume_cols].mean().mean() if volume_cols else 0
    rsi_avg = stock_data[rsi_cols].mean().mean() if rsi_cols else 0
    macd_avg = stock_data[macd_cols].mean().mean() if macd_cols else 0
    
    # Get sentiment features
    gen_sentiment_avg = stock_data[f"General Sentiment Score_{ticker}"].fillna(0).mean()
    topic_sentiment_avg = stock_data[f"Avg_Topic_Sentiment_{ticker}"].fillna(0).mean()
    
    # Try to get PE and EPS if available
    pe_avg = stock_data[pe_cols].mean().mean() if pe_cols else 0
    eps_avg = stock_data[eps_cols].mean().mean() if eps_cols else 0
    
    # Create node features vector
    node_features = [
        close_avg,                # Normalized close price
        volume_avg,               # Normalized volume
        rsi_avg,                  # Average RSI
        macd_avg,                 # Average MACD
        pe_avg,                   # Average PE Ratio
        eps_avg,                  # Average EPS
        gen_sentiment_avg,        # General sentiment
        topic_sentiment_avg       # Topic sentiment
    ]
    
    G.add_node(ticker, features=node_features)

# Define thresholds for different edge types
PRICE_THRESHOLD = 0.5
VOLUME_THRESHOLD = 0.4
RSI_THRESHOLD = 0.6
MACD_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.7
TOPIC_SENTIMENT_THRESHOLD = 0.6

# Function to add edges with type attribute
def add_typed_edge(G, ticker_i, ticker_j, weight, edge_type):
    if (ticker_i, ticker_j) in G.edges:
        # Edge already exists, update attributes
        G[ticker_i][ticker_j]['weight'] = max(G[ticker_i][ticker_j]['weight'], weight)
        G[ticker_i][ticker_j]['types'].append(edge_type)
    else:
        # Create new edge
        G.add_edge(ticker_i, ticker_j, weight=weight, types=[edge_type])

# Add edges based on price correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i != j:
            corr_value = price_corr.loc[ticker_i, ticker_j]
            if corr_value > PRICE_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'price')

# Add edges based on volume correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i != j:
            corr_value = volume_corr.loc[ticker_i, ticker_j]
            if corr_value > VOLUME_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'volume')

# Add edges based on RSI correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i != j:
            corr_value = rsi_corr.loc[ticker_i, ticker_j]
            if corr_value > RSI_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'rsi')

# Add edges based on MACD correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i != j:
            corr_value = macd_corr.loc[ticker_i, ticker_j]
            if corr_value > MACD_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'macd')

# Add edges based on general sentiment correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i < j:  # Only process each pair once
            corr_value = gen_sentiment_corr.loc[ticker_i, ticker_j]
            if corr_value > SENTIMENT_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'sentiment')

# Add edges based on topic sentiment correlation
for i, ticker_i in enumerate(stock_tickers):
    for j, ticker_j in enumerate(stock_tickers):
        if i < j:  # Only process each pair once
            corr_value = topic_sentiment_corr.loc[ticker_i, ticker_j]
            if corr_value > TOPIC_SENTIMENT_THRESHOLD:
                add_typed_edge(G, ticker_i, ticker_j, corr_value, 'topic_sentiment')

# For ST-GNN: Add temporal edges (optional)
# This connects each stock to itself across time steps
# Uncomment and modify if you want to include temporal edges
"""
for ticker in stock_tickers:
    # Add self-loops with temporal information
    G.add_edge(ticker, ticker, weight=1.0, types=['temporal'])
"""

# Convert NetworkX graph to PyTorch Geometric format
data = from_networkx(G)

# Extract node features as a tensor
node_features = torch.tensor([G.nodes[n]['features'] for n in G.nodes], dtype=torch.float)

# Save edge index, edge attributes, and node features
torch.save(data.edge_index, 'edge_index.pt')
torch.save(node_features, 'node_features.pt')

# Save edge types and weights
edge_types = []
edge_weights = []
for u, v, attrs in G.edges(data=True):
    edge_types.append(attrs['types'])
    edge_weights.append(attrs['weight'])

torch.save(edge_types, 'edge_types.pt')
torch.save(torch.tensor(edge_weights, dtype=torch.float), 'edge_weights.pt')

# Save node labels for visualization
node_labels = {i: node for i, node in enumerate(G.nodes())}
torch.save(node_labels, 'node_labels.pt')

# Print graph statistics
logger.info(f'✅ Enhanced graph successfully built!')
logger.info(f'🔹 Number of nodes: {data.num_nodes}')
logger.info(f'🔹 Number of edges: {data.num_edges}')
logger.info(f'🔹 Node feature dimensions: {node_features.size(1)}')

# Count edge types
edge_type_counts = {}
for edge in edge_types:
    for type_name in edge:
        if type_name in edge_type_counts:
            edge_type_counts[type_name] += 1
        else:
            edge_type_counts[type_name] = 1

logger.info(f'🔹 Edge type distribution:')
for edge_type, count in edge_type_counts.items():
    logger.info(f'   - {edge_type}: {count} edges')
