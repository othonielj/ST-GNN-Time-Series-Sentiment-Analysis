import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Create output directory if it doesn't exist
output_dir = 'eda_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load edge list, node features, edge types, and edge weights
edge_index = torch.load('edge_index.pt')
node_features = torch.load('node_features.pt')
edge_types = torch.load('edge_types.pt')
edge_weights = torch.load('edge_weights.pt')

# Load node labels if available
try:
    node_labels_dict = torch.load('node_labels.pt')
except:
    # Fallback to generic labels if file doesn't exist
    node_labels_dict = {i: f"Stock {i}" for i in range(node_features.size(0))}

# Define colors for different edge types
edge_type_colors = {
    'price': '#1f77b4',       # blue
    'volume': '#ff7f0e',      # orange
    'rsi': '#2ca02c',         # green
    'macd': '#d62728',        # red
    'sentiment': '#9467bd',   # purple
    'topic_sentiment': '#8c564b'  # brown
}

edge_type_styles = {
    'price': 'solid',
    'volume': 'dashed',
    'rsi': 'dotted',
    'macd': 'dashdot',
    'sentiment': (0, (3, 1, 1, 1)),  # loosely dashdotted
    'topic_sentiment': (0, (3, 5, 1, 5))  # dashdotdotted
}

# Reconstruct the graph using NetworkX
G = nx.Graph()

# Add edges with their types and weights
for i in range(edge_index.size(1)):
    source = edge_index[0, i].item()
    target = edge_index[1, i].item()
    
    # Find the corresponding edge in edge_types and edge_weights
    # This is a bit tricky since edge_index doesn't maintain the same order
    edge_found = False
    for idx, (u, v) in enumerate(G.edges()):
        if (u == source and v == target) or (u == target and v == source):
            edge_found = True
            break
    
    if not edge_found:
        # If edge not found, add it with default attributes
        # We'll update them later
        G.add_edge(source, target, weight=1.0, types=['unknown'])

# Add node features to the graph
for i in range(node_features.size(0)):
    G.nodes[i]['features'] = node_features[i].tolist()

# Update edge attributes with the correct types and weights
for i, (u, v) in enumerate(G.edges()):
    if i < len(edge_types):
        G[u][v]['types'] = edge_types[i]
        G[u][v]['weight'] = edge_weights[i].item()

# Create a custom colormap for node features
node_cmap = plt.cm.viridis

# Visualize the graph
plt.figure(figsize=(16, 12))
ax = plt.gca()  # Get current axes

# Use a deterministic layout for consistency
pos = nx.spring_layout(G, seed=42)

# Draw nodes with color based on the first feature (e.g., average close price)
# and size based on the second feature (e.g., volume)
node_colors = [features[0] for _, features in G.nodes(data='features')]
node_sizes = [300 + 1000 * features[1] for _, features in G.nodes(data='features')]  # Scale volume for visibility

nodes = nx.draw_networkx_nodes(
    G, pos, 
    node_color=node_colors, 
    node_size=node_sizes,
    cmap=node_cmap, 
    alpha=0.8,
    ax=ax
)

# Draw edges with different colors and styles for different relationship types
for edge_type in edge_type_colors.keys():
    # Get edges of this type
    edges_of_type = [(u, v) for u, v in G.edges() if edge_type in G[u][v]['types']]
    
    if edges_of_type:
        # Get edge weights for this type
        edge_weights_of_type = [G[u][v]['weight'] for u, v in edges_of_type]
        
        # Scale edge widths based on weights
        edge_widths = [1 + 3 * weight for weight in edge_weights_of_type]
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=edges_of_type, 
            edge_color=edge_type_colors[edge_type],
            style=edge_type_styles[edge_type],
            width=edge_widths,
            alpha=0.6,
            ax=ax
        )

# Draw labels using the loaded node labels
nx.draw_networkx_labels(G, pos, labels=node_labels_dict, font_size=10, font_weight='bold', ax=ax)

# Add color bar for nodes
sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Normalized Close Price')

# Add legend for edges
legend_patches = []
for edge_type, color in edge_type_colors.items():
    # Count edges of this type
    count = sum(1 for u, v in G.edges() if edge_type in G[u][v]['types'])
    if count > 0:  # Only add to legend if there are edges of this type
        legend_patches.append(
            mpatches.Patch(
                color=color, 
                label=f'{edge_type.replace("_", " ").title()} ({count})',
                alpha=0.6
            )
        )

# Add legend for node size
legend_patches.append(
    mpatches.Circle((0, 0), radius=5, color='gray', alpha=0.3, label='Node Size = Trading Volume')
)

plt.legend(handles=legend_patches, loc='upper right', title='Edge Types')

# Add title and adjust layout
plt.title('Enhanced Stock Relationship Graph', fontsize=16)
plt.tight_layout()

# Add a text box with graph statistics
stats_text = (
    f"Graph Statistics:\n"
    f"Nodes: {G.number_of_nodes()}\n"
    f"Edges: {G.number_of_edges()}\n"
    f"Node Features: {node_features.size(1)}\n"
    f"Edge Types: {len(set(t for types in edge_types for t in types))}"
)
plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Save to the output directory
plt.savefig(f'{output_dir}/stock_relationship_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graph visualization saved as '{output_dir}/stock_relationship_graph.png'")

# Optional: Create a separate visualization for each edge type
if True:  # Set to True to enable this feature
    for edge_type in edge_type_colors.keys():
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=node_cmap, alpha=0.8, ax=ax)
        
        # Draw only edges of this type
        edges_of_type = [(u, v) for u, v in G.edges() if edge_type in G[u][v]['types']]
        if edges_of_type:
            edge_weights_of_type = [G[u][v]['weight'] for u, v in edges_of_type]
            edge_widths = [1 + 3 * weight for weight in edge_weights_of_type]
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=edges_of_type, 
                edge_color=edge_type_colors[edge_type],
                style=edge_type_styles[edge_type],
                width=edge_widths,
                alpha=0.7,
                ax=ax
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=node_labels_dict, font_size=10, font_weight='bold', ax=ax)
        
        # Add title
        plt.title(f'Stock Relationships: {edge_type.replace("_", " ").title()} Edges', fontsize=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'{output_dir}/stock_graph_{edge_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization for {edge_type} edges saved as '{output_dir}/stock_graph_{edge_type}.png'")

# Create a new function for the price histogram visualization
def create_price_histogram_matrix():
    """
    Create a matrix visualization with price histograms in the upper triangle
    and correlation values in the lower triangle.
    """
    print("Creating price histogram matrix visualization...")
    
    # Load the stock data to create histograms
    try:
        stock_data = pd.read_csv('data/stock_data_preprocessed.csv')
        print("Loaded stock data for histograms")
    except Exception as e:
        print(f"Warning: Could not load stock data for histograms: {str(e)}. Using random data instead.")
        # Create dummy data if the file doesn't exist
        stock_data = pd.DataFrame(np.random.randn(100, len(node_labels_dict)), 
                                columns=[node_labels_dict[i] for i in range(len(node_labels_dict))])

    # Create a new figure
    fig = plt.figure(figsize=(16, 14))
    
    # Get the number of nodes as an integer
    num_nodes = len(node_labels_dict)
    
    # Create a grid of subplots
    gs = GridSpec(num_nodes, num_nodes, figure=fig)
    
    # Get node labels in order
    node_labels = [node_labels_dict[i] for i in range(num_nodes)]
    
    # Create masks for the upper and lower triangles
    mask_upper = np.triu_indices(num_nodes, k=1)  # Upper triangle without diagonal
    mask_lower = np.tril_indices(num_nodes, k=-1)  # Lower triangle without diagonal
    
    # Create a correlation matrix for each edge type
    corr_matrices = {}
    for edge_type in edge_type_colors.keys():
        corr_matrices[edge_type] = np.zeros((num_nodes, num_nodes))
        for i, j in G.edges():
            if edge_type in G[i][j]['types']:
                corr_matrices[edge_type][i, j] = G[i][j]['weight']
                corr_matrices[edge_type][j, i] = G[i][j]['weight']  # Mirror for undirected graph
    
    # Create a composite correlation matrix (maximum correlation across all types)
    composite_corr = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                max_corr = max(corr_matrices[edge_type][i, j] for edge_type in edge_type_colors.keys())
                composite_corr[i, j] = max_corr
    
    # Create a color matrix for the lower triangle
    color_matrix_lower = np.zeros((num_nodes, num_nodes, 3))
    for i, j in zip(*mask_lower):
        # Determine which edge types connect these nodes
        edge_types_present = []
        for edge_type in edge_type_colors.keys():
            if corr_matrices[edge_type][i, j] > 0:
                edge_types_present.append(edge_type)
        
        if edge_types_present:
            # Mix colors of all present edge types
            for edge_type in edge_types_present:
                color = edge_type_colors[edge_type].lstrip('#')
                rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                weight = corr_matrices[edge_type][i, j] / sum(corr_matrices[et][i, j] for et in edge_types_present)
                for k in range(3):
                    color_matrix_lower[i, j, k] += rgb[k] * weight
    
    # Normalize the color matrix
    max_val = color_matrix_lower.max()
    if max_val > 0:
        color_matrix_lower = color_matrix_lower / max_val
    
    # Plot the lower triangle with correlation values
    for i, j in zip(*mask_lower):
        ax = fig.add_subplot(gs[i, j])
        
        # Set background color based on edge types
        if np.any(color_matrix_lower[i, j]):
            ax.set_facecolor(color_matrix_lower[i, j])
        
        # Add correlation value text
        if composite_corr[i, j] > 0:
            ax.text(0.5, 0.5, f"{composite_corr[i, j]:.2f}", 
                    ha='center', va='center', fontsize=10, 
                    color='white' if np.mean(color_matrix_lower[i, j]) > 0.5 else 'black')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Plot histograms in the upper triangle
    for i, j in zip(*mask_upper):
        ax = fig.add_subplot(gs[i, j])
        
        # Get the stock tickers
        stock_i = node_labels[i]
        stock_j = node_labels[j]
        
        try:
            # Find columns for these stocks - specifically looking for Close prices
            cols_i = [col for col in stock_data.columns if stock_i in col and 'Close' in col]
            cols_j = [col for col in stock_data.columns if stock_j in col and 'Close' in col]
            
            if cols_i and cols_j:
                # Get the data
                data_i = stock_data[cols_i[0]].values
                data_j = stock_data[cols_j[0]].values
                
                # Calculate correlation
                corr = np.corrcoef(data_i, data_j)[0, 1]
                
                # Plot a regular histogram of stock_i's prices
                n, bins, patches = ax.hist(data_i, bins=15, alpha=0.7, color=edge_type_colors['price'])
                
                # Add a title with the stock ticker and correlation
                ax.text(0.5, 0.9, f"{stock_i}\nr={corr:.2f}", transform=ax.transAxes, 
                        ha='center', va='center', fontsize=8, color='black')
                
                # Add a small vertical line for the mean
                ax.axvline(np.mean(data_i), color='red', linestyle='dashed', linewidth=1)
            else:
                # If no data, just show a placeholder
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
        except Exception as e:
            # If there's an error, show a placeholder
            ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha='center', va='center', fontsize=6)
        
        # Remove most axis elements but keep a minimal frame
        ax.set_xticks([])
        ax.set_yticks([])
        # Keep a thin frame for visual separation
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    # Add diagonal labels
    for i in range(num_nodes):
        ax = fig.add_subplot(gs[i, i])
        ax.text(0.5, 0.5, node_labels[i], ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_facecolor('lightgray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Add a title
    plt.suptitle('Stock Relationship Matrix: Price Histograms (Upper) & Correlations (Lower)', fontsize=16)
    
    # Add legend for edge types
    legend_patches = []
    for edge_type, color in edge_type_colors.items():
        if np.any(corr_matrices[edge_type]):  # Only add to legend if there are edges of this type
            legend_patches.append(
                mpatches.Patch(
                    color=color, 
                    label=edge_type.replace("_", " ").title(),
                    alpha=0.7
                )
            )
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches), 
               bbox_to_anchor=(0.5, 0.02), title='Edge Types')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(f'{output_dir}/stock_price_histograms_and_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced correlation matrix with price histograms saved as '{output_dir}/stock_price_histograms_and_correlations.png'")

# Call the function to create the price histogram matrix
create_price_histogram_matrix() 