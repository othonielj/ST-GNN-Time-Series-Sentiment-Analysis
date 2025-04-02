import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import os
import logging

# Import configuration settings
from M_config import (
    NODE_FEATURES,
    HIDDEN_DIM,
    TEMPORAL_DIM,
    OUTPUT_DIM,
    FORECAST_HORIZON,
    NUM_GNN_LAYERS,
    NUM_TEMPORAL_LAYERS,
    EDGE_TYPES,
    DROPOUT,
    TEMPORAL_CELL,
    DEVICE,
    LOG_FILE_PATH,
    USE_BATCH_NORM,
    USE_LAYER_NORM,
    USE_RESIDUAL,
    USE_ATTENTION,
    NUM_HEADS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH
)
logger = logging.getLogger(__name__)

# Use device from config
device = DEVICE
logger.info(f"Using device: {device}")

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if USE_LAYER_NORM else nn.BatchNorm1d(dim)
        self.norm2 = nn.LayerNorm(dim) if USE_LAYER_NORM else nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x + identity)
        return x

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_types):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_types = edge_types
        
        # Create a linear transformation for each edge type
        self.edge_linears = nn.ModuleDict({
            edge_type: nn.Linear(in_channels, out_channels)
            for edge_type in edge_types
        })
        
        # Linear transformation for self-loop
        self.self_linear = nn.Linear(in_channels, out_channels)
        
        # Attention weights for combining outputs from different edge types
        self.attention = nn.Parameter(torch.ones(len(edge_types) + 1) / (len(edge_types) + 1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_types, edge_weights=None):
        # x: [batch_size * num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_types: [num_edges] (list of strings)
        # edge_weights: [num_edges] or None
        
        # Initialize output tensor
        outputs = []
        
        # Process each edge type
        for edge_type in self.edge_types:
            # Get edges of current type
            mask = [et == edge_type for et in edge_types]
            mask = torch.tensor(mask, device=x.device)
            if not torch.any(mask):
                continue
            
            current_edges = edge_index[:, mask]
            
            # Get source and target nodes
            src, dst = current_edges
            
            # Apply linear transformation to source nodes
            transformed = self.edge_linears[edge_type](x[src])
            
            # Apply edge weights if provided
            if edge_weights is not None:
                transformed = transformed * edge_weights[mask].unsqueeze(1)
            
            # Aggregate messages at target nodes
            out = torch.zeros_like(x)
            out.index_add_(0, dst, transformed)
            outputs.append(out)
        
        # Add self-loop transformation
        self_out = self.self_linear(x)
        outputs.append(self_out)
        
        # Stack and combine outputs using attention weights
        if outputs:
            stacked = torch.stack(outputs, dim=-1)  # [batch_size * num_nodes, out_channels, num_edge_types + 1]
            out = torch.sum(stacked * F.softmax(self.attention, dim=0), dim=-1)  # [batch_size * num_nodes, out_channels]
        else:
            out = self_out
        
        return F.gelu(out)  # Apply non-linearity

class TemporalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project input to hidden dimension if needed
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional
    
    def forward(self, x):
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Process through LSTM
        output, _ = self.rnn(x)
        
        # Apply normalization
        output = self.norm(output)
        return output

class STGNN(nn.Module):
    def __init__(self, node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM, temporal_dim=TEMPORAL_DIM, 
                 output_dim=OUTPUT_DIM, num_gnn_layers=NUM_GNN_LAYERS, num_temporal_layers=NUM_TEMPORAL_LAYERS,
                 edge_types=EDGE_TYPES, dropout=DROPOUT):
        super().__init__()
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(
                in_channels=node_features if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                edge_types=edge_types
            ) for i in range(num_gnn_layers)
        ])
        
        # Temporal processing with bidirectional LSTM
        self.temporal = TemporalModule(
            input_dim=hidden_dim,  # Input from GNN layers
            hidden_dim=temporal_dim,
            num_layers=num_temporal_layers
        )
        
        # Output layers with residual connections
        self.output_layer1 = nn.Linear(temporal_dim * 2, hidden_dim)  # *2 for bidirectional
        self.output_layer2 = nn.Linear(hidden_dim + node_features, hidden_dim)
        self.output_layer3 = nn.Linear(hidden_dim, output_dim)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_types, edge_weights=None):
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Process each timestep through GNN layers
        temporal_out = []
        for t in range(seq_len):
            current_x = x[:, t]  # [batch_size, num_nodes, features]
            current_x = current_x.reshape(batch_size * num_nodes, features)  # [batch_size * num_nodes, features]
            
            # Process through GNN layers
            h = current_x
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index, edge_types, edge_weights)
                h = self.dropout(h)
            
            # Reshape back to [batch_size, num_nodes, hidden_dim]
            h = h.view(batch_size, num_nodes, HIDDEN_DIM)
            temporal_out.append(h)
        
        # Stack temporal sequence
        temporal_out = torch.stack(temporal_out, dim=1)  # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # Reshape for temporal processing
        temporal_out = temporal_out.permute(0, 2, 1, 3)  # [batch_size, num_nodes, seq_len, hidden_dim]
        temporal_out = temporal_out.reshape(-1, seq_len, HIDDEN_DIM)  # [batch_size * num_nodes, seq_len, hidden_dim]
        temporal_out = self.temporal(temporal_out)  # [batch_size * num_nodes, seq_len, temporal_dim * 2]
        
        # Get final prediction
        out = F.gelu(self.output_layer1(temporal_out[:, -FORECAST_HORIZON:]))  # [batch_size * num_nodes, forecast_horizon, hidden_dim]
        
        # Add skip connection from input features
        input_features = x[:, -1:]  # Use last timestep's features [batch_size, 1, num_nodes, features]
        input_features = input_features.expand(-1, FORECAST_HORIZON, -1, -1)  # [batch_size, forecast_horizon, num_nodes, features]
        input_features = input_features.reshape(batch_size * num_nodes, FORECAST_HORIZON, -1)  # [batch_size * num_nodes, forecast_horizon, features]
        
        out = torch.cat([out, input_features], dim=-1)  # [batch_size * num_nodes, forecast_horizon, hidden_dim + features]
        out = self.norm1(F.gelu(self.output_layer2(out)))  # [batch_size * num_nodes, forecast_horizon, hidden_dim]
        out = self.dropout(out)
        out = self.output_layer3(out)  # [batch_size * num_nodes, forecast_horizon, output_dim]
        
        # Reshape back to [batch_size, forecast_horizon, num_nodes]
        out = out.view(batch_size, num_nodes, FORECAST_HORIZON, OUTPUT_DIM)
        out = out.transpose(1, 2)  # [batch_size, forecast_horizon, num_nodes, output_dim]
        out = out.squeeze(-1)  # [batch_size, forecast_horizon, num_nodes]
        
        return out


# Example of model instantiation (for documentation purposes)
if __name__ == "__main__":
    # This section won't run when imported, only if the file is executed directly
    
    # Example model parameters
    model_params = {
        'node_features': 8,
        'hidden_dim': 128,
        'temporal_dim': 128,
        'output_dim': 1,
        'forecast_horizon': 7,
        'num_gnn_layers': 3,
        'num_temporal_layers': 2,
        'edge_types': ['price', 'volume', 'rsi', 'macd', 'sentiment', 'topic_sentiment'],
        'dropout': 0.3,
        'temporal_cell': 'GRU',
        'num_stocks': 10
    }
    
    # Create model
    model = STGNN(**model_params)
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {device}") 