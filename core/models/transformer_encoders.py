import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=1024):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        pe = self.pe[:, :x.size(1), :]
        return x + pe

class TrajectoryAttentivePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Compute attention scores
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weight and sum all timesteps
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # [batch_size, d_model]
        return weighted_sum, attention_weights

class SelfAttentionTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers,
                embedding_dim, dropout, norm_first,
                use_input_dropout=False,
                input_dropout_rate=0.2,
                trajectory_pooling=True):
        super().__init__()
        self.use_input_dropout = use_input_dropout
        self.trajectory_pooling = trajectory_pooling
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(input_dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2, # you can adjust this factor for Lean model
            dropout=dropout,
            norm_first=norm_first,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )

        self._apply_trajectory_pooling = TrajectoryAttentivePooling(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the input to the hidden dimension
        x = self.input_projection(x) # scale by d_model but not required since its small here
        
        if self.use_input_dropout:
            x = self.dropout(x)
            
        x = self.positional_encoding(x) # [batch_size, seq_len, d_model]

        transformer_output = self.transformer_encoder(x) # [batch_size, seq_len, d_model]

        if self.trajectory_pooling:
            context, attention_weights = self._apply_trajectory_pooling(transformer_output) # [batch_size, d_model]
        else:
            # was the default
            context = torch.mean(transformer_output, dim=1) # [batch_size, d_model]
        
        embedding = self.fc(context) # [batch_size, embedding_dim]
        return embedding