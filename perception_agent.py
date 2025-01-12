import torch
import torch.nn as nn
from entmax import entmax15


class PerceptionAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, dropout=0.1):
        super(PerceptionAgent, self).__init__()
        
        # Initial embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Adaptive Sparse Transformer layer
        self.transformer = AdaptiveSparsityTransformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
    
    def predict_mask(self, state):
        # Ensure input has correct shape [batch_size, seq_len, features]
        if state.dim() == 2:
            state = state.unsqueeze(0)
        elif state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
                    
        x = self.input_embedding(state)
        x = self.transformer(x)
        
        # Take only the first sequence element's output
        x = x[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        # Generate single attention mask
        return self.output_layers(x)  # Shape: [batch_size, n_symptoms]

class AdaptiveSparsityTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(AdaptiveSparsityTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
        # Adaptive sparsity parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.sparsity_threshold = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        # Add shape validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch_size, seq_len, hidden_size), got shape {x.shape}")
            
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        # probabilities = entmax15(scores, dim=0)  # 稀疏化权重
        # Apply adaptive sparsity
        sparse_mask = (scores > self.sparsity_threshold).float()
        scores = scores * sparse_mask
        scores = scores / self.temperature
        
        # Apply softmax and dropout
        # attn = torch.softmax(scores, dim=-1)
        attn = entmax15(scores, dim=-1)  # differentiable, automatically adapt the sparsity level based on the input
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.output_linear(out) 