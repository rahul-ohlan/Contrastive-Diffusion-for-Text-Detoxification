"""
Diffusion Transformer model with cross-attention for text detoxification.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)
        self.to_v = nn.Linear(key_dim, query_dim)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, context_mask=None):
        h = self.num_heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(t.shape[0], -1, h, self.head_dim).transpose(1, 2), (q, k, v))

        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

        if context_mask is not None:
            mask = context_mask[:, None, None, :].float()
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(dots, dim=-1)
        self._last_attn_weights = attn  # Store for visualization

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, h * self.head_dim)
        return self.to_out(out)

    def get_attention_weights(self):
        return self._last_attn_weights

class DiffusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1, max_seq_len=150):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                CrossAttention(hidden_dim, input_dim, num_heads, dropout),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                )
            ]))
            
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.to_output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, timesteps, toxic_embeddings=None, toxic_mask=None, model_kwargs=None, **kwargs):
        """
        x: noisy embeddings [B, T, D]
        timesteps: [B]
        toxic_embeddings: [B, T, D] - can be passed directly or via model_kwargs
        toxic_mask: [B, T] - can be passed directly or via model_kwargs
        model_kwargs: dict - additional kwargs from diffusion model
        """
        # Extract toxic_embeddings and toxic_mask from model_kwargs if provided
        if model_kwargs is not None:
            toxic_embeddings = model_kwargs.get("toxic_embeddings", toxic_embeddings)
            toxic_mask = model_kwargs.get("toxic_mask", toxic_mask)
        
        t = self.time_embed(timesteps)
        
        h = self.input_projection(x)
        
        # Add time embeddings
        h = h + t[:, None, :]
        
        # Process through layers
        for norm1, cross_attn, norm2, ff in self.layers:
            # Cross attention
            h = h + cross_attn(norm1(h), toxic_embeddings, toxic_mask)
            # Feed forward
            h = h + ff(norm2(h))
            
        h = self.final_norm(h)
        return self.to_output(h)
    
    def get_embeds(self, input_ids):
        """
        The diffusion process calls this to get embeddings from input_ids.
        Since we're using pre-computed embeddings, this should never be called.
        """
        raise NotImplementedError("DiffusionTransformer uses pre-computed embeddings")
    
    def get_logits(self, hidden_repr):
        """
        The diffusion process calls this to get logits from hidden states.
        Since we're using pre-computed embeddings, this should never be called.
        """
        raise NotImplementedError("DiffusionTransformer uses pre-computed embeddings")
    
    def get_attention_weights(self, x, toxic_embeddings, toxic_mask=None):
        """Get attention weights from the last layer for visualization"""
        with torch.no_grad():
            h = self.input_projection(x)
            norm_h = self.layers[-1][0](h)  # Use last layer
            self.layers[-1][1](norm_h, toxic_embeddings, toxic_mask)
            return self.layers[-1][1].get_attention_weights() 