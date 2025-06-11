import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    dim: int = 4096 
    n_layers: int = 32 
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # for KV cache
    max_batch_size: int = 32 
    max_seq_len: int = 2048


# see original paper: https://arxiv.org/pdf/2104.09864.pdf for details
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):

    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)

    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)

    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()

    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex

    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch_size, seq_len, dim)
        return self.weight*self._norm(x.float()).type_as(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.args = config
        self.n_heads = config.n_heads


    def forward(self, x: torch.Tensor, start_position: int) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        
        # Apply attention
        x_attn = self.attention(x, x, x)[0]
        x = x + self.norm1(x_attn)

        # Apply feed-forward network
        x_ffn = self.ffn(x)
        x = x + self.norm2(x_ffn)

        return x

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        assert config.vocab_size != -1, "vocab_size must be set before initializing the model"
        
        self.args = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(EncoderBlock(config)) # we have n sequential encoder blocks
        
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2,device = self.args.device)

    def forward(self, x: torch.Tensor, start_position : int) -> torch.Tensor:
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        assert seq_len ==1, "The model only supports single token inputs for now. So only inference is supported. as we are using KV cache."

        # Get token embeddings
        x = self.tok_embeddings(x) #(batch_size, seq_len) -> (batch_size,seq_len,dim)

        freqs_complex = self.freqs_complex[start_position:start_position + seq_len]
        for layer in self.layers:
            x = layer(x, start_position)
        
        x = self.norm(x)
        output = self.output(x).float()
        return output

        
