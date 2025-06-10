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

        self.freqs_complex = precompute_theta_pos_freq(self.args.dim//self.args.n_heads, self.args.max_seq_len*2,device = self.args.device)

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

        
