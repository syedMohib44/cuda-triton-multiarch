import torch
import torch.nn as nn

# Import the kernels directly from the copied folders
from kernels.flash_attention_full import flash_attention_full_triton #
from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton #
from kernels.rmsnorm import rmsnorm_triton #

class OptimizedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # We still define standard weights to hold the trainable parameters
        self.norm1_weight = nn.Parameter(torch.ones(hidden_dim))
        self.norm2_weight = nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, x):
        # 1. Triton RMSNorm
        x_norm = rmsnorm_triton(x, self.norm1_weight)
        
        # 2. Triton FlashAttention (Fast and memory efficient)
        q, k, v = self.get_qkv(x_norm)
        attn_out = flash_attention_full_triton(q, k, v, causal=True) #
        x = x + attn_out
        
        # 3. Fused RMSNorm + SwiGLU 
        # This replaces three separate operations with a single ultra-fast GPU call
        gate, up = self.get_ffn_projections(x)
        ffn_out = fused_rmsnorm_swiglu_triton(x, gate, self.norm2_weight) #
        print ( "====" )
        
        return x + ffn_out
    
