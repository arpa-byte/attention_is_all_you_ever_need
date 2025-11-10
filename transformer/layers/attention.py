import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Scaled Dot-Product Attention as described in "Attention Is All You Need".

    This function is the core component of the Multi-Head Attention module. It calculates
    attention scores and applies them to the value tensor.

    Args:
        q (torch.Tensor): The query tensor. Shape: (..., seq_len_q, d_k)
        k (torch.Tensor): The key tensor. Shape: (..., seq_len_k, d_k)
        v (torch.Tensor): The value tensor. Shape: (..., seq_len_v, d_v) 
                          Note: seq_len_k and seq_len_v must be the same.
        mask (torch.Tensor, optional): A boolean mask to prevent attention to certain positions.
                                       Shape: (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The context vector after applying attention. Shape: (..., seq_len_q, d_v)
            - attention_weights (torch.Tensor): The attention weights. Shape: (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism from "Attention Is All You Need".

    This module projects the queries, keys, and values into multiple "heads",
    computes attention for each head in parallel, and then concatenates the results.
    """
    def __init__(self, d_model: int, h: int):
        """
        Args:
            d_model (int): The total dimensionality of the input/output features.
            h (int): The number of parallel attention heads.
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): The query tensor. Shape: (batch_size, seq_len_q, d_model)
            k (torch.Tensor): The key tensor. Shape: (batch_size, seq_len_k, d_model)
            v (torch.Tensor): The value tensor. Shape: (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): The mask to be applied. Defaults to None.

        Returns:
            torch.Tensor: The final output tensor. Shape: (batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # 1. Linearly project the Q, K, V and split into h heads.
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Now q, k, v have shape: (batch_size, h, seq_len, d_k)

        # 2. Apply scaled dot-product attention for each head.
        x, attention_weights = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 3. Concatenate the heads and pass through a final linear layer.
        # Reshape x back to (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)


# Standalone Test for all components in this file
if __name__ == '__main__':
    # --- Part 1: Test scaled_dot_product_attention ---
    print("--- Running Standalone Test for Scaled Dot-Product Attention ---")
    q_sdpa = torch.randn(4, 5, 8)
    k_sdpa = torch.randn(4, 5, 8)
    v_sdpa = torch.randn(4, 5, 10)
    output_sdpa, _ = scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    assert output_sdpa.shape == (4, 5, 10)
    print("Scaled Dot-Product Attention test passed!")
    
    # --- Part 2: Test MultiHeadAttention ---
    print("\n--- Running Standalone Test for Multi-Head Attention ---")
    
    # Define hyperparameters for the test
    batch_size = 4
    seq_len = 10
    d_model = 512
    h = 8 # Number of heads
    
    # Create dummy input tensor
    # In a real scenario, q, k, and v can be different. For self-attention, they are the same.
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create the MultiHeadAttention module
    mha = MultiHeadAttention(d_model=d_model, h=h)
    
    # Pass the input through the module
    output = mha(q=x, k=x, v=x, mask=None)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    # Verification check: The output shape must be the same as the input shape
    assert x.shape == output.shape, "Output shape does not match input shape!"
    
    print("Multi-Head Attention test passed!")
    print("\n--- All Standalone Tests Passed! ---")