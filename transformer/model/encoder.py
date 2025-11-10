import torch
import torch.nn as nn
from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Encoder.

    An encoder layer consists of two main sub-layers:
    1. A multi-head self-attention mechanism.
    2. A position-wise fully connected feed-forward network.

    Residual connections and layer normalization are applied around each of the two sub-layers.
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimensionality of the input and output features.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the FFN.
            dropout (float): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # Layer normalization for the two sub-layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderLayer.

        Args:
            src (torch.Tensor): The input tensor from the previous layer.
                                Shape: (batch_size, seq_len, d_model)
            src_mask (torch.Tensor): The mask for the input sequence.

        Returns:
            torch.Tensor: The output tensor of the layer.
                          Shape: (batch_size, seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention sub-layer
        # The query, key, and value are all the same: the source tensor
        attn_output = self.self_attention(q=src, k=src, v=src, mask=src_mask)
        # Apply dropout and the residual connection, followed by layer normalization
        src = self.norm1(src + self.dropout1(attn_output))

        # 2. Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(src)
        # Apply dropout and the residual connection, followed by layer normalization
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src

class Encoder(nn.Module):
    """
    Implements the full Transformer Encoder, which is a stack of N identical EncoderLayers.
    """
    def __init__(self, num_layers: int, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            num_layers (int): The number of EncoderLayers to stack (N).
            d_model (int): The dimensionality of the features.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the FFN.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the entire Encoder stack.

        Args:
            src (torch.Tensor): The source tensor. Shape: (batch_size, seq_len, d_model)
            src_mask (torch.Tensor): The mask for the source sequence.

        Returns:
            torch.Tensor: The output of the final encoder layer.
                          Shape: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# Standalone Test for the Encoder components
if __name__ == '__main__':
    print("--- Running Standalone Test for Encoder Components ---")

    # Define hyperparameters for the test
    batch_size = 4
    seq_len = 10
    d_model = 512
    h = 8
    d_ff = 2048
    num_layers = 6 # N=6 as in the paper

    # Create a dummy input tensor and a mask
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    # A simple padding mask (e.g., last 2 tokens are padding)
    dummy_mask = torch.ones(batch_size, 1, 1, seq_len)
    dummy_mask[:, :, :, -2:] = 0

    # --- Test Case 1: Single EncoderLayer ---
    print("\n[Test Case 1: Single EncoderLayer]")
    encoder_layer = EncoderLayer(d_model=d_model, h=h, d_ff=d_ff)
    output_layer = encoder_layer(dummy_input, dummy_mask)
    
    print("Input shape:", dummy_input.shape)
    print("Output shape (EncoderLayer):", output_layer.shape)
    assert output_layer.shape == dummy_input.shape, "Shape mismatch in EncoderLayer!"
    print("EncoderLayer test passed!")

    # --- Test Case 2: Full Encoder Stack ---
    print("\n[Test Case 2: Full Encoder Stack]")
    encoder = Encoder(num_layers=num_layers, d_model=d_model, h=h, d_ff=d_ff)
    output_encoder = encoder(dummy_input, dummy_mask)
    
    print("Input shape:", dummy_input.shape)
    print("Output shape (Encoder):", output_encoder.shape)
    assert output_encoder.shape == dummy_input.shape, "Shape mismatch in Encoder!"
    print("Encoder stack test passed!")
    
    print("\n--- All Standalone Tests Passed! ---")