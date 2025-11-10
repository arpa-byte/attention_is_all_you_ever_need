import torch
import torch.nn as nn
from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Decoder.

    A decoder layer consists of three main sub-layers:
    1. A masked multi-head self-attention mechanism (for the target sequence).
    2. A multi-head encoder-decoder attention mechanism (cross-attention).
    3. A position-wise fully connected feed-forward network.
    
    Residual connections and layer normalization are applied around each of the three sub-layers.
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimensionality of the features.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the FFN.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.encoder_decoder_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, trg: torch.Tensor, encoder_outputs: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DecoderLayer.

        Args:
            trg (torch.Tensor): The target sequence tensor. Shape: (batch_size, trg_seq_len, d_model)
            encoder_outputs (torch.Tensor): The output from the encoder stack. Shape: (batch_size, src_seq_len, d_model)
            trg_mask (torch.Tensor): The mask for the target sequence (look-ahead mask).
            src_mask (torch.Tensor): The mask for the source sequence (padding mask).

        Returns:
            torch.Tensor: The output tensor of the layer. Shape: (batch_size, trg_seq_len, d_model)
        """
        # 1. Masked Multi-Head Self-Attention on the target sequence
        attn_output = self.masked_self_attention(q=trg, k=trg, v=trg, mask=trg_mask)
        trg = self.norm1(trg + self.dropout1(attn_output))

        # 2. Encoder-Decoder Attention (Cross-Attention)
        # Query comes from the decoder's previous sub-layer (trg).
        # Key and Value come from the encoder's output.
        attn_output = self.encoder_decoder_attention(q=trg, k=encoder_outputs, v=encoder_outputs, mask=src_mask)
        trg = self.norm2(trg + self.dropout2(attn_output))

        # 3. Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout3(ff_output))

        return trg

class Decoder(nn.Module):
    """
    Implements the full Transformer Decoder, which is a stack of N identical DecoderLayers.
    """
    def __init__(self, num_layers: int, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, trg: torch.Tensor, encoder_outputs: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            trg = layer(trg, encoder_outputs, trg_mask, src_mask)
        return trg

# Standalone Test for the Decoder components
if __name__ == '__main__':
    print("--- Running Standalone Test for Decoder Components ---")

    # Define hyperparameters for the test
    batch_size = 4
    trg_seq_len = 10
    src_seq_len = 12
    d_model = 512
    h = 8
    d_ff = 2048
    num_layers = 6

    # Create dummy input tensors
    dummy_trg = torch.randn(batch_size, trg_seq_len, d_model)
    dummy_encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    # Create dummy masks
    dummy_trg_mask = torch.tril(torch.ones(trg_seq_len, trg_seq_len)).unsqueeze(0).unsqueeze(0)
    dummy_src_mask = torch.ones(batch_size, 1, 1, src_seq_len)

    # --- Test Case 1: Single DecoderLayer ---
    print("\n[Test Case 1: Single DecoderLayer]")
    decoder_layer = DecoderLayer(d_model=d_model, h=h, d_ff=d_ff)
    output_layer = decoder_layer(dummy_trg, dummy_encoder_output, dummy_trg_mask, dummy_src_mask)
    
    print("Target input shape:", dummy_trg.shape)
    print("Encoder output shape:", dummy_encoder_output.shape)
    print("Output shape (DecoderLayer):", output_layer.shape)
    assert output_layer.shape == dummy_trg.shape, "Shape mismatch in DecoderLayer!"
    print("DecoderLayer test passed!")

    # --- Test Case 2: Full Decoder Stack ---
    print("\n[Test Case 2: Full Decoder Stack]")
    decoder = Decoder(num_layers=num_layers, d_model=d_model, h=h, d_ff=d_ff)
    output_decoder = decoder(dummy_trg, dummy_encoder_output, dummy_trg_mask, dummy_src_mask)

    print("Target input shape:", dummy_trg.shape)
    print("Output shape (Decoder):", output_decoder.shape)
    assert output_decoder.shape == dummy_trg.shape, "Shape mismatch in Decoder!"
    print("Decoder stack test passed!")
    
    print("\n--- All Standalone Tests Passed! ---")