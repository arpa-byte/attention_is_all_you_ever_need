# transformer/model/transformer.py

import torch
import torch.nn as nn
import math
from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder
from transformer.layers.positional_encoding import PositionalEncoding
from transformer.utils import create_masks

class Transformer(nn.Module):
    """
    The main Transformer model architecture, putting together the Encoder, Decoder,
    and all necessary embeddings and layers.
    """
    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 h: int,
                 d_ff: int,
                 dropout: float,
                 pad_idx: int,
                 max_len: int = 5000):
        super().__init__()

        self.pad_idx = pad_idx

        # Embeddings and Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # Encoder and Decoder stacks
        self.encoder = Encoder(num_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, h, d_ff, dropout)
        
        # Final linear layer to project to target vocabulary size
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
        # Weight initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Weight sharing between target embedding and final linear layer
        self.fc_out.weight = self.trg_embedding.weight

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the entire Transformer model.

        Args:
            src (torch.Tensor): Source sequence token IDs. Shape: (batch_size, src_len)
            trg (torch.Tensor): Target sequence token IDs. Shape: (batch_size, trg_len)

        Returns:
            torch.Tensor: The output logits. Shape: (batch_size, trg_len, trg_vocab_size)
        """
        # 1. Create masks
        src_mask, trg_mask = create_masks(src, trg, self.pad_idx)

        # 2. Apply embeddings and positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim))
        trg_emb = self.pos_encoder(self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim))
        
        # 3. Pass through the Encoder
        encoder_outputs = self.encoder(src_emb, src_mask)
        
        # 4. Pass through the Decoder
        decoder_outputs = self.decoder(trg_emb, encoder_outputs, trg_mask, src_mask)
        
        # 5. Pass through the final linear layer
        output = self.fc_out(decoder_outputs)
        
        return output

# Standalone Test for the full Transformer model
if __name__ == '__main__':
    print("--- Running Standalone Test for Full Transformer Model ---")

    # Define hyperparameters
    src_vocab_size = 1000
    trg_vocab_size = 1200
    d_model = 512
    num_layers = 6
    h = 8
    d_ff = 2048
    dropout = 0.1
    pad_idx = 0
    
    batch_size = 4
    src_len = 15
    trg_len = 12

    # Create the model
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_layers, h, d_ff, dropout, pad_idx)

    # Create dummy input tensors (must be LongTensor for embeddings)
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    trg = torch.randint(1, trg_vocab_size, (batch_size, trg_len))

    # Test forward pass
    output = model(src, trg)

    print("Source input shape:", src.shape)
    print("Target input shape:", trg.shape)
    print("Final output shape:", output.shape)
    
    # Verification check
    expected_shape = (batch_size, trg_len, trg_vocab_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    
    print("\n--- Standalone Test Passed! ---")