import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that they can be summed.
    
    The formulas are from the "Attention Is All You Need" paper.
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimensionality of the embeddings and positional encodings.
            dropout (float): The dropout rate. Defaults to 0.1.
            max_len (int): The maximum possible length of a sequence. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        
        # Calculate the division term for the sine and cosine functions
        # The term is 10000^(2i / d_model)
        # We can write this as exp( (2i / d_model) * log(10000) )
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it can be easily added to the input tensor
        # The shape becomes (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register 'pe' as a buffer. Buffers are part of the model's state,
        # but they are not considered parameters to be trained.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor (batch of sequences).
                              Shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: The input tensor with positional encodings added.
                          Shape: (batch_size, seq_len, d_model)
        """
        # x.size(1) is the sequence length of the current batch.
        # We add the positional encodings up to that length.
        # The shape of self.pe is (1, max_len, d_model), so we slice it.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Standalone Test for the PositionalEncoding class
if __name__ == '__main__':
    print("--- Running Standalone Test for Positional Encoding ---")

    # Define hyperparameters for the test
    d_model = 512
    max_len = 100
    batch_size = 4
    seq_len = 30 # Sequence length of the input
    
    # --- Test Case 1: Check shapes and modification ---
    print("\n[Test Case 1: Shape and Modification Check]")
    
    # Create the PositionalEncoding module
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
    
    # !!! IMPORTANT: Switch module to evaluation mode to disable dropout !!!
    pos_encoder.eval()
    
    # Create a dummy input tensor of zeros
    dummy_input = torch.zeros(batch_size, seq_len, d_model)
    
    # Pass the input through the module
    output = pos_encoder(dummy_input)
    
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    
    # Verification checks
    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"
    assert not torch.equal(output, dummy_input), "Output is identical to input; encodings were not added!"
    
    # This assertion will now pass because dropout is disabled
    assert torch.allclose(output[0, :, :], output[1, :, :]), "Encodings are not consistent across the batch!"
    
    print("Test Case 1 Passed!")

    # --- Test Case 2: Visualize a slice of the encoding ---
    print("\n[Test Case 2: Value Check]")
    
    first_token_encoding = output[0, 0, :]
    print("Encoding for the first token (pos=0):", first_token_encoding[:10])
    
    # Because dropout is off, the values are now deterministic and predictable
    assert torch.allclose(first_token_encoding[0::2], torch.zeros(d_model // 2)), "Sine values for pos=0 should be 0"
    assert torch.allclose(first_token_encoding[1::2], torch.ones(d_model // 2)), "Cosine values for pos=0 should be 1"
    
    print("Test Case 2 Passed!")
    print("\n--- All Standalone Tests Passed! ---")