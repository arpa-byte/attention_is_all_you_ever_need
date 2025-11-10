import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN) from "Attention Is All You Need".

    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This is a two-layer fully connected network that is applied to each position
    separately and identically.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the inner-layer.
            dropout (float): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor. Shape: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Standalone Test for the PositionWiseFeedForward class
if __name__ == '__main__':
    print("--- Running Standalone Test for Position-wise Feed-Forward Network ---")

    # Define hyperparameters for the test
    batch_size = 4
    seq_len = 10
    d_model = 512
    d_ff = 2048 # As specified in the paper (4 * d_model)

    # Create the FFN module
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    # Pass the input through the module
    output = ffn(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # Verification check: The output shape must be the same as the input shape
    assert dummy_input.shape == output.shape, "Output shape does not match input shape!"

    print("\n--- Standalone Test Passed! ---")