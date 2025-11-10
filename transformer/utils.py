# transformer/utils.py

import torch

def create_masks(src: torch.Tensor, trg: torch.Tensor, pad_idx: int):
    """
    Creates the source and target masks needed for the Transformer model.

    Args:
        src (torch.Tensor): The source token ID tensor. Shape: (batch_size, src_len)
        trg (torch.Tensor): The target token ID tensor. Shape: (batch_size, trg_len)
        pad_idx (int): The index of the padding token.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - src_mask (torch.Tensor): The source padding mask. Shape: (batch_size, 1, 1, src_len)
            - trg_mask (torch.Tensor): The target look-ahead and padding mask. Shape: (batch_size, 1, trg_len, trg_len)
    """
    # Source padding mask: Hides padding tokens in the source sequence.
    # Shape: (batch_size, 1, 1, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target padding mask: Hides padding tokens in the target sequence.
    # Shape: (batch_size, 1, 1, trg_len)
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Target look-ahead mask (subsequent mask): Prevents attention to future tokens.
    # It's a lower triangular matrix.
    trg_len = trg.shape[1]
    trg_look_ahead_mask = torch.tril(torch.ones((trg_len, trg_len), device=src.device)).bool()
    
    # The final target mask is a combination of the padding mask and the look-ahead mask.
    # Shape: (batch_size, 1, trg_len, trg_len)
    trg_mask = trg_pad_mask & trg_look_ahead_mask
    
    return src_mask, trg_mask

# Standalone Test
if __name__ == '__main__':
    print("--- Running Standalone Test for Mask Creation ---")
    pad_idx = 0
    src = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    trg = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    
    src_mask, trg_mask = create_masks(src, trg, pad_idx)
    
    print("Source shape:", src.shape)
    print("Source mask shape:", src_mask.shape)
    print("Source mask:\n", src_mask)
    
    print("\nTarget shape:", trg.shape)
    print("Target mask shape:", trg_mask.shape)
    print("Target mask (first item in batch):\n", trg_mask[0])
    
    assert src_mask.shape == (src.shape[0], 1, 1, src.shape[1])
    assert trg_mask.shape == (trg.shape[0], 1, trg.shape[1], trg.shape[1])
    print("\n--- Standalone Test Passed! ---")