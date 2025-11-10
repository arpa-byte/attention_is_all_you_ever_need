import torch
import pytest

# Import the components we want to test
from transformer.layers.attention import scaled_dot_product_attention, MultiHeadAttention
from transformer.layers.positional_encoding import PositionalEncoding
from transformer.layers.feed_forward import PositionWiseFeedForward


@pytest.fixture
def attention_test_data():
    """Provides a set of reusable dummy tensors for testing the attention mechanism."""
    batch_size = 4
    seq_len = 5
    d_k = 8
    d_v = 10
    
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)
    
    return q, k, v, batch_size, seq_len, d_v

def test_scaled_dot_product_attention_no_mask(attention_test_data):
    """Tests the scaled_dot_product_attention function without any mask."""
    q, k, v, batch_size, seq_len, d_v = attention_test_data
    output, attention = scaled_dot_product_attention(q, k, v)
    assert output.shape == (batch_size, seq_len, d_v)
    assert attention.shape == (batch_size, seq_len, seq_len)
    sum_of_weights = attention.sum(dim=-1)
    assert torch.allclose(sum_of_weights, torch.ones(batch_size, seq_len))

def test_scaled_dot_product_attention_with_mask(attention_test_data):
    """Tests the scaled_dot_product_attention function with a look-ahead mask."""
    q, k, v, batch_size, seq_len, d_v = attention_test_data
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
    output, attention = scaled_dot_product_attention(q, k, v, mask=mask)
    assert output.shape == (batch_size, seq_len, d_v)
    assert attention.shape == (batch_size, seq_len, seq_len)
    upper_triangular_part = attention.triu(diagonal=1)
    assert torch.all(upper_triangular_part == 0)

def test_positional_encoding():
    """Tests the PositionalEncoding module."""
    d_model = 512
    seq_len = 100
    batch_size = 16
    
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=5000)
    pos_encoder.eval() # Disable dropout for testing
    
    input_tensor = torch.zeros(batch_size, seq_len, d_model)
    output = pos_encoder(input_tensor)
    
    assert output.shape == input_tensor.shape
    assert not torch.equal(output, input_tensor)
    assert torch.allclose(output[0,:,:], output[-1,:,:])

def test_multi_head_attention():
    """Tests the MultiHeadAttention module."""
    batch_size = 16
    seq_len = 20
    d_model = 512
    h = 8
    
    mha = MultiHeadAttention(d_model=d_model, h=h)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test without mask
    output = mha(q=x, k=x, v=x, mask=None)
    assert output.shape == x.shape
    
    # Test with mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    output_masked = mha(q=x, k=x, v=x, mask=mask)
    assert output_masked.shape == x.shape

# ### NEW TEST FOR POSITION-WISE FEED-FORWARD NETWORK ###

def test_position_wise_feed_forward():
    """
    Tests the PositionWiseFeedForward module.
    
    It checks for:
    1. Correct output shape.
    """
    batch_size = 16
    seq_len = 20
    d_model = 512
    d_ff = 2048
    
    # Create the module and a dummy input tensor
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Get the output
    output = ffn(x)
    
    # 1. Check shape
    assert output.shape == x.shape, \
        f"Expected output shape {x.shape}, but got {output.shape}"