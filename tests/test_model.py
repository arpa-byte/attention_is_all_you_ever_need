import torch
import pytest

from transformer.model.encoder import EncoderLayer, Encoder
from transformer.model.decoder import DecoderLayer, Decoder
from transformer.model.transformer import Transformer

def test_encoder_layer():
    """Tests the EncoderLayer for correct shape propagation."""
    batch_size = 16
    seq_len = 25
    d_model = 512
    h = 8
    d_ff = 2048
    
    encoder_layer = EncoderLayer(d_model=d_model, h=h, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    
    output = encoder_layer(x, mask)
    assert output.shape == x.shape

def test_encoder():
    """Tests the full Encoder stack for correct shape propagation."""
    batch_size = 16
    seq_len = 25
    d_model = 512
    h = 8
    d_ff = 2048
    num_layers = 6
    
    encoder = Encoder(num_layers=num_layers, d_model=d_model, h=h, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    
    output = encoder(x, mask)
    assert output.shape == x.shape

def test_decoder_layer():
    """Tests the DecoderLayer for correct shape propagation."""
    batch_size = 16
    trg_seq_len = 20
    src_seq_len = 25
    d_model = 512
    h = 8
    d_ff = 2048
    
    decoder_layer = DecoderLayer(d_model=d_model, h=h, d_ff=d_ff)
    trg = torch.randn(batch_size, trg_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    trg_mask = torch.tril(torch.ones(1, 1, trg_seq_len, trg_seq_len))
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)

    output = decoder_layer(trg, encoder_output, trg_mask, src_mask)
    assert output.shape == trg.shape

def test_decoder():
    """Tests the full Decoder stack for correct shape propagation."""
    batch_size = 16
    trg_seq_len = 20
    src_seq_len = 25
    d_model = 512
    h = 8
    d_ff = 2048
    num_layers = 6
    
    decoder = Decoder(num_layers=num_layers, d_model=d_model, h=h, d_ff=d_ff)
    trg = torch.randn(batch_size, trg_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    trg_mask = torch.tril(torch.ones(1, 1, trg_seq_len, trg_seq_len))
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    
    output = decoder(trg, encoder_output, trg_mask, src_mask)
    assert output.shape == trg.shape

# ### NEW FINAL INTEGRATION TEST ###

def test_transformer_forward_pass():
    """
    Tests the full Transformer model for a complete forward pass.
    This is the ultimate integration test.
    """
    # Define model hyperparameters
    src_vocab_size = 100
    trg_vocab_size = 120
    d_model = 512
    num_layers = 6
    h = 8
    d_ff = 2048
    dropout = 0.1
    pad_idx = 0
    
    # Define batch parameters
    batch_size = 16
    src_len = 30
    trg_len = 25
    
    # Create the model
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_layers, h, d_ff, dropout, pad_idx)
    
    # Create dummy input tensors
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    trg = torch.randint(1, trg_vocab_size, (batch_size, trg_len))
    
    # Get the output
    output = model(src, trg)
    
    # Check shape
    expected_shape = (batch_size, trg_len, trg_vocab_size)
    assert output.shape == expected_shape, \
        f"Expected final output shape {expected_shape}, but got {output.shape}"