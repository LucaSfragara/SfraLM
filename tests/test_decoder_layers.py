import torch
import pytest

from models.layers.decoder_layers import SelfAttentionDecoderLayer
from models.layers.sublayers import SelfAttentionLayer, FeedForwardLayer

@pytest.fixture
def decoder_config():
    """Provides a default configuration for the decoder layer."""
    return {
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 128,
        "dropout": 0.1
    }

@pytest.fixture
def decoder_layer(decoder_config):
    """Provides a SelfAttentionDecoderLayer instance."""
    return SelfAttentionDecoderLayer(**decoder_config)

def test_decoder_layer_initialization(decoder_layer, decoder_config):
    """
    Tests if the SelfAttentionDecoderLayer initializes its sublayers correctly.
    """
    assert isinstance(decoder_layer, torch.nn.Module), "Layer should be a PyTorch Module"
    
    # Check if sublayers are of the correct type
    assert isinstance(decoder_layer.self_attn, SelfAttentionLayer), "self_attn should be a SelfAttentionLayer"
    assert isinstance(decoder_layer.ffn, FeedForwardLayer), "ffn should be a FeedForwardLayer"

    # Check if parameters were passed correctly to sublayers
    assert decoder_layer.self_attn.d_model == decoder_config["d_model"]
    assert decoder_layer.self_attn.num_heads == decoder_config["num_heads"]
    assert decoder_layer.ffn.ffn[0].in_features == decoder_config["d_model"] # Check d_model in FFN
    assert decoder_layer.ffn.ffn[0].out_features == decoder_config["d_ff"] # Check d_ff in FFN

def test_decoder_layer_forward_pass_shapes(decoder_layer, decoder_config):
    """
    Tests the output shapes of the forward pass.
    """
    batch_size = 4
    seq_len = 10
    d_model = decoder_config["d_model"]
    num_heads = decoder_config["num_heads"]

    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attn_weights = decoder_layer(x)

    # Check output tensor shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, but got {output.shape}"

    # Check attention weights shape
    #assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
    #    f"Expected attention weights shape {(batch_size, num_heads, seq_len, seq_len)}, but got {attn_weights.shape}"

def test_decoder_layer_train_eval_mode(decoder_config):
    """
    Tests that dropout is applied in training mode but not in evaluation mode.
    """
    # Ensure dropout is active for this test
    assert decoder_config["dropout"] > 0, "Dropout must be > 0 for this test"
    
    layer = SelfAttentionDecoderLayer(**decoder_config)
    
    batch_size = 2
    seq_len = 5
    d_model = decoder_config["d_model"]
    
    # Use a fixed input to ensure consistency
    x = torch.ones(batch_size, seq_len, d_model)

    # --- Evaluation Mode ---
    # In eval mode, dropout is disabled, so outputs should be identical
    layer.eval()
    with torch.no_grad():
        output1_eval, _ = layer(x)
        output2_eval, _ = layer(x)
    
    assert torch.equal(output1_eval, output2_eval), "Outputs should be identical in evaluation mode"

    # --- Training Mode ---
    # In train mode, dropout is active, so outputs should differ
    layer.train()
    output1_train, _ = layer(x)
    output2_train, _ = layer(x)

    assert not torch.equal(output1_train, output2_train), "Outputs should differ in training mode due to dropout"
    
    # Also check that train output is different from eval output
    assert not torch.equal(output1_train, output1_eval), "Training and evaluation outputs should not be the same"
