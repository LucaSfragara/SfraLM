import pytest
import torch
from models.transformers import DecoderOnlyTransformer

def test_shapes():
    B,T,V,D = 2, 16, 101, 128
    m = DecoderOnlyTransformer(
        num_layers=2, d_model=D, num_classes = V, num_heads=4, d_ff=4*D, dropout=0.1,
        seq_len=T, weight_tying=True
    )
    
    toks = torch.randint(1, V, (B, T))
    toks[0, -3:] = 0  # pad a few
    logits, att = m(toks)
    assert logits.shape == (B,T,V)
    assert set(att.keys()) == {"layer1_dec_self", "layer2_dec_self"}

def test_weight_tying():
    V,D = 257, 64
    m = DecoderOnlyTransformer(1, D, 4, 256, 0.0, 32, V, weight_tying=True)
    assert m.final_linear.weight.data_ptr() == m.target_embedding.embed.weight.data_ptr()

@torch.no_grad()
def test_score_eval():
    V,D,T = 50, 64, 12
    m = DecoderOnlyTransformer(2, D, 4, 256, 0.0, 64, V, weight_tying=True).eval()
    toks = torch.randint(0, V, (1, T))
    nxt = m.score(toks)
    assert nxt.shape == (1, V)