import torch, math
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Builds cos/sin tables for RoPE with shape (max_seq_len, head_dim).
    Works with even head_dim; uses the standard base=10_000.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
        # Register in fp32 to avoid precision loss; move/cast on use
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # lazy-built buffers
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

    def _build_cache(self, device, dtype):
        T = self.max_seq_len
        half = self.head_dim // 2
        pos = torch.arange(T, device=device, dtype=self.inv_freq.dtype)  # (T,)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)              # (T, half)
        # Interleave to full dim: [c0,c0,c1,c1,...] / [s0,s0,s1,s1,...]
        cos = freqs.cos().repeat_interleave(2, dim=1)  # (T, head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=1)  # (T, head_dim)
        self._cos = cos.to(dtype=dtype, device=device)
        self._sin = sin.to(dtype=dtype, device=device)

    def get_cos_sin(self, seq_len: int, device, dtype):
        if (self._cos is None) or (self._cos.device != device) or (self._cos.dtype != dtype):
            self._build_cache(device=device, dtype=torch.float32)  # keep master in fp32
            # cast lightweight views on demand
            self._cos = self._cos.to(device=device, dtype=dtype)
            self._sin = self._sin.to(device=device, dtype=dtype)
        return self._cos[:seq_len], self._sin[:seq_len]  # (T, head_dim)
    
    
def apply_rope(q, k, cos, sin, pos_ids=None):
    """
    q,k: (B, H, T, Dh) ; cos,sin: (T, Dh)
    pos_ids: optional (B,T) for packed sequences; otherwise arange(T).
    """
    B,H,T,Dh = q.shape
    if pos_ids is None:
        cos_t = cos[:T]              # (T, Dh)
        sin_t = sin[:T]
        cos_t = cos_t.view(1,1,T,Dh)
        sin_t = sin_t.view(1,1,T,Dh)
    else:
        # gather positions per (B,T)
        cos_t = cos.index_select(0, pos_ids.reshape(-1)).view(B, T, Dh).unsqueeze(1)  # (B,1,T,Dh)
        sin_t = sin.index_select(0, pos_ids.reshape(-1)).view(B, T, Dh).unsqueeze(1)  # (B,1,T,Dh)

    # rotate pairs: even dims are "real", odd dims are "imag"
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]

    cos_e, sin_e = cos_t[..., ::2], sin_t[..., ::2]
    cos_o, sin_o = cos_t[..., 1::2], sin_t[..., 1::2]

    q_rot_even = q_even * cos_e - q_odd * sin_e
    q_rot_odd  = q_even * sin_o + q_odd * cos_o
    k_rot_even = k_even * cos_e - k_odd * sin_e
    k_rot_odd  = k_even * sin_o + k_odd * cos_o

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_out[..., ::2], q_out[..., 1::2] = q_rot_even, q_rot_odd
    k_out[..., ::2], k_out[..., 1::2] = k_rot_even, k_rot_odd
    
    return q_out, k_out