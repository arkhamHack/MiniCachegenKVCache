import torch
def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to Q or K tensor.
    Args:
        x: Tensor of shape (batch, heads, seq_len, head_dim)
        position_ids: Tensor of shape (batch, seq_len)
        theta: RoPE base (default 10,000)
    Returns:
        Tensor with RoPE applied
    """

    bsz, n_heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")

    half_dim = head_dim//2
    freq.seq = torch.arange(half_dim,dtype=torch.float32,device=x.device)
    inv_freq = 1.0 / (theta ** (freq_seq/half_dim))
    sinusoid_inp = torch.einsum()
    sin = sinusoid_inp.sin().unsqueeze(1)
    cos = sinusoid_inp.cos().unsqueeze(1)

    x1=x[...,:half_dim]
    x2=x[...,half_dim:]

    # Apply rotation: [cos, -sin; sin, cos]

    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)



    return x_rotated

