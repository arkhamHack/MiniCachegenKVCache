import torch

class KVCache:
    def __init__(self, batch_size, num_heads, head_dim, max_seq_len, device='cuda', dtype=torch.float16):
        self.B = batch_size
        self.H = num_heads
        self.D = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros(self.B, self.H, self.max_seq_len, self.D, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros(self.B, self.H, self.max_seq_len, self.D, device=self.device, dtype=self.dtype)
        self.cur_pos = 0
        
    def append(self,k:torch.Tensor,T:torch.Tensor):
        """
        Args:
            k, v: [B, H, T, D] — T usually = 1 in streaming
        """
        T = k.shape[2]
        assert self.cur_pos + T = 

    def get(self):
        return self.k_cache[:,:,:self.cur_pos,:], self.v_cache[:, :, :self.cur_pos, :]

