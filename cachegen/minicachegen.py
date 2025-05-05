import torch
import torch.nn.functional as F
from typing import List, Dict

class MiniCacheGen:
    def __init__(self, batch_size, num_heads, head_dim, compress_every=128, device='cuda', dtype=torch.float16):
        self.B = batch_size
        self.H = num_heads
        self.D = head_dim
        self.chunk_size = compress_every
        self.device = device
        self.dtype = dtype
        self.compressed_chunks:List[Dict]=[]
        self.uncompressed_kv = []
        self.cur_chunk=[]
        