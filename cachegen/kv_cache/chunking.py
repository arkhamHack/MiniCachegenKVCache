import os
import yaml
import torch
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    max_chunk_size: int
    min_chunk_size: int
    chunk_overlap: int
    max_chunks_in_memory: int

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ChunkConfig':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            max_chunk_size=config['chunking']['max_chunk_size'],
            min_chunk_size=config['chunking']['min_chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            max_chunks_in_memory=config['chunking']['max_chunks_in_memory']
        )

class KVChunkManager:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self._chunks_in_memory: Dict[str, torch.Tensor] = {}
    
    def generate_chunk_id(self, content: str, start_idx: int, end_idx: int) -> str:
        """Generate a unique ID for a chunk based on content and position."""
        chunk_info = f"{content}_{start_idx}_{end_idx}"
        return hashlib.sha256(chunk_info.encode()).hexdigest()[:16]
    
    def split_kv_cache(self, kv_cache: Tuple[torch.Tensor, ...], content: str) -> List[Dict]:
        """Split KV cache into chunks with overlap.
        
        Args:
            kv_cache: Tuple of key-value tensors from the model
            content: Original content string for chunk ID generation
            
        Returns:
            List of dictionaries containing chunk information and tensors
        """
        num_layers = len(kv_cache)
        seq_len = kv_cache[0][0].size(2)  # Assuming shape: (batch, num_heads, seq_len, head_dim)
        
        chunks = []
        start_idx = 0
        
        while start_idx < seq_len:
            end_idx = min(start_idx + self.config.max_chunk_size, seq_len)
            
            # Skip small chunks at the end
            if end_idx - start_idx < self.config.min_chunk_size:
                break
            
            chunk_id = self.generate_chunk_id(content, start_idx, end_idx)
            chunk_tensors = []
            
            # Extract chunk from each layer's KV cache
            for layer_idx in range(num_layers):
                k, v = kv_cache[layer_idx]
                k_chunk = k[:, :, start_idx:end_idx, :].clone()
                v_chunk = v[:, :, start_idx:end_idx, :].clone()
                chunk_tensors.append((k_chunk, v_chunk))
            
            chunks.append({
                'chunk_id': chunk_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'tensors': tuple(chunk_tensors)
            })
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.config.chunk_overlap
        
        return chunks
    
    def merge_chunks(self, chunks: List[Dict]) -> Tuple[torch.Tensor, ...]:
        """Merge chunks back into a complete KV cache."""
        if not chunks:
            raise ValueError("No chunks provided for merging")
        
        # Sort chunks by start_idx
        chunks = sorted(chunks, key=lambda x: x['start_idx'])
        
        # Initialize with first chunk's tensors
        merged = []
        num_layers = len(chunks[0]['tensors'])
        
        for layer_idx in range(num_layers):
            layer_chunks_k = []
            layer_chunks_v = []
            
            for chunk in chunks:
                k, v = chunk['tensors'][layer_idx]
                layer_chunks_k.append(k)
                layer_chunks_v.append(v)
            
            # Concatenate along sequence length dimension
            merged_k = torch.cat(layer_chunks_k, dim=2)
            merged_v = torch.cat(layer_chunks_v, dim=2)
            merged.append((merged_k, merged_v))
        
        return tuple(merged)
    
    def cache_chunk(self, chunk_id: str, chunk_tensors: Tuple[torch.Tensor, ...]):
        """Cache chunk in memory, evicting old chunks if necessary."""
        if len(self._chunks_in_memory) >= self.config.max_chunks_in_memory:
            # Simple LRU: remove oldest chunk
            oldest_chunk_id = next(iter(self._chunks_in_memory))
            del self._chunks_in_memory[oldest_chunk_id]
        
        self._chunks_in_memory[chunk_id] = chunk_tensors
    
    def get_cached_chunk(self, chunk_id: str) -> Optional[Tuple[torch.Tensor, ...]]:
        """Retrieve a chunk from memory cache."""
        return self._chunks_in_memory.get(chunk_id)
