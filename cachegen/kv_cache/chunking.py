import os
import yaml
import torch
import hashlib
from typing import Dict, List, Tuple, Optional, BinaryIO
from dataclasses import dataclass
from .encoder import KVCacheEncoder
from .exceptions import ChunkError, EncodingError

@dataclass
class ChunkConfig:
    max_chunk_size: int  # Maximum tokens per chunk
    min_chunk_size: int  # Minimum tokens per chunk
    chunk_overlap: int   # Number of overlapping tokens
    max_chunks_in_memory: int  # Maximum chunks to keep in memory cache
    num_bits: int = 8    # Number of bits for quantization

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ChunkConfig':
        try:
            if not os.path.exists(config_path):
                raise ChunkError(f"Config file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ChunkError(f"Invalid YAML format in config file: {str(e)}")
                    
            chunking_config = config.get('chunking', {})
            if not isinstance(chunking_config, dict):
                raise ChunkError("Missing or invalid 'chunking' section in config")
                
            return cls(
                max_chunk_size=chunking_config.get('max_chunk_size', 512),
                min_chunk_size=chunking_config.get('min_chunk_size', 64),
                chunk_overlap=chunking_config.get('chunk_overlap', 16),
                max_chunks_in_memory=chunking_config.get('max_chunks_in_memory', 1000),
                num_bits=chunking_config.get('num_bits', 8)
            )
        except Exception as e:
            if isinstance(e, ChunkError):
                raise
            raise ChunkError(f"Failed to load config: {str(e)}")

class KVChunkManager:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.encoder = KVCacheEncoder(num_bits=config.num_bits)
        self._chunks_in_memory: Dict[str, bytes] = {}
        self._chunk_metadata: Dict[str, List[Dict]] = {}
    
    def generate_chunk_id(self, content: str, start_idx: int, end_idx: int) -> str:
        """Generate a unique ID for a chunk based on content and position."""
        try:
            if not isinstance(content, str):
                raise ChunkError("Content must be a string")
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise ChunkError("Indices must be integers")
            if start_idx < 0 or end_idx < start_idx:
                raise ChunkError("Invalid index values")
                
            chunk_info = f"{content}_{start_idx}_{end_idx}"
            return hashlib.sha256(chunk_info.encode()).hexdigest()[:16]
        except Exception as e:
            if isinstance(e, ChunkError):
                raise
            raise ChunkError(f"Failed to generate chunk ID: {str(e)}")
    
    def split_kv_cache(self, kv_cache: Tuple[torch.Tensor, ...], content: str) -> List[Dict]:
        """Split and encode KV cache into compressed chunks with overlap.
        
        Args:
            kv_cache: Tuple of key-value tensors from the model
            content: Original content string for chunk ID generation
            
        Returns:
            List of dictionaries containing chunk information and compressed data
        """
        try:
            if not isinstance(kv_cache, tuple) or not kv_cache:
                raise ChunkError("KV cache must be a non-empty tuple of tensors")
                
            if not all(isinstance(layer, tuple) and len(layer) == 2 for layer in kv_cache):
                raise ChunkError("Each layer must be a tuple of (key, value) tensors")
                
            num_layers = len(kv_cache)
            try:
                seq_len = kv_cache[0][0].size(2)  # Assuming shape: (batch, num_heads, seq_len, head_dim)
            except (IndexError, AttributeError) as e:
                raise ChunkError(f"Invalid KV cache tensor shape: {str(e)}")
            
            chunks = []
            start_idx = 0
            
            while start_idx < seq_len:
                try:
                    end_idx = min(start_idx + self.config.max_chunk_size, seq_len)
                    
                    # Skip small chunks at the end
                    if end_idx - start_idx < self.config.min_chunk_size:
                        break
                    
                    chunk_id = self.generate_chunk_id(content, start_idx, end_idx)
                    chunk_data = []
                    
                    # Extract and encode chunk from each layer's KV cache
                    for layer_idx in range(num_layers):
                        try:
                            k, v = kv_cache[layer_idx]
                            k_chunk = k[:, :, start_idx:end_idx, :]
                            v_chunk = v[:, :, start_idx:end_idx, :]
                            
                            # Encode key and value tensors to compressed bitstreams
                            try:
                                encoded_k = self.encoder.encode_tensor(k_chunk, use_cuda=True)
                                encoded_v = self.encoder.encode_tensor(v_chunk, use_cuda=True)
                                chunk_data.extend([encoded_k, encoded_v])
                            except EncodingError as e:
                                raise ChunkError(f"Failed to encode tensors for layer {layer_idx}: {str(e)}")
                        except Exception as e:
                            raise ChunkError(f"Error processing layer {layer_idx}: {str(e)}")
                    
                    chunks.append({
                        'chunk_id': chunk_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'compressed_data': chunk_data
                    })
                    
                    # Move to next chunk with overlap
                    start_idx = end_idx - self.config.chunk_overlap
                    
                except Exception as e:
                    raise ChunkError(f"Failed to process chunk at indices {start_idx}:{end_idx}: {str(e)}")
            
            if not chunks:
                raise ChunkError("No valid chunks were generated")
                
            return chunks
            
        except Exception as e:
            if isinstance(e, (ChunkError, EncodingError)):
                raise
            raise ChunkError(f"Unexpected error during KV cache splitting: {str(e)}")
    
    def merge_chunks(self, chunks: List[Dict]) -> Tuple[torch.Tensor, ...]:
        """Merge compressed chunks back into a complete KV cache."""
        try:
            if not isinstance(chunks, list):
                raise ChunkError("Input must be a list of chunks")
            if not chunks:
                raise ChunkError("No chunks provided for merging")
                
            # Validate chunk format
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    raise ChunkError(f"Invalid chunk format at index {i}")
                if not all(key in chunk for key in ['chunk_id', 'start_idx', 'end_idx', 'compressed_data']):
                    raise ChunkError(f"Missing required fields in chunk at index {i}")
            
            # Sort chunks by start_idx and validate sequence
            try:
                chunks = sorted(chunks, key=lambda x: x['start_idx'])
                
                for i in range(len(chunks) - 1):
                    if chunks[i]['end_idx'] < chunks[i]['start_idx']:
                        raise ChunkError(f"Invalid chunk boundaries at index {i}")
                    if chunks[i]['end_idx'] < chunks[i+1]['start_idx']:
                        raise ChunkError(f"Gap between chunks at indices {i} and {i+1}")
                        
                # Initialize lists for each layer's tensors
                num_layers = len(chunks[0]['compressed_data']) // 2  # Each layer has K and V
                if num_layers == 0:
                    raise ChunkError("No layer data found in chunks")
                    
                merged = []
                
                for layer_idx in range(num_layers):
                    try:
                        # Get all key tensors for this layer
                        layer_k = []
                        layer_v = []
                        
                        for i, chunk in enumerate(chunks):
                            try:
                                if len(chunk['compressed_data']) <= layer_idx * 2 + 1:
                                    raise ChunkError(f"Missing layer data in chunk {i}")
                                    
                                k_data, v_data = chunk['compressed_data'][layer_idx * 2:layer_idx * 2 + 2]
                                
                                try:
                                    k_tensor = self.encoder.decode_tensor(k_data)
                                    v_tensor = self.encoder.decode_tensor(v_data)
                                except Exception as e:
                                    raise ChunkError(f"Failed to decode tensors in chunk {i}: {str(e)}")
                                    
                                layer_k.append(k_tensor)
                                layer_v.append(v_tensor)
                                
                            except Exception as e:
                                raise ChunkError(f"Error processing chunk {i} for layer {layer_idx}: {str(e)}")
                        
                        try:
                            # Concatenate tensors along sequence dimension
                            if not layer_k or not layer_v:
                                raise ChunkError(f"No valid tensors found for layer {layer_idx}")
                                
                            k = torch.cat(layer_k, dim=2)
                            v = torch.cat(layer_v, dim=2)
                            merged.append((k, v))
                            
                        except Exception as e:
                            raise ChunkError(f"Failed to concatenate tensors for layer {layer_idx}: {str(e)}")
                            
                    except Exception as e:
                        raise ChunkError(f"Failed to process layer {layer_idx}: {str(e)}")
                
                if not merged:
                    raise ChunkError("No layers were successfully merged")
                    
                return tuple(merged)
                
            except Exception as e:
                raise ChunkError(f"Failed to merge chunks: {str(e)}")
                
        except Exception as e:
            if isinstance(e, ChunkError):
                raise
            raise ChunkError(f"Unexpected error during chunk merging: {str(e)}")
    
    def cache_chunk(self, chunk_id: str, compressed_data: List[Tuple[bytes, Dict]]):
        """Cache compressed chunk in memory, evicting old chunks if necessary."""
        try:
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ChunkError("Invalid chunk ID")
            if not isinstance(compressed_data, list) or not compressed_data:
                raise ChunkError("Invalid compressed data format")
                
            # Validate compressed data format
            for i, (data, meta) in enumerate(compressed_data):
                if not isinstance(data, bytes):
                    raise ChunkError(f"Invalid data type at index {i}")
                if not isinstance(meta, dict):
                    raise ChunkError(f"Invalid metadata type at index {i}")
                if 'compressed_size' not in meta:
                    raise ChunkError(f"Missing compressed_size in metadata at index {i}")
            
            try:
                if len(self._chunks_in_memory) >= self.config.max_chunks_in_memory:
                    # Simple LRU: remove oldest chunk
                    try:
                        oldest_chunk_id = next(iter(self._chunks_in_memory))
                        del self._chunks_in_memory[oldest_chunk_id]
                        del self._chunk_metadata[oldest_chunk_id]
                    except Exception as e:
                        raise ChunkError(f"Failed to evict old chunk: {str(e)}")
                
                # Store compressed data and metadata separately
                try:
                    self._chunks_in_memory[chunk_id] = b''.join(data for data, _ in compressed_data)
                    self._chunk_metadata[chunk_id] = [meta for _, meta in compressed_data]
                except Exception as e:
                    raise ChunkError(f"Failed to store chunk data: {str(e)}")
                    
            except Exception as e:
                raise ChunkError(f"Failed to cache chunk {chunk_id}: {str(e)}")
                
        except Exception as e:
            if isinstance(e, ChunkError):
                raise
            raise ChunkError(f"Unexpected error during chunk caching: {str(e)}")
    
    def get_cached_chunk(self, chunk_id: str) -> Optional[List[Tuple[bytes, Dict]]]:
        """Retrieve a compressed chunk from memory cache."""
        try:
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ChunkError("Invalid chunk ID")
                
            if chunk_id not in self._chunks_in_memory:
                return None
                
            try:
                compressed_data = self._chunks_in_memory[chunk_id]
                metadata = self._chunk_metadata[chunk_id]
                
                if not isinstance(compressed_data, bytes):
                    raise ChunkError("Invalid compressed data format in cache")
                if not isinstance(metadata, list):
                    raise ChunkError("Invalid metadata format in cache")
                    
                # Split the concatenated bitstream back into individual tensors
                result = []
                offset = 0
                
                for i, meta in enumerate(metadata):
                    try:
                        if not isinstance(meta, dict) or 'compressed_size' not in meta:
                            raise ChunkError(f"Invalid metadata at index {i}")
                            
                        size = meta['compressed_size']
                        if not isinstance(size, int) or size < 0:
                            raise ChunkError(f"Invalid compressed_size at index {i}")
                            
                        if offset + size > len(compressed_data):
                            raise ChunkError(f"Compressed data size mismatch at index {i}")
                            
                        chunk = compressed_data[offset:offset + size]
                        result.append((chunk, meta))
                        offset += size
                        
                    except Exception as e:
                        raise ChunkError(f"Failed to process chunk at index {i}: {str(e)}")
                        
                if offset != len(compressed_data):
                    raise ChunkError("Compressed data size does not match metadata")
                    
                return result
                
            except Exception as e:
                raise ChunkError(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
                
        except Exception as e:
            if isinstance(e, ChunkError):
                raise
            raise ChunkError(f"Unexpected error during chunk retrieval: {str(e)}")
