import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ctypes import c_void_p
from .exceptions import DecodingError, CUDAError, MetadataError
from . import cuda_helper

class KVCacheDecoder:
    def __init__(self, num_bits: int = 8):
        """Initialize KV Cache decoder.
        
        Args:
            num_bits: Base number of bits used in quantization (default: 8)
        """
        self.num_bits = num_bits
        self.cuda_module = None
    
    def _load_cuda_module(self):
        """Load CUDA module for fast decoding if not already loaded."""
        if self.cuda_module is None:
            try:
                self.cuda_module = cuda_helper
            except Exception as e:
                raise CUDAError("Failed to load CUDA decoder module", e)
    
    def dequantize_tensor(self, 
                         quantized: torch.Tensor, 
                         metadata: Dict) -> torch.Tensor:
        """
        Dequantize tensor values back to original range.
        Handles layer-wise quantization with different bit depths.
        """
        try:
            if not all(key in metadata for key in ['min_val', 'max_val']):
                raise MetadataError("Missing required metadata fields for dequantization")
            
            # Handle constant tensors
            if metadata.get('is_constant', False):
                const_val = metadata['min_val']
                return torch.full_like(quantized, const_val, dtype=torch.float32)
            
            min_val = metadata['min_val']
            max_val = metadata['max_val']
            bits = metadata.get('bits', self.num_bits)
            
            if min_val >= max_val:
                raise DecodingError("Invalid min/max values in metadata")
            
            # Dequantize
            max_quant_val = 2**bits - 1
            scale = (max_val - min_val) / max_quant_val
            
            try:
                dequantized = quantized.float() * scale + min_val
                
                # Restore original dtype if specified
                if 'dtype' in metadata:
                    target_dtype = getattr(torch, metadata['dtype'].replace('torch.', ''))
                    dequantized = dequantized.to(dtype=target_dtype)
                    
                return dequantized
            except Exception as e:
                raise DecodingError(f"Failed to dequantize tensor: {str(e)}")
        except Exception as e:
            if isinstance(e, (DecodingError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during dequantization: {str(e)}")
    
    def reconstruct_from_deltas(self, 
                                anchors: torch.Tensor,
                                deltas: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct tensor from anchors and deltas.
        
        Args:
            anchors: [batch, num_heads, 1, head_dim]
            deltas: [batch, num_heads, seq_len-1, head_dim]
            
        Returns:
            Reconstructed tensor
        """
        try:
            if deltas.size(2) == 0:
                return anchors
            
            # Cumulative sum to reconstruct absolute values
            batch, num_heads, _, head_dim = anchors.shape
            seq_len = deltas.size(2) + 1
            
            reconstructed = torch.zeros(batch, num_heads, seq_len, head_dim,
                                       dtype=anchors.dtype,
                                       device=anchors.device)
            
            reconstructed[:, :, 0:1, :] = anchors
            
            # Cumulative sum of deltas
            cumsum_deltas = torch.cumsum(deltas, dim=2)
            reconstructed[:, :, 1:, :] = anchors + cumsum_deltas
            
            return reconstructed
            
        except Exception as e:
            raise DecodingError(f"Failed to reconstruct from deltas: {str(e)}")
    
    def ungroup_channels(self, 
                        grouped_tensor: torch.Tensor,
                        metadata: Dict) -> torch.Tensor:
        """
        Restore original channel structure from grouped representation.
        
        Args:
            grouped_tensor: [batch, num_heads, seq_len, num_groups, group_size]
            metadata: Channel grouping metadata
            
        Returns:
            Ungrouped tensor
        """
        try:
            # Denormalize each group
            group_means = metadata['group_means'].unsqueeze(-1)
            group_stds = metadata['group_stds'].unsqueeze(-1)
            
            denormalized = grouped_tensor * group_stds + group_means
            
            # Reshape back
            batch, num_heads, seq_len, num_groups, group_size = denormalized.shape
            reconstructed = denormalized.reshape(batch, num_heads, seq_len, -1)
            
            # Remove padding if any
            original_head_dim = metadata['original_head_dim']
            reconstructed = reconstructed[..., :original_head_dim]
            
            return reconstructed
            
        except Exception as e:
            raise DecodingError(f"Failed to ungroup channels: {str(e)}")
    
    def decode_tensor(self, 
                     compressed: bytes, 
                     metadata: Dict, 
                     use_cuda: bool = True, 
                     stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """
        Decode compressed bitstream with full CacheGen pipeline:
        1. Arithmetic decoding
        2. Dequantization  
        3. Delta reconstruction (if used)
        4. Channel ungrouping (if used)
        
        Args:
            compressed: Compressed bitstream
            metadata: Encoding metadata
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Decoded tensor
        """
        try:
            if not isinstance(compressed, bytes):
                raise DecodingError("Input must be a bytes object")
            
            use_delta = metadata.get('use_delta_encoding', False)
            use_grouping = metadata.get('use_channel_grouping', False)
            
            if use_delta:
                # Decode anchors and deltas separately
                anchor_metadata = metadata['anchor_metadata']
                delta_metadata = metadata['delta_metadata']
                anchor_size = metadata['anchor_size']
                
                anchor_compressed = compressed[:anchor_size]
                delta_compressed = compressed[anchor_size:]
                
                # Decode anchors
                anchor_shape = (metadata['original_shape'][0],
                              metadata['original_shape'][1],
                              1,
                              metadata['original_shape'][3] if not use_grouping 
                              else metadata['channel_metadata']['num_groups'] * 
                                   metadata['channel_metadata']['group_size'])
                
                anchor_probs = torch.tensor(metadata['anchor_probs'], dtype=torch.float32)
                q_anchors = torch.empty(np.prod(anchor_shape), dtype=torch.uint8)
                
                if use_cuda and torch.cuda.is_available():
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    self.cuda_module.decode_arithmetic(
                        anchor_compressed, anchor_probs.cuda(),
                        c_void_p(q_anchors.data_ptr()),
                        len(anchor_compressed), len(anchor_probs),
                        q_anchors.numel(), cuda_stream
                    )
                else:
                    q_anchors = torch.frombuffer(anchor_compressed, dtype=torch.uint8)
                
                q_anchors = q_anchors.reshape(anchor_shape)
                anchors = self.dequantize_tensor(q_anchors, anchor_metadata)
                
                # Decode deltas
                delta_shape = (metadata['original_shape'][0],
                             metadata['original_shape'][1],
                             metadata['original_shape'][2] - 1,
                             anchor_shape[3])
                
                delta_probs = torch.tensor(metadata['delta_probs'], dtype=torch.float32)
                q_deltas = torch.empty(np.prod(delta_shape), dtype=torch.uint8)
                
                if use_cuda and torch.cuda.is_available():
                    self.cuda_module.decode_arithmetic(
                        delta_compressed, delta_probs.cuda(),
                        c_void_p(q_deltas.data_ptr()),
                        len(delta_compressed), len(delta_probs),
                        q_deltas.numel(), cuda_stream
                    )
                else:
                    q_deltas = torch.frombuffer(delta_compressed, dtype=torch.uint8)
                
                q_deltas = q_deltas.reshape(delta_shape)
                deltas = self.dequantize_tensor(q_deltas, delta_metadata)
                
                # Reconstruct from deltas
                reconstructed = self.reconstruct_from_deltas(anchors, deltas)
                
            else:
                # Standard decoding without delta
                quant_metadata = metadata['quant_metadata']
                probs = torch.tensor(metadata['probs'], dtype=torch.float32)
                output_size = np.prod(metadata['original_shape'])
                
                output = torch.empty(output_size, dtype=torch.uint8)
                
                if use_cuda and torch.cuda.is_available():
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    self.cuda_module.decode_arithmetic(
                        compressed, probs.cuda(),
                        c_void_p(output.data_ptr()),
                        len(compressed), len(probs),
                        output_size, cuda_stream
                    )
                else:
                    output = torch.frombuffer(compressed, dtype=torch.uint8)
                
                output = output.reshape(metadata['original_shape'])
                reconstructed = self.dequantize_tensor(output, quant_metadata)
            
            # Ungroup channels if needed
            if use_grouping:
                channel_metadata = metadata['channel_metadata']
                # Reshape to grouped format
                batch, num_heads, seq_len, flat_dim = reconstructed.shape
                num_groups = channel_metadata['num_groups']
                group_size = channel_metadata['group_size']
                
                grouped = reconstructed.reshape(batch, num_heads, seq_len, num_groups, group_size)
                reconstructed = self.ungroup_channels(grouped, channel_metadata)
            
            return reconstructed
                
        except Exception as e:
            if isinstance(e, (DecodingError, CUDAError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during tensor decoding: {str(e)}")
    
    def decode_kv_cache(self, 
                       compressed_chunks: List[Tuple[bytes, Dict]], 
                       use_cuda: bool = True, 
                       stream: Optional[torch.cuda.Stream] = None) -> Tuple[torch.Tensor, ...]:
        """Decode full KV cache (all layers) from compressed chunks."""
        try:
            if not isinstance(compressed_chunks, list):
                raise DecodingError("Input must be a list of compressed chunks")
                
            if len(compressed_chunks) % 2 != 0:
                raise DecodingError("Invalid number of chunks: must be even (key-value pairs)")
                
            num_layers = len(compressed_chunks) // 2
            decoded_tensors = []
            
            for layer_idx in range(num_layers):
                try:
                    # Decode key tensor
                    k_compressed, k_metadata = compressed_chunks[layer_idx * 2]
                    if not isinstance(k_metadata, dict) or k_metadata.get('type') != 'key':
                        raise MetadataError(f"Invalid key metadata for layer {layer_idx}")
                    k_tensor = self.decode_tensor(k_compressed, k_metadata, use_cuda, stream)
                    
                    # Decode value tensor
                    v_compressed, v_metadata = compressed_chunks[layer_idx * 2 + 1]
                    if not isinstance(v_metadata, dict) or v_metadata.get('type') != 'value':
                        raise MetadataError(f"Invalid value metadata for layer {layer_idx}")
                    v_tensor = self.decode_tensor(v_compressed, v_metadata, use_cuda, stream)
                    
                    decoded_tensors.append((k_tensor, v_tensor))
                except (DecodingError, CUDAError, MetadataError) as e:
                    raise DecodingError(f"Failed to decode layer {layer_idx}: {str(e)}")
                    
            return tuple(decoded_tensors)
            
        except Exception as e:
            if isinstance(e, (DecodingError, CUDAError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during KV cache decoding: {str(e)}")
    
    def decode_chunk_stream(self, 
                           compressed_stream: bytes, 
                           metadata_list: List[Dict], 
                           use_cuda: bool = True) -> List[Tuple[bytes, Dict]]:
        """Decode a concatenated stream of compressed chunks."""
        try:
            if not isinstance(compressed_stream, bytes):
                raise DecodingError("Input stream must be bytes")
                
            if not isinstance(metadata_list, list):
                raise MetadataError("Metadata must be provided as a list")
            
            chunks = []
            offset = 0
            
            for metadata in metadata_list:
                if not isinstance(metadata, dict):
                    raise MetadataError(f"Invalid metadata format")
                if 'chunk_size' not in metadata:
                    raise MetadataError(f"Missing chunk_size in metadata")
                    
                chunk_size = metadata['chunk_size']
                chunk_data = compressed_stream[offset:offset + chunk_size]
                
                if len(chunk_data) != chunk_size:
                    raise DecodingError(f"Incomplete chunk data: expected {chunk_size} bytes")
                    
                chunks.append((chunk_data, metadata))
                offset += chunk_size
                
            return chunks
                
        except Exception as e:
            if isinstance(e, (DecodingError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during stream decoding: {str(e)}")