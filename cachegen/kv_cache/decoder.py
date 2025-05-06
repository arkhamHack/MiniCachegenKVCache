import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ctypes import c_void_p
from .exceptions import DecodingError, CUDAError, MetadataError

class KVCacheDecoder:
    def __init__(self, num_bits: int = 8):
        """Initialize KV Cache decoder.
        
        Args:
            num_bits: Number of bits used in quantization (default: 8)
        """
        self.num_bits = num_bits
        self.cuda_module = None
    
    def _load_cuda_module(self):
        """Load CUDA module for fast decoding if not already loaded."""
        if self.cuda_module is None:
            try:
                from . import cuda_encoder
                self.cuda_module = cuda_encoder
            except Exception as e:
                raise CUDAError("Failed to load CUDA decoder module", e)
    
    def dequantize_tensor(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Dequantize tensor values back to original range."""
        try:
            if not all(key in metadata for key in ['min_val', 'max_val', 'dtype']):
                raise MetadataError("Missing required metadata fields for dequantization")
                
            min_val = metadata['min_val']
            max_val = metadata['max_val']
            if min_val >= max_val:
                raise DecodingError("Invalid min/max values in metadata")
                
            scale = (2**self.num_bits - 1) / (max_val - min_val)
            
            try:
                dequantized = quantized.float() / scale + min_val
                return dequantized.to(dtype=torch.dtype(metadata['dtype']))
            except Exception as e:
                raise DecodingError(f"Failed to dequantize tensor: {str(e)}")
        except Exception as e:
            if isinstance(e, (DecodingError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during dequantization: {str(e)}")
    
    def decode_tensor(self, compressed: bytes, metadata: Dict, use_cuda: bool = True, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Decode compressed bitstream back to tensor using arithmetic coding.
        
        Args:
            compressed: Compressed bitstream
            metadata: Metadata containing shape, dtype, and probability info
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Decoded tensor
        """
        try:
            if not isinstance(compressed, bytes):
                raise DecodingError("Input must be a bytes object")
                
            if not all(key in metadata for key in ['probs', 'shape']):
                raise MetadataError("Missing required metadata fields for decoding")
                
            try:
                probs = torch.tensor(metadata['probs'], dtype=torch.float32)
                output_size = np.prod(metadata['shape'])
            except Exception as e:
                raise MetadataError(f"Invalid metadata format: {str(e)}")
            
            # Prepare output buffer
            try:
                output = torch.empty(output_size, dtype=torch.uint8)
            except Exception as e:
                raise DecodingError(f"Failed to allocate output buffer: {str(e)}")
            
            if use_cuda and torch.cuda.is_available():
                try:
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    self.cuda_module.decode_arithmetic(
                        compressed,
                        probs.cuda(),
                        c_void_p(output.data_ptr()),
                        len(compressed),
                        len(probs),
                        output_size,
                        cuda_stream
                    )
                except Exception as e:
                    raise CUDAError("CUDA decoding failed", e)
            else:
                try:
                    output = torch.frombuffer(compressed, dtype=torch.uint8)
                except Exception as e:
                    raise DecodingError(f"CPU decoding failed: {str(e)}")
            
            try:
                # Reshape and dequantize
                output = output.reshape(metadata['shape'])
                return self.dequantize_tensor(output, metadata)
            except Exception as e:
                raise DecodingError(f"Failed to reshape or dequantize output: {str(e)}")
                
        except Exception as e:
            if isinstance(e, (DecodingError, CUDAError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during tensor decoding: {str(e)}")
    
    def decode_kv_cache(self, compressed_chunks: List[Tuple[bytes, Dict]], use_cuda: bool = True, stream: Optional[torch.cuda.Stream] = None) -> Tuple[torch.Tensor, ...]:
        """Decode full KV cache (all layers) from compressed chunks.
        
        Args:
            compressed_chunks: List of (compressed bitstream, metadata) tuples
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Tuple of (key, value) tensors for each layer
        """
        try:
            if not isinstance(compressed_chunks, list):
                raise DecodingError("Input must be a list of compressed chunks")
                
            if len(compressed_chunks) % 2 != 0:
                raise DecodingError("Invalid number of chunks: must be even (key-value pairs)")
                
            num_layers = len(compressed_chunks) // 2  # Each layer has K and V
            decoded_tensors = []
            
            for layer_idx in range(num_layers):
                try:
                    # Decode key tensor
                    k_compressed, k_metadata = compressed_chunks[layer_idx * 2]
                    if not isinstance(k_metadata, dict) or 'type' not in k_metadata or k_metadata['type'] != 'key':
                        raise MetadataError(f"Invalid key metadata for layer {layer_idx}")
                    k_tensor = self.decode_tensor(k_compressed, k_metadata, use_cuda, stream)
                    
                    # Decode value tensor
                    v_compressed, v_metadata = compressed_chunks[layer_idx * 2 + 1]
                    if not isinstance(v_metadata, dict) or 'type' not in v_metadata or v_metadata['type'] != 'value':
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
    
    def decode_chunk_stream(self, compressed_stream: bytes, metadata_list: List[Dict], use_cuda: bool = True) -> List[Tuple[bytes, Dict]]:
        """Decode a concatenated stream of compressed chunks.
        
        This method is used when receiving a stream of compressed chunks,
        typically from network transfer or disk storage.
        
        Args:
            compressed_stream: Concatenated compressed chunks
            metadata_list: List of metadata for each chunk
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            List of (decompressed chunk, metadata) tuples
        """
        try:
            if not isinstance(compressed_stream, bytes):
                raise DecodingError("Input stream must be bytes")
                
            if not isinstance(metadata_list, list):
                raise MetadataError("Metadata must be provided as a list")
                
            # Verify metadata format
            for i, metadata in enumerate(metadata_list):
                if not isinstance(metadata, dict):
                    raise MetadataError(f"Invalid metadata format at index {i}")
                if 'chunk_size' not in metadata:
                    raise MetadataError(f"Missing chunk_size in metadata at index {i}")
                    
            # Process stream
            try:
                chunks = []
                offset = 0
                
                for metadata in metadata_list:
                    chunk_size = metadata['chunk_size']
                    chunk_data = compressed_stream[offset:offset + chunk_size]
                    
                    if len(chunk_data) != chunk_size:
                        raise DecodingError(f"Incomplete chunk data: expected {chunk_size} bytes")
                        
                    chunks.append((chunk_data, metadata))
                    offset += chunk_size
                    
                return chunks
                
            except Exception as e:
                raise DecodingError(f"Failed to process chunk stream: {str(e)}")
                
        except Exception as e:
            if isinstance(e, (DecodingError, MetadataError)):
                raise
            raise DecodingError(f"Unexpected error during stream decoding: {str(e)}")


        offset = 0
        
        for metadata in metadata_list:
            size = metadata['compressed_size']
            chunk = compressed_stream[offset:offset + size]
            result.append((chunk, metadata))
            offset += size
            
        return result
