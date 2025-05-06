import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ctypes import c_void_p
from .exceptions import EncodingError, CUDAError

class KVCacheEncoder:
    def __init__(self, num_bits: int = 8):
        """Initialize KV Cache encoder with quantization bits.
        
        Args:
            num_bits: Number of bits for quantization (default: 8)
        """
        self.num_bits = num_bits
        self.cuda_module = None  # Will be loaded when CUDA encoding is needed
        
    def _load_cuda_module(self):
        """Load CUDA module for fast encoding if not already loaded."""
        if self.cuda_module is None:
            try:
                from . import cuda_encoder
                self.cuda_module = cuda_encoder
            except Exception as e:
                raise CUDAError("Failed to load CUDA encoder module", e)
    
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor values to specified number of bits."""
        try:
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val == max_val:
                raise EncodingError("Cannot quantize tensor with identical min and max values")
                
            scale = (2**self.num_bits - 1) / (max_val - min_val)
            quantized = ((tensor - min_val) * scale).round().clamp(0, 2**self.num_bits - 1)
            return quantized.to(torch.uint8)
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to quantize tensor: {str(e)}")
    
    def compute_probabilities(self, quantized: torch.Tensor) -> torch.Tensor:
        """Compute probability distribution for arithmetic coding."""
        try:
            freqs = torch.bincount(quantized.view(-1), minlength=2**self.num_bits)
            if freqs.sum() == 0:
                raise EncodingError("Empty tensor - cannot compute probabilities")
            probs = freqs.float() / freqs.sum()
            return probs
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to compute probabilities: {str(e)}")
    
    def encode_tensor(self, tensor: torch.Tensor, use_cuda: bool = True, stream: Optional[torch.cuda.Stream] = None) -> Tuple[bytes, Dict]:
        """Encode tensor to compressed bitstream using arithmetic coding.
        
        Args:
            tensor: Input tensor to encode
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Tuple of (compressed bitstream, metadata for decoding)
        """
        try:
            if not isinstance(tensor, torch.Tensor):
                raise EncodingError("Input must be a PyTorch tensor")
                
            quantized = self.quantize_tensor(tensor)
            probs = self.compute_probabilities(quantized)
            
            metadata = {
                'shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'min_val': tensor.min().item(),
                'max_val': tensor.max().item(),
                'probs': probs.cpu().numpy().tolist()
            }
            
            if use_cuda and tensor.is_cuda:
                try:
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    compressed = self.cuda_module.encode_arithmetic(
                        quantized.cuda(),
                        probs.cuda(),
                        c_void_p(quantized.data_ptr()),
                        len(quantized),
                        len(probs),
                        cuda_stream
                    )
                except Exception as e:
                    raise CUDAError("CUDA encoding failed", e)
            else:
                try:
                    compressed = quantized.cpu().numpy().tobytes()
                except Exception as e:
                    raise EncodingError(f"CPU encoding failed: {str(e)}")
                
            return compressed, metadata
            
        except Exception as e:
            if isinstance(e, (EncodingError, CUDAError)):
                raise
            raise EncodingError(f"Unexpected error during tensor encoding: {str(e)}")
    
    def encode_kv_cache(self, kv_cache: Tuple[torch.Tensor, ...], use_cuda: bool = True, stream: Optional[torch.cuda.Stream] = None) -> List[Tuple[bytes, Dict]]:
        """Encode full KV cache (all layers) to compressed bitstreams.
        
        Args:
            kv_cache: Tuple of (key, value) tensors for each layer
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            List of (compressed bitstream, metadata) tuples for each tensor
        """
        if not isinstance(kv_cache, tuple):
            raise EncodingError("KV cache must be a tuple of tensors")
            
        encoded_chunks = []
        
        try:
            for layer_idx, (k, v) in enumerate(kv_cache):
                try:
                    # Encode key tensor
                    k_compressed, k_metadata = self.encode_tensor(k, use_cuda, stream)
                    k_metadata['layer'] = layer_idx
                    k_metadata['type'] = 'key'
                    
                    # Encode value tensor
                    v_compressed, v_metadata = self.encode_tensor(v, use_cuda, stream)
                    v_metadata['layer'] = layer_idx
                    v_metadata['type'] = 'value'
                    
                    encoded_chunks.extend([
                        (k_compressed, k_metadata),
                        (v_compressed, v_metadata)
                    ])
                except (EncodingError, CUDAError) as e:
                    raise EncodingError(f"Failed to encode layer {layer_idx}: {str(e)}")
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to encode KV cache: {str(e)}")
            
        return encoded_chunks

        
        return encoded_chunks
