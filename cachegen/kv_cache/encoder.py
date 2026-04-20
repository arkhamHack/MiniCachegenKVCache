import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ctypes import c_void_p
from .exceptions import EncodingError, CUDAError
from . import cuda_helper

class KVCacheEncoder:
    def __init__(self, 
                 num_bits: int = 8,
                 num_layers: int = 32,
                 use_delta_encoding: bool = True,
                 use_layer_wise_quant: bool = True,
                 use_channel_grouping: bool = True,
                 channel_group_size: int = 16):
        """Initialize KV Cache encoder with CacheGen features.
        
        Args:
            num_bits: Base number of bits for quantization (default: 8)
            num_layers: Total number of transformer layers
            use_delta_encoding: Enable token-wise locality via delta encoding
            use_layer_wise_quant: Enable layer-wise sensitivity quantization
            use_channel_grouping: Enable channel grouping
            channel_group_size: Size of channel groups
        """
        self.num_bits = num_bits
        self.num_layers = num_layers
        self.use_delta_encoding = use_delta_encoding
        self.use_layer_wise_quant = use_layer_wise_quant
        self.use_channel_grouping = use_channel_grouping
        self.channel_group_size = channel_group_size
        self.cuda_module = None
        
        # Compute layer-wise quantization bits
        self.layer_bits = self._compute_layer_bits()
        
    def _compute_layer_bits(self) -> List[int]:
        """
        Compute bits per layer based on sensitivity.
        Earlier layers (more sensitive) get more bits.
        """
        bits_per_layer = []
        
        for layer_idx in range(self.num_layers):
            # Earlier layers are more sensitive to quantization loss
            if layer_idx < self.num_layers * 0.25:  # First 25%
                bits = self.num_bits + 2  # e.g., 10 bits
            elif layer_idx < self.num_layers * 0.5:  # 25-50%
                bits = self.num_bits + 1  # e.g., 9 bits
            elif layer_idx < self.num_layers * 0.75:  # 50-75%
                bits = self.num_bits      # e.g., 8 bits
            else:  # Last 25%
                bits = max(4, self.num_bits - 2)  # e.g., 6 bits (more aggressive)
                
            bits_per_layer.append(bits)
            
        return bits_per_layer
        
    def _load_cuda_module(self):
        """Load CUDA module for fast encoding if not already loaded."""
        if self.cuda_module is None:
            self.cuda_module = cuda_helper
    
    def compute_deltas(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute deltas between consecutive tokens (token-wise locality).
        
        Args:
            tensor: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            anchors: First token values (absolute)
            deltas: Differences from previous token
        """
        try:
            if tensor.dim() != 4:
                raise EncodingError(f"Expected 4D tensor, got {tensor.dim()}D")
                
            # First token is anchor (absolute value)
            anchors = tensor[:, :, 0:1, :]  # [batch, num_heads, 1, head_dim]
            
            # Compute deltas for remaining tokens
            if tensor.size(2) > 1:
                deltas = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            else:
                deltas = torch.zeros_like(tensor[:, :, 0:0, :])
                
            return anchors, deltas
            
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to compute deltas: {str(e)}")
    
    def group_channels(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Group channels for better compression (channel-wise locality).
        
        Args:
            tensor: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            grouped_tensor: Reshaped and normalized
            group_metadata: Statistics per group
        """
        try:
            batch, num_heads, seq_len, head_dim = tensor.shape
            
            # Pad if needed
            pad_size = 0
            if head_dim % self.channel_group_size != 0:
                pad_size = self.channel_group_size - (head_dim % self.channel_group_size)
                tensor = torch.nn.functional.pad(tensor, (0, pad_size))
                head_dim = tensor.shape[-1]
            
            num_groups = head_dim // self.channel_group_size
            
            # Reshape: [batch, num_heads, seq_len, num_groups, group_size]
            grouped = tensor.reshape(batch, num_heads, seq_len, num_groups, self.channel_group_size)
            
            # Compute per-group statistics for normalization
            group_means = grouped.mean(dim=-1, keepdim=True)  # [batch, heads, seq_len, num_groups, 1]
            group_stds = grouped.std(dim=-1, keepdim=True)
            group_stds = torch.where(group_stds > 1e-6, group_stds, torch.ones_like(group_stds))
            
            # Normalize each group
            normalized = (grouped - group_means) / group_stds
            
            metadata = {
                'group_means': group_means.squeeze(-1),
                'group_stds': group_stds.squeeze(-1),
                'num_groups': num_groups,
                'group_size': self.channel_group_size,
                'original_head_dim': head_dim - pad_size,
                'pad_size': pad_size
            }
            
            return normalized, metadata
            
        except Exception as e:
            raise EncodingError(f"Failed to group channels: {str(e)}")
    
    def quantize_tensor(self, 
                       tensor: torch.Tensor, 
                       layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize tensor with layer-wise bit allocation.
        
        Args:
            tensor: Input tensor
            layer_idx: Layer index for layer-wise quantization
            
        Returns:
            quantized tensor and quantization metadata
        """
        try:
            # Determine number of bits based on layer
            if self.use_layer_wise_quant and layer_idx is not None:
                bits = self.layer_bits[layer_idx]
            else:
                bits = self.num_bits
            
            min_val = tensor.min()
            max_val = tensor.max()
            
            if min_val == max_val:
                # Handle constant tensor
                quantized = torch.zeros_like(tensor, dtype=torch.uint8)
                metadata = {
                    'min_val': min_val.item(),
                    'max_val': max_val.item(),
                    'bits': bits,
                    'is_constant': True
                }
                return quantized, metadata
            
            # Quantize
            max_quant_val = 2**bits - 1
            scale = max_quant_val / (max_val - min_val)
            quantized = ((tensor - min_val) * scale).round().clamp(0, max_quant_val)
            
            # Use appropriate dtype based on bits
            if bits <= 8:
                quantized = quantized.to(torch.uint8)
            else:
                quantized = quantized.to(torch.int16)
            
            metadata = {
                'min_val': min_val.item(),
                'max_val': max_val.item(),
                'bits': bits,
                'scale': scale.item(),
                'is_constant': False
            }
            
            return quantized, metadata
            
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to quantize tensor: {str(e)}")
    
    def compute_probabilities(self, quantized: torch.Tensor, bits: int) -> torch.Tensor:
        """Compute probability distribution for arithmetic coding."""
        try:
            num_symbols = 2**bits
            freqs = torch.bincount(quantized.view(-1).long(), minlength=num_symbols)
            if freqs.sum() == 0:
                raise EncodingError("Empty tensor - cannot compute probabilities")
            probs = freqs.float() / freqs.sum()
            return probs
        except Exception as e:
            if isinstance(e, EncodingError):
                raise
            raise EncodingError(f"Failed to compute probabilities: {str(e)}")
    
    def encode_tensor(self, 
                     tensor: torch.Tensor, 
                     layer_idx: Optional[int] = None,
                     use_cuda: bool = True, 
                     stream: Optional[torch.cuda.Stream] = None) -> Tuple[bytes, Dict]:
        """
        Encode tensor with full CacheGen pipeline:
        1. Delta encoding (if enabled)
        2. Channel grouping (if enabled)  
        3. Layer-wise quantization
        4. Arithmetic coding
        
        Args:
            tensor: Input tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index for layer-wise quantization
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            Tuple of (compressed bitstream, metadata for decoding)
        """
        try:
            if not isinstance(tensor, torch.Tensor):
                raise EncodingError("Input must be a PyTorch tensor")
            
            metadata = {
                'original_shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'layer_idx': layer_idx,
                'use_delta_encoding': self.use_delta_encoding,
                'use_channel_grouping': self.use_channel_grouping
            }
            
            # Step 1: Channel grouping (if enabled)
            if self.use_channel_grouping and tensor.dim() == 4:
                tensor, group_metadata = self.group_channels(tensor)
                metadata['channel_metadata'] = group_metadata
                # Flatten for subsequent processing
                tensor = tensor.reshape(*tensor.shape[:3], -1)
            
            # Step 2: Delta encoding (if enabled)
            if self.use_delta_encoding and tensor.dim() == 4 and tensor.size(2) > 1:
                anchors, deltas = self.compute_deltas(tensor)
                
                # Quantize anchors
                q_anchors, anchor_metadata = self.quantize_tensor(anchors, layer_idx)
                
                # Quantize deltas
                q_deltas, delta_metadata = self.quantize_tensor(deltas, layer_idx)
                
                # Encode both
                anchor_probs = self.compute_probabilities(q_anchors, anchor_metadata['bits'])
                delta_probs = self.compute_probabilities(q_deltas, delta_metadata['bits'])
                
                if use_cuda and tensor.is_cuda:
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    
                    anchor_compressed = self.cuda_module.encode_arithmetic(
                        q_anchors.cuda(), anchor_probs.cuda(),
                        c_void_p(q_anchors.data_ptr()),
                        q_anchors.numel(), len(anchor_probs), cuda_stream
                    )
                    
                    delta_compressed = self.cuda_module.encode_arithmetic(
                        q_deltas.cuda(), delta_probs.cuda(),
                        c_void_p(q_deltas.data_ptr()),
                        q_deltas.numel(), len(delta_probs), cuda_stream
                    )
                else:
                    anchor_compressed = q_anchors.cpu().numpy().tobytes()
                    delta_compressed = q_deltas.cpu().numpy().tobytes()
                
                metadata.update({
                    'anchor_metadata': anchor_metadata,
                    'delta_metadata': delta_metadata,
                    'anchor_probs': anchor_probs.cpu().numpy().tolist(),
                    'delta_probs': delta_probs.cpu().numpy().tolist(),
                    'anchor_size': len(anchor_compressed),
                    'delta_size': len(delta_compressed)
                })
                
                # Concatenate compressed data
                compressed = anchor_compressed + delta_compressed
                
            else:
                # Standard encoding without delta
                quantized, quant_metadata = self.quantize_tensor(tensor, layer_idx)
                probs = self.compute_probabilities(quantized, quant_metadata['bits'])
                
                if use_cuda and tensor.is_cuda:
                    self._load_cuda_module()
                    cuda_stream = stream.cuda_stream if stream else 0
                    compressed = self.cuda_module.encode_arithmetic(
                        quantized.cuda(), probs.cuda(),
                        c_void_p(quantized.data_ptr()),
                        quantized.numel(), len(probs), cuda_stream
                    )
                else:
                    compressed = quantized.cpu().numpy().tobytes()
                
                metadata.update({
                    'quant_metadata': quant_metadata,
                    'probs': probs.cpu().numpy().tolist()
                })
                
            return compressed, metadata
            
        except Exception as e:
            if isinstance(e, (EncodingError, CUDAError)):
                raise
            raise EncodingError(f"Unexpected error during tensor encoding: {str(e)}")
    
    def encode_kv_cache(self, 
                       kv_cache: Tuple[torch.Tensor, ...], 
                       use_cuda: bool = True, 
                       stream: Optional[torch.cuda.Stream] = None) -> List[Tuple[bytes, Dict]]:
        """
        Encode full KV cache with layer-wise strategies.
        
        Args:
            kv_cache: Tuple of (key, value) tensors for each layer
            use_cuda: Whether to use CUDA acceleration
            
        Returns:
            List of (compressed bitstream, metadata) tuples
        """
        if not isinstance(kv_cache, tuple):
            raise EncodingError("KV cache must be a tuple of tensors")
            
        encoded_chunks = []
        
        try:
            for layer_idx, (k, v) in enumerate(kv_cache):
                try:
                    # Encode key tensor with layer-specific quantization
                    k_compressed, k_metadata = self.encode_tensor(k, layer_idx, use_cuda, stream)
                    k_metadata['type'] = 'key'
                    
                    # Encode value tensor with layer-specific quantization
                    v_compressed, v_metadata = self.encode_tensor(v, layer_idx, use_cuda, stream)
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
    
    def get_compression_stats(self, 
                             original_cache: Tuple[torch.Tensor, ...],
                             encoded_chunks: List[Tuple[bytes, Dict]]) -> Dict:
        """
        Compute compression statistics.
        """
        # Original size
        original_bytes = sum(k.element_size() * k.numel() + v.element_size() * v.numel() 
                            for k, v in original_cache)
        
        # Compressed size
        compressed_bytes = sum(len(chunk) for chunk, _ in encoded_chunks)
        
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
        
        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': ratio,
            'space_saved_percent': (1 - 1/ratio) * 100 if ratio > 0 else 0,
            'original_mb': original_bytes / (1024**2),
            'compressed_mb': compressed_bytes / (1024**2),
            'layer_bits': self.layer_bits
        }