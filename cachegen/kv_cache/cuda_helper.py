"""
cachegen/kv_cache/cuda_helper.py

KV-cache arithmetic coding helper.

* When the compiled CUDA extension ``_cuda_helper_ext`` is available it is used
  for GPU-accelerated range-coding.
* Otherwise encoding/decoding fall back to **zlib**, which is lossless,
  guaranteed to roundtrip, and requires no native build step.

To build the optional CUDA extension (for GPU acceleration) wrap cuda_helper.cu
with pybind11 and name the resulting module ``_cuda_helper_ext``.  See
cuda_helper.cu for per-function notes on the required interface changes.
"""

from __future__ import annotations

import ctypes
import zlib
from ctypes import c_void_p

import numpy as np
import torch

# ── optional compiled CUDA extension ─────────────────────────────────────────
_cuda_ext = None
try:
    from . import _cuda_helper_ext as _cuda_ext   # compiled pybind11 extension
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API (mirrors the extern-C signatures in cuda_helper.cu)
# ─────────────────────────────────────────────────────────────────────────────

def encode_arithmetic(
    quantized: torch.Tensor,
    probs: torch.Tensor,
    output_ptr: c_void_p,
    numel: int,
    num_symbols: int,
    cuda_stream: int = 0,
) -> bytes:
    """Compress *quantized* symbols to a compact byte string.

    GPU path  – delegates to the compiled CUDA extension (range-coder).
    CPU path  – lossless zlib of the raw tensor bytes.  The ``probs`` argument
                is preserved in metadata for future GPU use but is not consumed
                by the CPU compressor.

    Returns:
        Compressed bytes suitable for ``decode_arithmetic``.
    """
    if _cuda_ext is not None:
        return _cuda_ext.encode_arithmetic(
            quantized, probs, output_ptr, numel, num_symbols, cuda_stream
        )

    # CPU fallback: flatten to raw bytes and zlib-compress.
    # .view(np.uint8) reinterprets any dtype (uint8, int16, …) as raw bytes so
    # the decoder can reconstruct the exact original memory layout.
    raw = quantized.cpu().contiguous().numpy().view(np.uint8).tobytes()
    return zlib.compress(raw, level=6)


def decode_arithmetic(
    compressed: bytes,
    probs: torch.Tensor,
    output_ptr: c_void_p,
    compressed_len: int,
    num_symbols: int,
    output_len: int,
    cuda_stream: int = 0,
) -> None:
    """Decompress *compressed* bytes into the buffer pointed to by *output_ptr*.

    GPU path  – delegates to the compiled CUDA extension.
    CPU path  – zlib decompression written directly into the pre-allocated
                PyTorch tensor via ctypes, so the caller can use the tensor
                immediately after this call returns.
    """
    if _cuda_ext is not None:
        _cuda_ext.decode_arithmetic(
            compressed, probs, output_ptr, compressed_len,
            num_symbols, output_len, cuda_stream
        )
        return

    # CPU fallback ────────────────────────────────────────────────────────────
    raw = np.frombuffer(
        zlib.decompress(compressed[:compressed_len]), dtype=np.uint8
    )
    if raw.size < output_len:
        raise RuntimeError(
            f"Decompressed {raw.size} bytes but expected at least {output_len}"
        )
    # Write into the pre-allocated PyTorch tensor memory without copying the
    # tensor object; ctypes.memmove requires a concrete pointer on the source.
    src = np.ascontiguousarray(raw[:output_len])
    ctypes.memmove(output_ptr, src.ctypes.data_as(ctypes.c_char_p), output_len)
