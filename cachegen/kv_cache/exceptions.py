class KVCacheError(Exception):
    """Base exception for KV cache operations."""
    pass

class DecodingError(KVCacheError):
    """Raised when decoding fails."""
    pass

class EncodingError(KVCacheError):
    """Raised when encoding fails."""
    pass

class ChunkError(KVCacheError):
    """Raised when chunk operations fail."""
    pass

class CUDAError(KVCacheError):
    """Raised when CUDA operations fail."""
    def __init__(self, message: str, cuda_error: Exception = None):
        super().__init__(message)
        self.cuda_error = cuda_error

class PipelineError(KVCacheError):
    """Raised when pipeline operations fail."""
    pass

class MetadataError(KVCacheError):
    """Raised when metadata is invalid or corrupted."""
    pass

class StorageError(KVCacheError):
    """Raised when storage operations fail."""
    pass
