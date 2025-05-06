from typing import Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .chunking import KVChunkManager, ChunkConfig
from .storage import save_chunk, load_chunk
from .exceptions import KVCacheError, ChunkError, EncodingError, StorageError

class KVCacheManager:
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize KV cache manager.
        
        Args:
            config: Optional chunking configuration. If None, default config is used.
        """
        self.config = config or ChunkConfig(
            max_chunk_size=512,
            min_chunk_size=64,
            chunk_overlap=16,
            max_chunks_in_memory=1000,
            num_bits=8
        )
        self.chunk_manager = KVChunkManager(self.config)
        
    def store_kv(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer, 
                 prompt: str,
                 save_path: str = "./kv_cache_store/") -> List[str]:
        """Calculate and store KV cache for a prompt.
        
        Args:
            model: The language model
            tokenizer: The model's tokenizer
            prompt: Input text to generate KV cache for
            save_path: Path to save chunks
            
        Returns:
            List of chunk IDs for the stored chunks
            
        Raises:
            KVCacheError: If KV cache calculation or storage fails
        """
        try:
            # Encode input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate KV cache
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False
                )
                kv_cache = outputs.past_key_values
                
            if not kv_cache:
                raise KVCacheError("Model did not return KV cache")

            # Split into chunks
            try:
                chunks = self.chunk_manager.split_kv_cache(kv_cache, prompt)
            except ChunkError as e:
                raise KVCacheError(f"Failed to split KV cache: {str(e)}")

            # Store chunks
            chunk_ids = []
            for chunk in chunks:
                try:
                    chunk_id = chunk['chunk_id']
                    save_chunk(chunk_id, chunk, save_path)
                    chunk_ids.append(chunk_id)
                except StorageError as e:
                    raise KVCacheError(f"Failed to save chunk {chunk_id}: {str(e)}")

            return chunk_ids
            
        except Exception as e:
            if isinstance(e, KVCacheError):
                raise
            raise KVCacheError(f"Unexpected error in store_kv: {str(e)}")
            
    def get_kv(self, 
               chunk_ids: List[str],
               load_path: str = "./kv_cache_store/") -> Optional[Tuple[torch.Tensor, ...]]:
        """Load and merge KV cache chunks.
        
        Args:
            chunk_ids: List of chunk IDs to load
            load_path: Path where chunks are stored
            
        Returns:
            Merged KV cache tensors, or None if chunks not found
            
        Raises:
            KVCacheError: If loading or merging fails
        """
        try:
            chunks = []
            for chunk_id in chunk_ids:
                try:
                    # Try memory cache first
                    chunk = self.chunk_manager.get_cached_chunk(chunk_id)
                    if chunk is None:
                        # Load from disk if not in memory
                        try:
                            chunk = load_chunk(chunk_id, load_path)
                            self.chunk_manager.cache_chunk(chunk_id, chunk)
                        except StorageError as e:
                            raise KVCacheError(f"Failed to load chunk {chunk_id}: {str(e)}")
                    chunks.append(chunk)
                except Exception as e:
                    raise KVCacheError(f"Error processing chunk {chunk_id}: {str(e)}")

            if not chunks:
                return None

            try:
                return self.chunk_manager.merge_chunks(chunks)
            except ChunkError as e:
                raise KVCacheError(f"Failed to merge chunks: {str(e)}")
                
        except Exception as e:
            if isinstance(e, KVCacheError):
                raise
            raise KVCacheError(f"Unexpected error in get_kv: {str(e)}")
