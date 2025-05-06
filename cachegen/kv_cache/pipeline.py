import asyncio
import torch
import torch.cuda
from typing import Dict, List, Tuple, Optional, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from .decoder import KVCacheDecoder
from .encoder import KVCacheEncoder
from .storage import load_chunk, save_chunk

@dataclass
class PipelineConfig:
    num_cuda_streams: int = 4
    prefetch_chunks: int = 2
    max_concurrent_io: int = 4

class KVCachePipeline:
    def __init__(self, config: PipelineConfig):
        """Initialize the KV cache pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.decoder = KVCacheDecoder()
        self.encoder = KVCacheEncoder()
        self._cuda_streams = [
            torch.cuda.Stream() for _ in range(config.num_cuda_streams)
        ]
        self._io_executor = ThreadPoolExecutor(max_workers=config.max_concurrent_io)
        self._current_stream = 0
    
    def _next_stream(self) -> torch.cuda.Stream:
        """Get next CUDA stream in round-robin fashion."""
        stream = self._cuda_streams[self._current_stream]
        self._current_stream = (self._current_stream + 1) % len(self._cuda_streams)
        return stream
    
    async def _load_chunk_async(self, chunk_id: str, path: str) -> Tuple[bytes, List[Dict]]:
        """Asynchronously load a chunk from storage."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._io_executor,
            load_chunk,
            chunk_id,
            path
        )
    
    async def _save_chunk_async(self, chunk_id: str, data: Tuple[bytes, List[Dict]], path: str):
        """Asynchronously save a chunk to storage."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._io_executor,
            save_chunk,
            chunk_id,
            data,
            path
        )
    
    async def decode_chunks_pipelined(
        self,
        chunk_ids: List[str],
        storage_path: str
    ) -> AsyncIterator[Tuple[str, Tuple[torch.Tensor, ...]]]:
        """Decode chunks with pipelined I/O and computation.
        
        This method overlaps:
        1. I/O: Loading compressed chunks from storage
        2. H2D: Transferring compressed data to GPU
        3. Compute: CUDA decoding kernels
        4. D2H: Transferring decoded tensors back to CPU
        
        Args:
            chunk_ids: List of chunk IDs to decode
            storage_path: Path to chunk storage
            
        Yields:
            Tuples of (chunk_id, decoded_tensors)
        """
        # Prefetch queue for loading chunks
        load_tasks = []
        for chunk_id in chunk_ids[:self.config.prefetch_chunks]:
            task = asyncio.create_task(self._load_chunk_async(chunk_id, storage_path))
            load_tasks.append((chunk_id, task))
        
        next_chunk_idx = self.config.prefetch_chunks
        
        # Process chunks in pipeline
        while load_tasks:
            # Start loading next chunk if available
            if next_chunk_idx < len(chunk_ids):
                chunk_id = chunk_ids[next_chunk_idx]
                task = asyncio.create_task(self._load_chunk_async(chunk_id, storage_path))
                load_tasks.append((chunk_id, task))
                next_chunk_idx += 1
            
            # Wait for the earliest chunk to load
            chunk_id, load_task = load_tasks.pop(0)
            compressed_data, metadata = await load_task
            
            # Get next CUDA stream
            stream = self._next_stream()
            
            # Process in dedicated CUDA stream
            with torch.cuda.stream(stream):
                # Decode chunk
                decoded_tensors = self.decoder.decode_kv_cache(
                    [(compressed_data, meta) for meta in metadata],
                    use_cuda=True
                )
                
                # Ensure tensors are moved to CPU before yielding
                cpu_tensors = tuple(
                    (k.cpu(), v.cpu()) for k, v in decoded_tensors
                )
            
            # Yield results as they become available
            yield chunk_id, cpu_tensors
    
    async def encode_chunks_pipelined(
        self,
        chunks: List[Tuple[str, Tuple[torch.Tensor, ...]]],
        storage_path: str
    ):
        """Encode chunks with pipelined computation and I/O.
        
        This method overlaps:
        1. H2D: Transferring tensors to GPU
        2. Compute: CUDA encoding kernels
        3. D2H: Transferring compressed data to CPU
        4. I/O: Saving compressed chunks to storage
        
        Args:
            chunks: List of (chunk_id, tensors) tuples to encode
            storage_path: Path to chunk storage
        """
        save_tasks = []
        
        for chunk_id, tensors in chunks:
            # Get next CUDA stream
            stream = self._next_stream()
            
            # Process in dedicated CUDA stream
            with torch.cuda.stream(stream):
                # Move tensors to GPU and encode
                gpu_tensors = tuple(
                    (k.cuda(), v.cuda()) for k, v in tensors
                )
                compressed_chunks = self.encoder.encode_kv_cache(gpu_tensors, use_cuda=True)
                
                # Concatenate compressed data
                compressed_data = b''.join(data for data, _ in compressed_chunks)
                metadata = [meta for _, meta in compressed_chunks]
            
            # Save asynchronously
            task = asyncio.create_task(
                self._save_chunk_async(chunk_id, (compressed_data, metadata), storage_path)
            )
            save_tasks.append(task)
        
        # Wait for all saves to complete
        await asyncio.gather(*save_tasks)
