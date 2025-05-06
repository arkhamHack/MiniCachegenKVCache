import os
import pickle
from typing import Any, Dict, List, Optional
from .exceptions import StorageError

def save_chunk(chunk_id: str, encoded_chunk: Any, path: str = "./kv_cache_store/") -> None:
    """Save an encoded chunk to disk.
    
    Args:
        chunk_id: Unique identifier for the chunk
        encoded_chunk: Compressed chunk data to save
        path: Directory path to store chunks
    
    Raises:
        StorageError: If saving fails
    """
    try:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise StorageError("Invalid chunk ID")
            
        os.makedirs(path, exist_ok=True)
        chunk_path = os.path.join(path, f"{chunk_id}.bin")
        
        try:
            with open(chunk_path, "wb") as f:
                pickle.dump(encoded_chunk, f)
        except (IOError, pickle.PickleError) as e:
            raise StorageError(f"Failed to save chunk {chunk_id}: {str(e)}")
            
    except Exception as e:
        if isinstance(e, StorageError):
            raise
        raise StorageError(f"Unexpected error saving chunk: {str(e)}")

def load_chunk(chunk_id: str, path: str = "./kv_cache_store/") -> Any:
    """Load an encoded chunk from disk.
    
    Args:
        chunk_id: Unique identifier for the chunk to load
        path: Directory path where chunks are stored
        
    Returns:
        The loaded chunk data
        
    Raises:
        StorageError: If loading fails
    """
    try:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise StorageError("Invalid chunk ID")
            
        chunk_path = os.path.join(path, f"{chunk_id}.bin")
        
        if not os.path.exists(chunk_path):
            raise StorageError(f"Chunk {chunk_id} not found")
            
        try:
            with open(chunk_path, "rb") as f:
                return pickle.load(f)
        except (IOError, pickle.PickleError) as e:
            raise StorageError(f"Failed to load chunk {chunk_id}: {str(e)}")
            
    except Exception as e:
        if isinstance(e, StorageError):
            raise
        raise StorageError(f"Unexpected error loading chunk: {str(e)}")
