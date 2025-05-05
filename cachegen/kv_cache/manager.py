from kv_cache.chunking import split_kv_cache
from kv_cache.compressor import compress_chunk, decompress_chunk
from kv_cache.storage import save_chunk, load_chunk
from utils.hashing import get_chunk_ids

def store_kv(model, tokenizer, prompt: str):
    kv = calculate_kv(model, tokenizer, prompt)
    chunks = split_kv_cache(kv)
    chunk_map = {}
    for i,chunk in enumerate(chunks):
        chunk_id = get_chunk_ids(prompt,i)
        encoded = compress_chunk(i)
        save_chunk(chunk_id, encoded)
        chunk_map[chunk_id]=encoded
    return chunk_map


def get_kv(prompt: str, chunk_index: int):
    chunk_id = get_chunk_ids(prompt, chunk_index)
    encoded = load_chunk(chunk_id)
    return decompress_chunk(encoded)

    
