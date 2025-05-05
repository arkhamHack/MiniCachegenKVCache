def save_chunk(chunk_id, encoded_chunk, path="./kv_cache_store/"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{chunk_id}.bin"), "wb") as f:
        pickle.dump(encoded_chunk, f)

def load_chunk(chunk_id, path="./kv_cache_store/"):
    with open(os.path.join(path, f"{chunk_id}.bin"), "rb") as f:
        return pickle.load(f)
