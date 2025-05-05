def calculate_kv(model, tokenizer, prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=0, return_dict_in_generate=True)
    return out["past_key_values"]

def generate_with_kv(model, kv_cache, max_tokens: int):
    input_ids = torch.tensor([[model.config.pad_token_id]]).to(model.device)
    return model.generate(
        input_ids=input_ids,
        past_key_values=kv_cache,
        max_new_tokens=max_tokens,
    )
