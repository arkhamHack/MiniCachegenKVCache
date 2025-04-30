# 🚀 MiniCacheGenKVCache — Streaming & Compressed KV Cache with RoPE from Scratch

> A modern, research-inspired Key/Value caching engine for Transformers — built from scratch with RoPE, streaming, and token-wise compression inspired by [CacheGen (2023)](https://arxiv.org/abs/2310.07240).  

⚡️ Ideal for understanding LLM inference optimizations.  
🎯 Easily pluggable into custom models or demo apps.  
🧠 Educational + performant = blog-worthy!

---

## 🔍 What's inside?

- 🧠 **RoPE-Integrated KV Cache**  
  Rotary positional embeddings applied correctly to `Q` and `K`.

- 💾 **MiniCacheGen Compression**  
  Compresses KV cache using:
  - Anchor + delta token strategy  
  - Quantized per-layer encoding (4/6/8 bit)  
  - Decompression on demand

- 📦 **Streaming Support**  
  Inference-friendly architecture: cache grows with tokens, compressed in chunks.

- 🔬 **Benchmark-Ready**  
  Compare:
  - Uncompressed KV cache vs MiniCacheGenKVCache  
  - Speed & memory use  
  - Quality (optionally)
---

## 🧪 Quickstart

### 1. Clone + install
```bash
git clone https://github.com/your-username/minicachegen-kvcache.git
cd minicachegen-kvcache
pip install -r requirements.txt
